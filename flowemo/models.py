"""FlowEmo model components.

Modalities are always handled in the order text, audio, vision.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .losses import FocalLoss

MODALITIES = ("text", "audio", "vision")


# --------------------------------------------------------------------------- encoders

def _mlp(in_dim, hidden_dims, out_dim):
    layers = []
    for h in hidden_dims:
        layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.LeakyReLU(0.2), nn.Dropout(0.1)]
        in_dim = h
    layers.append(nn.Linear(in_dim, out_dim))
    return nn.Sequential(*layers)


class MLPEncoder(nn.Module):
    """Vector features -> L2-normalized latent (hidden sizes ``in/2``, ``in/4``)."""

    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.net = _mlp(input_dim, [input_dim // 2, input_dim // 4], latent_dim)

    def forward(self, x, mask=None):
        return F.normalize(self.net(x), dim=1)


class AttentionPoolingEncoder(nn.Module):
    """Sequence ``[B, T, in]`` -> per-step projection -> additive attention pooling -> MLP -> latent."""

    def __init__(self, input_dim, latent_dim):
        super().__init__()
        hidden = [input_dim // 2, input_dim // 4]
        self.proj = nn.Sequential(nn.Linear(input_dim, hidden[0]), nn.LeakyReLU(0.2), nn.Dropout(0.1))
        self.score = nn.Linear(hidden[0], 1)
        self.head = _mlp(hidden[0], hidden[1:], latent_dim)

    def forward(self, x, mask=None):
        h = self.proj(x)
        scores = self.score(h)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        pooled = (F.softmax(scores, dim=1) * h).sum(dim=1)
        return F.normalize(self.head(pooled), dim=1)


class CLSCrossAttentionEncoder(nn.Module):
    """Sequence encoder where a learned ``[CLS]`` query cross-attends over the projected steps."""

    def __init__(self, input_dim, latent_dim, num_heads=8, dropout=0.1):
        super().__init__()
        hidden = [input_dim // 2, input_dim // 4]
        dim = hidden[0]
        if dim % num_heads:
            raise ValueError(f"projected dim {dim} is not divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.proj = nn.Sequential(nn.Linear(input_dim, dim), nn.LeakyReLU(0.2), nn.Dropout(0.1))
        self.cls_token = nn.Parameter(torch.randn(1, dim) * 0.02)
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = (nn.Linear(dim, dim) for _ in range(4))
        self.attn_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        self.head = _mlp(dim, hidden[1:], latent_dim)

    def forward(self, x, mask=None):
        b = x.size(0)
        h = self.proj(x)
        query = self.cls_token.expand(b, -1)

        def split_heads(t):
            return t.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)

        q = split_heads(self.q_proj(query).unsqueeze(1))
        k, v = split_heads(self.k_proj(h)), split_heads(self.v_proj(h))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask[:, None, None, :] == 0, float("-inf"))
        attn = self.attn_dropout(F.softmax(scores, dim=-1))
        attended = self.out_proj((attn @ v).transpose(1, 2).reshape(b, -1))
        return F.normalize(self.head(self.norm(attended + query)), dim=1)


def build_encoder(kind, input_dim, latent_dim, num_heads=8):
    if kind == "mlp":
        return MLPEncoder(input_dim, latent_dim)
    if kind == "attention":
        return AttentionPoolingEncoder(input_dim, latent_dim)
    if kind == "cls":
        return CLSCrossAttentionEncoder(input_dim, latent_dim, num_heads)
    raise ValueError(f"Unknown encoder type: {kind}")


# --------------------------------------------------------------------------- flow matching

def sinusoidal_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(torch.arange(half, device=t.device, dtype=t.dtype) * -(math.log(10000) / (half - 1)))
    args = t[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class FlowMatching(nn.Module):
    """Rectified-flow velocity field ``v(x_t, t [, c])`` on the latent space.

    Training regresses ``v`` onto ``x_1 - x_0`` along the straight path
    ``x_t = (1 - t) x_0 + t x_1``. With ``ot_pairing`` the batch of targets ``x_1`` is first
    re-ordered by a minimum-cost (squared L2) assignment to the sources ``x_0``.
    With ``condition_dim`` the field is also conditioned on a vector (the router weights).
    """

    TIME_EMBED_DIM = 128

    def __init__(self, latent_dim, hidden_dim=1024, num_layers=4, condition_dim=None, ot_pairing=True):
        super().__init__()
        self.ot_pairing = ot_pairing
        self.time_mlp = nn.Sequential(
            nn.Linear(self.TIME_EMBED_DIM, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        in_dim = latent_dim + hidden_dim
        self.condition_mlp = None
        if condition_dim is not None:
            self.condition_mlp = nn.Sequential(
                nn.Linear(condition_dim, hidden_dim // 4), nn.SiLU(), nn.Linear(hidden_dim // 4, hidden_dim)
            )
            in_dim += hidden_dim

        layers = [nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Dropout(0.1)]
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.velocity = nn.Sequential(*layers)
        nn.init.zeros_(self.velocity[-1].weight)
        nn.init.zeros_(self.velocity[-1].bias)

    def forward(self, x_t, t, condition=None):
        parts = [x_t, self.time_mlp(sinusoidal_embedding(t.reshape(-1), self.TIME_EMBED_DIM))]
        if self.condition_mlp is not None:
            if condition is None:
                parts.append(torch.zeros_like(parts[1]))
            else:
                parts.append(self.condition_mlp(condition))
        return self.velocity(torch.cat(parts, dim=-1))

    def loss(self, x_0, x_1, condition=None):
        if self.ot_pairing:
            cost = torch.cdist(x_0, x_1).pow(2).detach().cpu().numpy()
            _, cols = linear_sum_assignment(cost)
            x_1 = x_1[torch.from_numpy(np.asarray(cols)).to(x_1.device)]
        t = torch.rand(x_0.shape[0], device=x_0.device)
        x_t = (1 - t[:, None]) * x_0 + t[:, None] * x_1
        return F.mse_loss(self(x_t, t, condition), x_1 - x_0)

    @torch.no_grad()
    def integrate(self, x, num_steps=10, reverse=False, condition=None, return_path=False):
        """Euler integration from t=0 to 1 (or from 1 back to 0 with ``reverse``)."""
        dt = 1.0 / num_steps
        path = [x]
        for i in range(num_steps):
            t_val = 1.0 - i / num_steps if reverse else i / num_steps
            t = torch.full((x.shape[0],), t_val, device=x.device)
            v = self(x, t, condition)
            x = x - v * dt if reverse else x + v * dt
            path.append(x)
        return path if return_path else x


# --------------------------------------------------------------------------- full model

class TriModalFlowModel(nn.Module):
    """Modality encoders plus either one shared flow or one flow per modality."""

    def __init__(self, input_dims, latent_dim=512, sequence_encoder=None, num_heads=8,
                 shared_flow=False, conditional_flow=False, ot_pairing=True):
        super().__init__()
        self.shared_flow = shared_flow
        self.conditional_flow = conditional_flow
        self.encoders = nn.ModuleDict({
            "text": MLPEncoder(input_dims["text"], latent_dim),
            **{m: build_encoder(sequence_encoder or "mlp", input_dims[m], latent_dim, num_heads)
               for m in ("audio", "vision")},
        })
        condition_dim = len(MODALITIES) if conditional_flow else None
        if shared_flow:
            self.flows = nn.ModuleDict({"shared": FlowMatching(latent_dim, 1024, 4, condition_dim, ot_pairing)})
        else:
            hidden, layers = (1024, 4) if conditional_flow else (512, 3)
            self.flows = nn.ModuleDict(
                {m: FlowMatching(latent_dim, hidden, layers, condition_dim, ot_pairing) for m in MODALITIES}
            )

    def encode(self, text, audio, vision, audio_mask=None, vision_mask=None):
        return {
            "text": self.encoders["text"](text),
            "audio": self.encoders["audio"](audio, audio_mask),
            "vision": self.encoders["vision"](vision, vision_mask),
        }

    def flow(self, modality):
        return self.flows["shared"] if self.shared_flow else self.flows[modality]


class RouterNetwork(nn.Module):
    """Per-sample modality weights ``softmax(G([z_t; z_a; z_v]) / temperature)``."""

    def __init__(self, latent_dim, num_modalities=3, temperature=0.1):
        super().__init__()
        in_dim = latent_dim * num_modalities
        self.temperature = temperature
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2), nn.BatchNorm1d(in_dim // 2), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(in_dim // 2, in_dim // 4), nn.BatchNorm1d(in_dim // 4), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(in_dim // 4, num_modalities),
        )

    def forward(self, latents):
        return F.softmax(self.net(torch.cat([latents[m] for m in MODALITIES], dim=1)) / self.temperature, dim=1)


def fuse(latents, router=None):
    """Router-weighted sum of the modality latents (or their mean). Returns ``(fused, weights | None)``."""
    stacked = [latents[m] for m in MODALITIES]
    if router is None:
        return sum(stacked) / len(stacked), None
    weights = router(latents)
    return sum(z * weights[:, i:i + 1] for i, z in enumerate(stacked)), weights


class PrototypeClassifier(nn.Module):
    """Cosine-similarity classifier against one learnable prototype per class."""

    def __init__(self, num_classes, latent_dim, temperature=0.1, loss="ce",
                 class_weights=None, focal_gamma=2.0, focal_reduction="sum"):
        super().__init__()
        self.temperature = temperature
        self.prototypes = nn.Parameter(torch.empty(num_classes, latent_dim))
        nn.init.xavier_uniform_(self.prototypes)
        self.focal = FocalLoss(class_weights, focal_gamma, focal_reduction) if loss == "focal" else None

    def normalized(self):
        return F.normalize(self.prototypes, dim=1)

    def similarities(self, features):
        return F.normalize(features, dim=1) @ self.normalized().T

    def forward(self, features, labels):
        logits = self.similarities(features) / self.temperature
        return self.focal(logits, labels) if self.focal is not None else F.cross_entropy(logits, labels)

    def predict(self, features):
        return self.similarities(features).argmax(dim=1)
