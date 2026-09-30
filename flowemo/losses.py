import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    """Supervised contrastive loss: samples with the same class label are positives."""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        features = F.normalize(features, dim=1)
        logits = features @ features.T / self.temperature
        not_self = 1.0 - torch.eye(features.shape[0], device=features.device)
        positives = (labels[:, None] == labels[None, :]).float() * not_self
        exp_logits = torch.exp(logits) * not_self
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        return -((positives * log_prob).sum(1) / (positives.sum(1) + 1e-8)).mean()


def cross_modal_contrastive_loss(latents, temperature=0.07):
    """InfoNCE between modalities of the same sample, averaged over all modality pairs."""
    names = list(latents)
    z = {m: F.normalize(latents[m], dim=1) for m in names}
    eye = torch.eye(z[names[0]].shape[0], device=z[names[0]].device)
    losses = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            exp_sim = torch.exp(z[a] @ z[b].T / temperature)
            losses.append(-torch.log((exp_sim * eye).sum(1) / (exp_sim.sum(1) + 1e-8)).mean())
    return sum(losses) / len(losses)


def router_entropy_loss(weights, eps=1e-8):
    """Negative routing entropy scaled by batch size; minimizing it keeps routing from collapsing."""
    entropy = -(weights * torch.log(weights + eps)).sum(dim=1)
    return -entropy.mean() * weights.size(0)


def prototype_orthogonality_loss(prototypes):
    """Mean squared deviation of the prototype Gram matrix from identity (keeps prototypes apart)."""
    k = prototypes.shape[0]
    gram = prototypes @ prototypes.T
    return ((gram - torch.eye(k, device=prototypes.device)) ** 2).sum() / k ** 2


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="sum"):
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        loss = (1 - torch.exp(-ce)) ** self.gamma * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
