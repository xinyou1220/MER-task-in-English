"""Training and evaluation for FlowEmo."""
import os
from collections import defaultdict

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from .losses import (
    SupervisedContrastiveLoss,
    cross_modal_contrastive_loss,
    prototype_orthogonality_loss,
    router_entropy_loss,
)
from .models import MODALITIES, fuse


def to_device(batch, device):
    return [t.to(device) if t is not None else None for t in batch]


def flow_sources(labels, latents, prototypes, cfg):
    """Flow start points: noisy class prototypes projected to the unit sphere, or pure Gaussian noise."""
    if not cfg.flow.prototype_source:
        return {m: torch.randn_like(latents[m]) for m in MODALITIES}
    center = prototypes[labels]
    return {m: F.normalize(center + torch.randn_like(latents[m]) * cfg.flow.noise_scale, dim=1)
            for m in MODALITIES}


def classification_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }


@torch.no_grad()
def evaluate(model, prototype, router, loader, device, cfg, flow_inference=False):
    """Prototype-based evaluation.

    With ``flow_inference`` (and a prototype-sourced flow) each modality latent is integrated
    backwards along the learned field toward its prototype and blended with the original latent:
    ``z <- alpha * z + (1 - alpha) * reverse_flow(z)``.
    """
    model.eval()
    prototype.eval()
    if router is not None:
        router.eval()

    y_true, y_pred, weights_all = [], [], []
    total_loss = 0.0
    for text, audio, vision, labels, audio_mask, vision_mask in (to_device(b, device) for b in loader):
        latents = model.encode(text, audio, vision, audio_mask, vision_mask)

        if flow_inference and cfg.flow.prototype_source:
            condition = None
            if model.conditional_flow and router is not None:
                condition = router(latents)
            alpha = cfg.inference.flow_alpha
            latents = {
                m: alpha * z + (1 - alpha) * model.flow(m).integrate(
                    z, cfg.inference.flow_steps, reverse=True, condition=condition)
                for m, z in latents.items()
            }

        fused, weights = fuse(latents, router)
        total_loss += prototype(fused, labels).item()
        y_true += labels.cpu().tolist()
        y_pred += prototype.predict(fused).cpu().tolist()
        if weights is not None:
            weights_all.append(weights.cpu())

    metrics = classification_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / len(loader)
    return {
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
        "router_weights": torch.cat(weights_all).mean(dim=0).tolist() if weights_all else None,
    }


def train(model, prototype, router, train_loader, val_loader, device, cfg, save_dir):
    """Trains all components jointly; keeps the checkpoint with the best ``cfg.train.select_metric``."""
    t = cfg.train
    opt_encoder = torch.optim.AdamW(model.encoders.parameters(), lr=t.lr_encoder, betas=t.betas,
                                    weight_decay=t.weight_decay)
    opt_flow = torch.optim.AdamW(model.flows.parameters(), lr=t.lr_flow, betas=t.betas,
                                 weight_decay=t.weight_decay)
    opt_proto = torch.optim.AdamW(prototype.parameters(), lr=t.lr_prototype)
    opt_router = torch.optim.AdamW(router.parameters(), lr=t.lr_router) if router is not None else None
    optimizers = [o for o in (opt_encoder, opt_flow, opt_proto, opt_router) if o is not None]
    schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=t.epochs, eta_min=1e-6)
                  for o in (opt_encoder, opt_flow, opt_proto)]
    supcon = SupervisedContrastiveLoss(t.contrastive_temperature)

    history = defaultdict(list)
    best_score = float("-inf")
    for epoch in range(1, t.epochs + 1):
        model.train()
        prototype.train()
        if router is not None:
            router.train()
        sums = defaultdict(float)
        y_true, y_pred = [], []

        for step, batch in enumerate(train_loader):
            text, audio, vision, labels, audio_mask, vision_mask = to_device(batch, device)
            for opt in optimizers:
                opt.zero_grad()

            latents = model.encode(text, audio, vision, audio_mask, vision_mask)

            # Flow matching: transport (noisy) prototypes to the encoder latents.
            sources = flow_sources(labels, latents, prototype.normalized().detach(), cfg)
            targets = {m: z.detach() if cfg.flow.detach_target else z for m, z in latents.items()}
            condition = None
            if model.conditional_flow and router is not None:
                condition = router({m: z.detach() for m, z in latents.items()})
            loss_flow = sum(model.flow(m).loss(sources[m], targets[m], condition) for m in MODALITIES) / 3

            loss_intra = sum(supcon(latents[m], labels) for m in MODALITIES) / 3
            loss_cross = cross_modal_contrastive_loss(latents, t.contrastive_temperature)
            loss_contrastive = t.lambda_intra * loss_intra + t.lambda_cross * loss_cross

            fused, weights = fuse(latents, router)
            loss_router = router_entropy_loss(weights) if weights is not None else torch.zeros((), device=device)
            loss_proto = prototype(fused, labels)
            loss_ortho = prototype_orthogonality_loss(prototype.normalized())

            loss = (t.lambda_flow * loss_flow + loss_contrastive + t.lambda_prototype * loss_proto
                    + t.lambda_router_entropy * loss_router + t.lambda_ortho * loss_ortho)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=t.grad_clip)
            torch.nn.utils.clip_grad_norm_(prototype.parameters(), max_norm=t.grad_clip)
            for opt in optimizers:
                opt.step()

            with torch.no_grad():
                y_true += labels.cpu().tolist()
                y_pred += prototype.predict(fused).cpu().tolist()
            for key, value in (("flow", loss_flow), ("contrastive", loss_contrastive),
                               ("prototype", loss_proto), ("router_entropy", loss_router)):
                sums[key] += value.item()

            if step % t.log_every == 0:
                router_info = ""
                if weights is not None:
                    w = weights.mean(dim=0)
                    router_info = f" | router T={w[0]:.2f} A={w[1]:.2f} V={w[2]:.2f}"
                print(f"[epoch {epoch:03d}/{t.epochs}] step {step:04d}/{len(train_loader)} "
                      f"flow {loss_flow.item():.4f} | contr {loss_contrastive.item():.4f} | "
                      f"proto {loss_proto.item():.4f}{router_info}")

        for scheduler in schedulers:
            scheduler.step()

        for key, value in sums.items():
            history[f"train_{key}_loss"].append(value / len(train_loader))
        history["train_accuracy"].append(accuracy_score(y_true, y_pred))

        val = evaluate(model, prototype, router, val_loader, device, cfg,
                       flow_inference=cfg.inference.val_flow_inference)
        for key, value in val["metrics"].items():
            history[f"val_{key}"].append(value)
        if val["router_weights"] is not None:
            for name, w in zip(MODALITIES, val["router_weights"]):
                history[f"val_router_{name}"].append(w)

        m = val["metrics"]
        print(f"\n[epoch {epoch}/{t.epochs}] train acc {history['train_accuracy'][-1]:.4f} | "
              f"val acc {m['accuracy']:.4f} macro-F1 {m['macro_f1']:.4f} weighted-F1 {m['weighted_f1']:.4f}")

        score = m[t.select_metric]
        if score > best_score:
            best_score = score
            state = {"epoch": epoch, "model": model.state_dict(), "prototype": prototype.state_dict(),
                     "val_metrics": m}
            if router is not None:
                state["router"] = router.state_dict()
            torch.save(state, os.path.join(save_dir, "best_model.pth"))
            print(f"Saved best model (val {t.select_metric} {best_score:.4f})")
        print()

    return dict(history)
