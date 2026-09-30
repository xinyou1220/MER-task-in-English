"""Plots: training curves, confusion matrices and PCA views of the learned flow.

Flow views use one batch of the given loader. The flow starts at each sample's class prototype and
is integrated forward; for per-modality flows the text flow is shown.
"""
import math
import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

from .models import fuse  # noqa: E402
from .trainer import to_device  # noqa: E402


def plot_confusion_matrix(y_true, y_pred, class_names, save_path, normalize=False):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    if normalize:
        cm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)
    ticks = range(len(class_names))
    ax.set(title="Confusion matrix" + (" (normalized)" if normalize else ""),
           xlabel="Predicted label", ylabel="True label", xticks=ticks, yticks=ticks)
    ax.set_xticklabels(class_names, rotation=45)
    ax.set_yticklabels(class_names)
    threshold = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], ".2f" if normalize else "d"), ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history, save_path):
    names = sorted(history)
    n_cols = 3
    n_rows = math.ceil(len(names) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)
    for ax, name in zip(axes.flat, names):
        ax.plot(range(1, len(history[name]) + 1), history[name], linewidth=1.5)
        ax.set(title=name.replace("_", " "), xlabel="Epoch")
        ax.grid(True, alpha=0.3)
    for ax in list(axes.flat)[len(names):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def _flow_setup(model, prototype, router, loader, device, num_points):
    model.eval()
    prototype.eval()
    if router is not None:
        router.eval()
    text, audio, vision, labels, audio_mask, vision_mask = to_device(next(iter(loader)), device)
    latents = model.encode(text, audio, vision, audio_mask, vision_mask)
    latents = {m: z[:num_points] for m, z in latents.items()}
    labels = labels[:num_points]
    x_1, weights = fuse(latents, router)
    condition = weights if model.conditional_flow else None
    return x_1, labels, prototype.normalized(), condition, model.flow("text")


def _grid_velocity(flow, pca, grid_low, t, condition, device):
    grid_high = torch.tensor(pca.inverse_transform(grid_low), dtype=torch.float32, device=device)
    cond = condition.mean(dim=0, keepdim=True).expand(len(grid_high), -1) if condition is not None else None
    v = flow(grid_high, torch.full((len(grid_high),), t, device=device), cond)
    return pca.transform((grid_high + v).cpu().numpy()) - grid_low


@torch.no_grad()
def visualize_flow_2d(model, prototype, router, loader, device, save_dir, tag="final", num_points=200, steps=10):
    """Vector field at t=0.5, transport trajectories, and start / transported / real distributions."""
    x_1, labels, protos, condition, flow = _flow_setup(model, prototype, router, loader, device, num_points)
    path = [p.cpu().numpy() for p in flow.integrate(protos[labels], steps, condition=condition, return_path=True)]
    traj = np.stack(path)
    pca = PCA(n_components=2).fit(traj.reshape(-1, traj.shape[-1]))
    traj_2d = np.stack([pca.transform(p) for p in traj])
    real_2d = pca.transform(x_1.cpu().numpy())
    proto_2d = pca.transform(protos.cpu().numpy())

    grid_res = 20
    lo, hi = traj_2d.reshape(-1, 2).min(axis=0), traj_2d.reshape(-1, 2).max(axis=0)
    margin = (hi[0] - lo[0]) * 0.1
    xx, yy = np.meshgrid(np.linspace(lo[0] - margin, hi[0] + margin, grid_res),
                         np.linspace(lo[1] - margin, hi[1] + margin, grid_res))
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    v = _grid_velocity(flow, pca, grid, 0.5, condition, device)

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    axes[0].quiver(xx, yy, v[:, 0].reshape(xx.shape), v[:, 1].reshape(xx.shape),
                   color="blue", alpha=0.6, scale=20, width=0.003)
    axes[0].scatter(*proto_2d.T, c="orange", s=200, marker="*", label="Prototypes", zorder=5)
    axes[0].set_title("Projected vector field (t = 0.5)")
    axes[0].legend()

    colors = plt.cm.tab10(labels.cpu().numpy() % 10)
    for i in range(traj_2d.shape[1]):
        axes[1].plot(traj_2d[:, i, 0], traj_2d[:, i, 1], color=colors[i], alpha=0.3, linewidth=1)
        axes[1].scatter(*traj_2d[0, i], color=colors[i], s=20, marker="o", alpha=0.6)
        axes[1].scatter(*traj_2d[-1, i], color=colors[i], s=20, marker=">", alpha=0.8)
    axes[1].set_title("Transport trajectories (prototype -> feature)")

    axes[2].scatter(*traj_2d[0].T, c="orange", alpha=0.6, s=50, edgecolors="k", label=r"$\pi_0$ (prototypes)")
    axes[2].scatter(*traj_2d[-1].T, c="green", alpha=0.4, s=50, label="Transported")
    axes[2].scatter(*real_2d.T, c="blue", alpha=0.2, s=30, label=r"$\pi_1$ (real features)")
    axes[2].set_title("Start vs. transported vs. real")
    axes[2].legend()
    for ax in axes:
        ax.set(xlabel="PC 1", ylabel="PC 2")

    fig.tight_layout()
    path = os.path.join(save_dir, f"flow_2d_{tag}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


@torch.no_grad()
def visualize_flow_frames(model, prototype, router, loader, device, save_dir, tag="final",
                          num_frames=20, num_points=200):
    """One PNG per integration step (``frames_<tag>/frame_XXX.png``) for making an animation."""
    frame_dir = os.path.join(save_dir, f"frames_{tag}")
    shutil.rmtree(frame_dir, ignore_errors=True)
    os.makedirs(frame_dir)

    x_1, labels, protos, condition, flow = _flow_setup(model, prototype, router, loader, device, num_points)
    path = flow.integrate(protos[labels], num_frames, condition=condition, return_path=True)
    all_points = torch.cat(path).cpu().numpy()
    pca = PCA(n_components=2).fit(all_points)
    pts_2d = pca.transform(all_points)
    lo, hi = pts_2d.min(axis=0), pts_2d.max(axis=0)
    margin = (hi - lo) * 0.1
    grid_res = 20
    xx, yy = np.meshgrid(np.linspace(lo[0] - margin[0], hi[0] + margin[0], grid_res),
                         np.linspace(lo[1] - margin[1], hi[1] + margin[1], grid_res))
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    proto_2d = pca.transform(protos.cpu().numpy())

    for i, x_t in enumerate(path):
        t = i / num_frames
        v = _grid_velocity(flow, pca, grid, t, condition, device)
        x_2d = pca.transform(x_t.cpu().numpy())
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.quiver(xx, yy, v[:, 0].reshape(xx.shape), v[:, 1].reshape(xx.shape),
                  color="blue", alpha=0.5, scale=20, width=0.003)
        ax.scatter(*x_2d.T, c=labels.cpu().numpy(), cmap="tab10", s=30, alpha=0.8, edgecolors="w")
        ax.scatter(*proto_2d.T, c="orange", marker="*", s=200, label="Prototypes", zorder=5)
        ax.set(title=f"Flow at t = {t:.2f}", xlabel="PC 1", ylabel="PC 2",
               xlim=(lo[0] - margin[0], hi[0] + margin[0]), ylim=(lo[1] - margin[1], hi[1] + margin[1]))
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(frame_dir, f"frame_{i:03d}.png"), dpi=100)
        plt.close(fig)
    print(f"Saved {len(path)} frames to {frame_dir}")


@torch.no_grad()
def visualize_flow_3d(model, prototype, router, loader, device, save_dir, tag="final",
                      num_points=200, noise_scale=0.2, steps=10):
    """3-D PCA view of trajectories from noisy prototypes, with the vector field at t=0.5."""
    x_1, labels, protos, condition, flow = _flow_setup(model, prototype, router, loader, device, num_points)
    x_0 = F.normalize(protos[labels] + torch.randn_like(x_1) * noise_scale, dim=1)
    traj = np.stack([p.cpu().numpy() for p in flow.integrate(x_0, steps, condition=condition, return_path=True)])
    pca = PCA(n_components=3).fit(traj.reshape(-1, traj.shape[-1]))
    traj_3d = np.stack([pca.transform(p) for p in traj])
    proto_3d = pca.transform(protos.cpu().numpy())

    lo, hi = traj_3d.reshape(-1, 3).min(axis=0), traj_3d.reshape(-1, 3).max(axis=0)
    axes_lin = [np.linspace(lo[d], hi[d], 6) for d in range(3)]
    grid = np.stack([g.ravel() for g in np.meshgrid(*axes_lin)], axis=1)
    v = _grid_velocity(flow, pca, grid, 0.5, condition, device)

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection="3d")
    colors = plt.cm.tab10(labels.cpu().numpy() % 10)
    for i in range(traj_3d.shape[1]):
        ax.plot(*traj_3d[:, i].T, color=colors[i], alpha=0.3, linewidth=0.8)
    ax.scatter(*traj_3d[0].T, c="green", s=20, alpha=0.4, label="Start ($x_0$)")
    ax.scatter(*traj_3d[-1].T, c="blue", s=20, alpha=0.4, label="Transported ($x_1$)")
    ax.scatter(*proto_3d.T, c="orange", s=300, marker="*", edgecolors="black", label="Prototypes")
    ax.quiver(*grid.T, *v.T, length=0.5, normalize=True, color="gray", alpha=0.3, arrow_length_ratio=0.3)
    ax.set(title=f"3-D flow field and trajectories ({tag})", xlabel="PC 1", ylabel="PC 2", zlabel="PC 3")
    ax.legend()
    ax.view_init(elev=30, azim=45)
    fig.tight_layout()
    path = os.path.join(save_dir, f"flow_3d_{tag}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")
