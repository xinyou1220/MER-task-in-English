"""Train and evaluate FlowEmo on one experiment configuration.

Examples:
    python train.py --config configs/meld_7class.yaml --train data/meld/train.pkl \
        --val data/meld/dev.pkl --test data/meld/test.pkl
    python train.py --config configs/iemocap_6class.yaml --set train.epochs=20 --set model.use_router=true
"""
import argparse
import os
import random
from datetime import datetime

import numpy as np
import torch
import yaml

from flowemo.config import load_config
from flowemo.data import CLASS_NAMES, build_datasets, build_loaders, class_weights
from flowemo.models import PrototypeClassifier, RouterNetwork, TriModalFlowModel
from flowemo.trainer import evaluate, train
from flowemo.visualize import (
    plot_confusion_matrix,
    plot_training_curves,
    visualize_flow_2d,
    visualize_flow_3d,
    visualize_flow_frames,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="YAML experiment config (see configs/)")
    p.add_argument("--train", help="overrides data.train")
    p.add_argument("--val", help="overrides data.val")
    p.add_argument("--test", help="overrides data.test")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override any config entry, e.g. --set train.epochs=10")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    args = parse_args()
    overrides = list(args.set) + [f"data.{k}={v}" for k, v in
                                  (("train", args.train), ("val", args.val), ("test", args.test)) if v]
    cfg, cfg_dict = load_config(args.config, overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)

    name = os.path.splitext(os.path.basename(args.config))[0] if args.config else "default"
    run_dir = os.path.join(cfg.output_dir, f"{name}_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_dict, f, sort_keys=False)
    print(f"Device: {device} | outputs: {run_dir}")

    datasets = build_datasets(cfg, device)
    train_loader, val_loader, test_loader = build_loaders(datasets, cfg)
    text, audio, vision, _ = datasets["train"][0]
    input_dims = {"text": text.shape[-1], "audio": audio.shape[-1], "vision": vision.shape[-1]}
    print(f"Input dims: {input_dims} | sizes: " + ", ".join(f"{k}={len(v)}" for k, v in datasets.items()))

    m = cfg.model
    model = TriModalFlowModel(
        input_dims, m.latent_dim,
        sequence_encoder=m.sequence_encoder if cfg.data.sequence_features else None,
        num_heads=m.num_heads, shared_flow=m.shared_flow, conditional_flow=m.conditional_flow,
        ot_pairing=cfg.flow.ot_pairing,
    ).to(device)
    weights = class_weights(datasets["train"].labels, m.num_classes).to(device)
    print(f"Class weights: {np.round(weights.cpu().numpy(), 3)}")
    prototype = PrototypeClassifier(
        m.num_classes, m.latent_dim, m.prototype_temperature, m.prototype_loss,
        class_weights=weights if m.prototype_loss == "focal" else None,
        focal_gamma=m.focal_gamma, focal_reduction=m.focal_reduction,
    ).to(device)
    router = RouterNetwork(m.latent_dim, 3, m.router_temperature).to(device) if m.use_router else None

    history = train(model, prototype, router, train_loader, val_loader, device, cfg, run_dir)

    checkpoint = torch.load(os.path.join(run_dir, "best_model.pth"), map_location=device)
    model.load_state_dict(checkpoint["model"])
    prototype.load_state_dict(checkpoint["prototype"])
    if router is not None:
        router.load_state_dict(checkpoint["router"])
    print(f"Loaded best model from epoch {checkpoint['epoch']}")

    result = evaluate(model, prototype, router, test_loader, device, cfg,
                      flow_inference=cfg.inference.test_flow_inference)
    lines = [f"Test results ({name}, best epoch {checkpoint['epoch']})",
             f"  flow inference: {cfg.inference.test_flow_inference} "
             f"(steps {cfg.inference.flow_steps}, alpha {cfg.inference.flow_alpha})"]
    lines += [f"  {k:<12} {v:.4f}" for k, v in result["metrics"].items()]
    if result["router_weights"] is not None:
        lines.append("  router weights (text/audio/vision): "
                     + " / ".join(f"{w:.3f}" for w in result["router_weights"]))
    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(run_dir, "test_results.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")

    class_names = cfg.data.class_names
    if class_names is None and cfg.data.format == "pickle":
        class_names = CLASS_NAMES.get(cfg.data.label_parser)
    class_names = [str(c) for c in (class_names or range(m.num_classes))]
    for normalize in (False, True):
        suffix = "_normalized" if normalize else ""
        plot_confusion_matrix(result["y_true"], result["y_pred"], class_names,
                              os.path.join(run_dir, f"confusion_matrix{suffix}.png"), normalize)
    plot_training_curves(history, os.path.join(run_dir, "training_curves.png"))
    np.save(os.path.join(run_dir, "learned_prototypes.npy"), prototype.normalized().detach().cpu().numpy())

    if cfg.visualize:
        visualize_flow_2d(model, prototype, router, test_loader, device, run_dir)
        visualize_flow_frames(model, prototype, router, test_loader, device, run_dir)
        visualize_flow_3d(model, prototype, router, test_loader, device, run_dir, noise_scale=cfg.flow.noise_scale)
    print(f"\nAll outputs saved to {run_dir}")


if __name__ == "__main__":
    main()
