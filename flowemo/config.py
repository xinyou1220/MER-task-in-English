"""Experiment configuration: built-in defaults, overridden by a YAML file and ``--set key=value``."""
import copy
from types import SimpleNamespace

import yaml

DEFAULTS = {
    "seed": 16,
    "output_dir": "runs",
    "data": {
        "format": "pickle",              # pickle | mosei_pt
        "train": None,
        "val": None,
        "test": None,
        "label_parser": "binary_sentiment",  # pickle: binary_sentiment | mosei_7class | meld | mosei_emotion | iemocap
        "sequence_features": False,      # pickle: keep audio/vision as padded sequences instead of mean-pooling
        "text_model": "bert-base-uncased",
        "text_max_length": 64,
        "label_mode": "sentiment",       # mosei_pt: sentiment | sentiment_3class | emotion
        "pooling": "mean",               # mosei_pt: mean | max | last
        "balanced_sampling": True,       # class-balanced WeightedRandomSampler for training
        "num_workers": 0,
        "class_names": None,
    },
    "model": {
        "num_classes": 2,
        "latent_dim": 512,
        "sequence_encoder": "attention",  # attention | cls (used when data.sequence_features is true)
        "num_heads": 8,                  # cls encoder only
        "use_router": False,
        "router_temperature": 0.05,
        "prototype_temperature": 0.1,
        "prototype_loss": "ce",          # ce | focal (focal uses inverse-frequency class weights)
        "focal_gamma": 2.0,
        "focal_reduction": "sum",
        "shared_flow": False,            # one flow for all modalities vs. one per modality
        "conditional_flow": False,       # condition the flow on router weights
    },
    "flow": {
        "prototype_source": True,        # start from noisy class prototypes instead of N(0, I)
        "noise_scale": 0.2,
        "ot_pairing": True,              # Hungarian matching of sources and targets within a batch
        "detach_target": True,           # flow loss does not back-propagate into the encoders
    },
    "train": {
        "batch_size": 128,
        "epochs": 50,
        "lr_encoder": 2e-4,
        "lr_flow": 1e-4,
        "lr_prototype": 1e-3,
        "lr_router": 1e-3,
        "betas": [0.5, 0.999],
        "weight_decay": 1e-4,
        "lambda_flow": 1.0,
        "lambda_intra": 0.5,
        "lambda_cross": 0.3,
        "lambda_prototype": 1.0,
        "lambda_router_entropy": 0.5,
        "lambda_ortho": 0.1,
        "contrastive_temperature": 0.07,
        "grad_clip": 1.0,
        "select_metric": "weighted_f1",  # accuracy | macro_f1 | weighted_f1 (validation)
        "log_every": 100,
    },
    "inference": {
        "val_flow_inference": True,      # reverse-flow feature refinement during validation
        "test_flow_inference": False,    # ... and at test time
        "flow_steps": 10,
        "flow_alpha": 0.5,               # z <- alpha * z + (1 - alpha) * reverse_flow(z)
    },
    "visualize": True,
}


def _merge(base, override, path=""):
    for key, value in override.items():
        if key not in base:
            raise KeyError(f"Unknown config key: {path}{key}")
        if isinstance(base[key], dict) and isinstance(value, dict):
            _merge(base[key], value, f"{path}{key}.")
        else:
            base[key] = value


def _namespace(d):
    return SimpleNamespace(**{k: _namespace(v) if isinstance(v, dict) else v for k, v in d.items()})


def load_config(path=None, overrides=()):
    """Returns ``(namespace, plain dict)``; ``overrides`` are ``"section.key=value"`` strings (YAML-parsed)."""
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path, encoding="utf-8") as f:
            _merge(cfg, yaml.safe_load(f) or {})
    for item in overrides:
        key, _, raw = item.partition("=")
        nested = yaml.safe_load(raw)
        for part in reversed(key.split(".")):
            nested = {part: nested}
        _merge(cfg, nested)
    return _namespace(cfg), cfg
