# FlowEmo: Multimodal Emotion Recognition with Prototype-Anchored Flow Matching

FlowEmo recognizes sentiment and emotion from **text, audio and vision**. The three modalities are encoded into a shared, L2-normalized latent space and classified by similarity to learned **class prototypes**. A **rectified-flow** field is trained to transport noisy prototypes to the real encoder features, which shapes the space around the prototypes.

Supported datasets and tasks:

| Config | Dataset | Classes | Audio / vision encoder |
|---|---|---|---|
| `configs/cmu_mosei_binary.yaml` | CMU-MOSEI | 2 (negative / positive) | mean-pooled vectors, MLP |
| `configs/cmu_mosei_7class.yaml` | CMU-MOSEI | 7 sentiment classes | mean-pooled vectors, MLP |
| `configs/iemocap_6class.yaml` | IEMOCAP | 6 (hap, sad, neu, ang, exc, fru) | sequences, attention pooling |
| `configs/meld_7class.yaml` | MELD | 7 (joy, sadness, anger, fear, disgust, surprise, neutral) | sequences, attention pooling |
| `configs/meld_7class_cls.yaml` | MELD | 7 | sequences, `[CLS]` cross-attention |
| `configs/cmu_mosei_emotion.yaml` | CMU-MOSEI (MELD emotion labels) | 7 | early experiment, see below |
| `configs/cmu_mosei_pt_sentiment.yaml` | CMU-MOSEI `.pt` features (GloVe / COVAREP / Facet) | 7 | early experiment, see below |

## Method

```
text  ─► BERT [CLS] ─► MLP encoder ────────┐
audio ─► MLP | attention | CLS encoder ────┼─► z_T, z_A, z_V  (unit sphere)
vision ─► MLP | attention | CLS encoder ───┘        │
                                                     ├─► fusion: mean or router  Σ W_m z_m ─► prototypes ─► class
                                                     ├─► supervised contrastive (per modality) + cross-modal InfoNCE
                                                     └─► flow matching:  noisy prototype  ──v(x,t)──►  z_m
```

1. **Encoders.** Text is the `[CLS]` vector of a frozen `bert-base-uncased`, followed by an MLP. Audio and vision are either mean-pooled over time and passed through an MLP, or kept as padded sequences. Sequences are summarized by *attention pooling* or by a learned *`[CLS]` query that cross-attends* over the time steps. Every latent is L2-normalized.
2. **Prototype classifier.** Each class has a learnable prototype. Logits are cosine similarities divided by a temperature. The loss is cross-entropy or focal loss with inverse-frequency class weights. An orthogonality penalty `‖PPᵀ − I‖²` keeps the prototypes apart. Prediction is the nearest prototype.
3. **Flow matching (rectified flow).** For each modality, the source is `x₀ = normalize(p_y + σ·ε)`, a noisy prototype of the sample's class. The target is the encoder latent `x₁`. A velocity MLP with a sinusoidal time embedding regresses `v(x_t, t) ≈ x₁ − x₀` along `x_t = (1−t)x₀ + t·x₁`. Within a batch, targets are paired to sources by a Hungarian (optimal-transport) assignment. The flow can be shared across modalities or per modality, and optionally conditioned on the router weights.
4. **Contrastive alignment.** A supervised contrastive loss is applied within each modality, and InfoNCE is applied across modalities of the same sample (T–A, T–V, A–V).
5. **Fusion.** The modality latents are averaged, or combined with per-sample weights from a router MLP (softmax with temperature). A router-entropy term keeps the router from collapsing.
6. **Optional flow inference.** Each latent is integrated *backwards* along the field, toward its prototype, and blended with the original: `z ← α·z + (1−α)·reverse_flow(z)`. By default this is used on the validation set only.

Total loss: `λ_flow·L_flow + λ_intra·L_supcon + λ_cross·L_infonce + λ_proto·L_proto + λ_router·L_entropy + λ_ortho·L_ortho`. Encoders, flows and prototypes use AdamW with cosine learning-rate decay, and training batches are class-balanced by a weighted sampler.

## Installation

```bash
pip install -r requirements.txt
```

Python 3.9+ and PyTorch 2.x are required. The first run downloads `bert-base-uncased` from the Hugging Face Hub.

## Data

The datasets are not redistributed here. Obtain CMU-MOSEI, MELD and IEMOCAP from their official sources and extract features.

**Pickle format** (`data.format: pickle`, all main configs). Each split is a pickled list of samples:

```python
(features, label, sample_id)
features = (words, visual_seq, acoustic_seq, raw_text, visual_len, acoustic_len)
#          visual_seq: [T_v, d_v]   acoustic_seq: [T_a, d_a]   raw_text: str
```

Only `features[1:4]` are used. How `label` is parsed depends on `data.label_parser`:

| Parser | Accepted labels |
|---|---|
| `binary_sentiment` | `"0"` / `"1"`, or `"negative,..."` / `"positive,..."` |
| `mosei_7class` | an integer 0–6 in the first comma-separated field |
| `meld` | an emotion word, in the 3rd field for the `sentiment,score,emotion,...` training format, otherwise the only field |
| `iemocap` | an IEMOCAP code (`hap`, `sad`, `neu`, `ang`, `exc`, `fru`) in the last field |
| `mosei_emotion` | a MELD emotion word in the first field |

**CMU-MOSEI `.pt` format** (`data.format: mosei_pt`). Each split is a `torch.save`-d list of dicts with the keys `visual [T, 35]`, `audio [T, 74]`, `languages [T, 300]` and `label [7]` (sentiment, happy, sad, anger, surprise, disgust, fear).

## Usage

```bash
python train.py --config configs/meld_7class.yaml \
    --train data/meld/train.pkl --val data/meld/dev.pkl --test data/meld/test.pkl
```

Any config entry can be overridden from the command line:

```bash
python train.py --config configs/iemocap_6class.yaml --train ... --val ... --test ... \
    --set model.use_router=true --set train.epochs=20 --set inference.test_flow_inference=true
```

`flowemo/config.py` lists every option with its default. The main switches are `model.use_router`, `model.shared_flow`, `model.conditional_flow`, `model.prototype_loss`, `flow.prototype_source`, `flow.ot_pairing`, `train.select_metric` and `inference.*`.

### Outputs

Each run writes to `runs/<config>_<timestamp>/`:

| File | Content |
|---|---|
| `config.yaml` | The fully resolved configuration |
| `best_model.pth` | Encoders + flows, prototypes and router at the best validation epoch |
| `test_results.txt` | Test accuracy, macro-F1, weighted-F1 (and router weights) |
| `confusion_matrix*.png`, `training_curves.png` | Evaluation and training plots |
| `learned_prototypes.npy` | Normalized class prototypes |
| `flow_2d_final.png`, `flow_3d_final.png`, `frames_final/` | PCA views of the learned field and transport trajectories on a test batch (the `frames_final/` images can be turned into an animation) |

### Early experiments

`cmu_mosei_emotion.yaml` and `cmu_mosei_pt_sentiment.yaml` reproduce the first version of the method. That version has no OT pairing (`flow.ot_pairing: false`), lets the flow loss back-propagate into the encoders (`flow.detach_target: false`), and selects checkpoints by accuracy or macro-F1. They are kept for completeness. The other configs are the final setup.

## Repository layout

```
flowemo/
  config.py     defaults and YAML / command-line overrides
  data.py       pickle and .pt datasets, label parsers, BERT text features, padding collate, samplers
  models.py     encoders, flow matching, tri-modal model, router, prototype classifier
  losses.py     supervised contrastive, cross-modal InfoNCE, router entropy, orthogonality, focal loss
  trainer.py    training loop and evaluation
  visualize.py  confusion matrices, training curves, 2-D / 3-D flow visualizations
configs/        one YAML file per experiment
train.py        entry point
```

## Status

This project is archived and no longer maintained. Issues and pull requests may not receive a response.

## License

The code is released under the [MIT License](LICENSE). Datasets and pretrained models used by the scripts are subject to their own licenses.
