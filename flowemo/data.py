"""Datasets for CMU-MOSEI, MELD and IEMOCAP.

Two on-disk formats are supported:

* ``pickle`` – a list of ``(features, label, id)`` where ``features[1]`` is the visual sequence,
  ``features[2]`` the acoustic sequence and ``features[3]`` the raw utterance text. Text is encoded
  with BERT (``[CLS]`` of the last layer); audio/vision are either mean-pooled over time or kept as
  sequences for the attention encoders.
* ``mosei_pt`` – a ``torch.save``-d list of dicts with ``visual``, ``audio``, ``languages``
  (word-vector sequence) and a 7-d ``label`` (sentiment, happy, sad, anger, surprise, disgust, fear).
"""
import pickle
from collections import Counter

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

MELD_EMOTIONS = ("joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral")
IEMOCAP_EMOTIONS = ("hap", "sad", "neu", "ang", "exc", "fru")


# --------------------------------------------------------------------------- label parsers
# Each parser maps the raw label field of a pickle sample to a class id.

def _fields(raw):
    return [p.strip() for p in str(raw).strip().lower().split(",")]


def parse_binary_sentiment(raw):
    """``"0"``/``"1"`` or ``"negative,..."``/``"positive,..."`` -> 0 / 1."""
    first = _fields(raw)[0]
    if first in ("negative", "positive"):
        return int(first == "positive")
    return int(first)


def parse_mosei_7class(raw):
    """Integer sentiment class in the first field, clipped to [0, 6]."""
    return min(max(int(_fields(raw)[0]), 0), 6)


def parse_meld(raw):
    """Emotion word in the third field (train format ``sentiment,score,emotion,...``) or the only field."""
    fields = _fields(raw)
    emotion = fields[2] if len(fields) >= 3 else fields[0]
    return MELD_EMOTIONS.index(emotion) if emotion in MELD_EMOTIONS else MELD_EMOTIONS.index("neutral")


def parse_mosei_emotion(raw):
    """Emotion word (MELD label set) in the first field; unknown words map to neutral."""
    emotion = _fields(raw)[0]
    return MELD_EMOTIONS.index(emotion) if emotion in MELD_EMOTIONS else MELD_EMOTIONS.index("neutral")


def parse_iemocap(raw):
    """IEMOCAP 6-class emotion code in the last field; unknown codes map to neutral."""
    emotion = _fields(raw)[-1]
    if emotion not in IEMOCAP_EMOTIONS:
        print(f"Warning: unknown IEMOCAP label '{raw}', mapped to 'neu'")
        emotion = "neu"
    return IEMOCAP_EMOTIONS.index(emotion)


LABEL_PARSERS = {
    "binary_sentiment": parse_binary_sentiment,
    "mosei_7class": parse_mosei_7class,
    "meld": parse_meld,
    "mosei_emotion": parse_mosei_emotion,
    "iemocap": parse_iemocap,
}

CLASS_NAMES = {
    "binary_sentiment": ("negative", "positive"),
    "meld": MELD_EMOTIONS,
    "mosei_emotion": MELD_EMOTIONS,
    "iemocap": IEMOCAP_EMOTIONS,
}


# --------------------------------------------------------------------------- text encoder

class BertCLSEncoder:
    """Frozen BERT; returns the last-layer ``[CLS]`` vector of each utterance."""

    def __init__(self, model_name="bert-base-uncased", max_length=64, device="cpu", batch_size=256):
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.max_length = max_length
        self.device = device
        self.batch_size = batch_size

    @torch.no_grad()
    def __call__(self, texts):
        outputs = []
        for start in range(0, len(texts), self.batch_size):
            batch = self.tokenizer(texts[start:start + self.batch_size], return_tensors="pt",
                                   padding="max_length", truncation=True, max_length=self.max_length)
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs.append(self.model(**batch).last_hidden_state[:, 0].float().cpu())
        return torch.cat(outputs)


def _clean_text(text):
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    return str(text)


def _to_tensor(features, pool_time):
    x = torch.as_tensor(np.asarray(features), dtype=torch.float32)
    if pool_time and x.dim() > 1:
        x = x.mean(dim=0)
    return torch.nan_to_num(x)


# --------------------------------------------------------------------------- datasets

class PickleMERDataset(Dataset):
    """Returns ``(text [768], audio, vision, label)``; audio/vision are ``[dim]`` or ``[T, dim]``."""

    def __init__(self, path, label_parser, text_encoder, sequence_features=False):
        with open(path, "rb") as f:
            samples = pickle.load(f)
        parse = LABEL_PARSERS[label_parser]
        self.labels = [parse(label) for _, label, _ in samples]
        pool_time = not sequence_features
        self.audio = [_to_tensor(feats[2], pool_time) for feats, _, _ in samples]
        self.vision = [_to_tensor(feats[1], pool_time) for feats, _, _ in samples]
        print(f"Encoding {len(samples)} utterances from {path} with BERT...")
        self.text = text_encoder([_clean_text(feats[3]) for feats, _, _ in samples])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.text[idx], self.audio[idx], self.vision[idx], torch.tensor(self.labels[idx])


class MoseiPtDataset(Dataset):
    """CMU-MOSEI ``.pt`` features (word vectors, COVAREP, Facet) pooled over time.

    ``label_mode``: ``sentiment`` (``num_classes`` equal-width bins over [-3, 3]), ``sentiment_3class``
    (negative < -0.5 < neutral < 0.5 < positive) or ``emotion`` (arg-max of the six emotion scores).
    """

    def __init__(self, path, label_mode="sentiment", num_classes=7, pooling="mean"):
        self.samples = torch.load(path, weights_only=False)
        self.label_mode = label_mode
        self.num_classes = num_classes
        self.pooling = pooling
        self.labels = [self._label(s["label"]) for s in self.samples]

    def _label(self, raw):
        if self.label_mode == "sentiment":
            bin_width = 6.0 / self.num_classes
            return min(max(int((raw[0].item() + 3.0) / bin_width), 0), self.num_classes - 1)
        if self.label_mode == "sentiment_3class":
            s = raw[0].item()
            return 0 if s < -0.5 else (2 if s > 0.5 else 1)
        if self.label_mode == "emotion":
            return raw[1:].argmax().item()
        raise ValueError(f"Unknown label_mode: {self.label_mode}")

    def _pool(self, seq):
        if self.pooling == "max":
            return seq.max(dim=0).values
        if self.pooling == "last":
            return seq[-1]
        return seq.mean(dim=0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        # COVAREP features contain inf values; bound everything to a sane range.
        vision = torch.nan_to_num(s["visual"], nan=0.0, posinf=100.0, neginf=-100.0).clamp(-100, 100)
        audio = torch.nan_to_num(s["audio"], nan=0.0, posinf=100.0, neginf=-100.0).clamp(-100, 100)
        text = torch.nan_to_num(s["languages"], nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10, 10)
        return self._pool(text), self._pool(audio), self._pool(vision), torch.tensor(self.labels[idx])


# --------------------------------------------------------------------------- loading

def collate(batch):
    """Stacks a batch. Sequence modalities are zero-padded and get a ``[B, T]`` validity mask.

    Returns ``(text, audio, vision, labels, audio_mask, vision_mask)``; masks are ``None`` for
    vector features.
    """
    text, audio, vision, labels = zip(*batch)
    out = [torch.stack(text)]
    masks = []
    for seqs in (audio, vision):
        if seqs[0].dim() == 1:
            out.append(torch.stack(seqs))
            masks.append(None)
        else:
            lengths = torch.tensor([len(s) for s in seqs])
            padded = pad_sequence(seqs, batch_first=True)
            out.append(padded)
            masks.append((torch.arange(padded.size(1))[None, :] < lengths[:, None]).float())
    return (*out, torch.stack(labels), *masks)


def class_weights(labels, num_classes):
    """``N / (num_classes * count_c)`` (0 for classes absent from ``labels``)."""
    counts = Counter(labels)
    return torch.tensor([len(labels) / (num_classes * counts[c]) if counts[c] else 0.0
                         for c in range(num_classes)], dtype=torch.float32)


def build_datasets(cfg, device):
    """Returns ``{'train', 'val', 'test'}`` datasets described by the ``data`` config section."""
    d = cfg.data
    paths = {"train": d.train, "val": d.val, "test": d.test}
    missing = [k for k, v in paths.items() if not v]
    if missing:
        raise ValueError(f"Missing data paths: {missing} (set them in the config or with --train/--val/--test)")

    if d.format == "mosei_pt":
        return {k: MoseiPtDataset(p, d.label_mode, cfg.model.num_classes, d.pooling) for k, p in paths.items()}
    if d.format == "pickle":
        encoder = BertCLSEncoder(d.text_model, d.text_max_length, device)
        datasets = {k: PickleMERDataset(p, d.label_parser, encoder, d.sequence_features) for k, p in paths.items()}
        del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return datasets
    raise ValueError(f"Unknown data format: {d.format}")


def build_loaders(datasets, cfg):
    d, batch_size = cfg.data, cfg.train.batch_size
    common = dict(batch_size=batch_size, num_workers=d.num_workers, collate_fn=collate)
    if d.balanced_sampling:
        counts = Counter(datasets["train"].labels)
        weights = torch.tensor([1.0 / counts[y] for y in datasets["train"].labels], dtype=torch.double)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(datasets["train"], sampler=sampler, drop_last=True, **common)
    else:
        train_loader = DataLoader(datasets["train"], shuffle=True, drop_last=True, **common)
    return (
        train_loader,
        DataLoader(datasets["val"], shuffle=False, **common),
        DataLoader(datasets["test"], shuffle=False, **common),
    )
