"""Minimal NumPy readout of ADEMPTOR's exported TopK SAE.

Source: issue503_cyclone_profiles.py (PaperSaeEncoder, raw BF16 decoder),
issue507_layerwise_adamw.py (checkpoint schema), and the issue518 F1 operator.
Equation: TopK32(ReLU((center(x) / max(L2(center(x)), 1e-6) - b_pre) @ enc_w)).
The exported dead mask is a training diagnostic; it does not mask inference.
"""
from pathlib import Path
import json

import numpy as np


def load_sae(checkpoint_path, *, layer=8, seed=50108):
    """Load an NPZ checkpoint and verify its entry in checkpoints/metadata.json."""
    path = Path(checkpoint_path)
    config = json.loads(path.with_name("metadata.json").read_text())[path.name]
    if config["layer"] != layer or config["initialization"]["seed"] != seed:
        raise ValueError("Checkpoint layer or training seed differs from requested SAE")
    model = config["model"]
    if (model["d_in"], model["d_sae"], model["k"]) != (512, 4096, 32):
        raise ValueError("Expected the 512-to-4096, TopK-32 checkpoint")
    if not model["row_normalize"] or model["input_epsilon"] != 1e-6:
        raise ValueError("Checkpoint row normalization differs from the accepted readout")
    with np.load(path, allow_pickle=False) as stored:
        enc_w = stored["enc_w"].copy()
        b_pre = stored["b_pre"].copy()
    if enc_w.shape != (512, 4096) or b_pre.shape != (512,):
        raise ValueError("Checkpoint encoder geometry differs")
    if any(a.dtype != np.float32 or not np.isfinite(a).all()
           for a in (enc_w, b_pre)):
        raise ValueError("Encoder parameters must be finite float32 arrays")
    return dict(enc_w=enc_w, b_pre=b_pre, layer=layer, seed=seed,
                top_k=model["k"], epsilon=model["input_epsilon"])



def load_corpus(path):
    """Decode one GraphCast snapshot, preserving its complete mesh-node order."""
    raw = np.load(path, allow_pickle=False)
    if raw.shape not in ((40962, 1, 512), (40962, 512)):
        raise ValueError("Expected one 40962-node, 512-channel processor snapshot")
    if raw.dtype.kind == "V" and raw.dtype.itemsize == 2:
        values = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    elif raw.dtype == np.float32:
        values = raw
    else:
        raise ValueError("Corpus must contain raw BF16 bits or decoded float32")
    values = values.reshape(40962, 512)
    if not np.isfinite(values).all():
        raise ValueError("Corpus contains non-finite activations")
    return values


def encode(corpus, sae, batch_size=512):
    """Return (feature_ids, values), each [nodes, 32], without a full dense code.

    Exact ties choose smaller feature IDs, matching JAX lax.top_k. NumPy/CPU
    and the original JAX backend can differ slightly in float32 arithmetic.
    """
    x = np.asarray(corpus, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != sae["enc_w"].shape[0]:
        raise ValueError("Corpus channel dimension differs from checkpoint")
    if batch_size < 1 or not np.isfinite(x).all():
        raise ValueError("Use finite corpus values and a positive batch size")
    k = sae["top_k"]
    ids = np.empty((len(x), k), dtype=np.int32)
    values = np.empty((len(x), k), dtype=np.float32)
    for start in range(0, len(x), batch_size):
        stop = min(start + batch_size, len(x))
        centered = x[start:stop] - x[start:stop].mean(axis=1, keepdims=True)
        norm = np.linalg.norm(centered, axis=1, keepdims=True)
        normalized = centered / np.maximum(norm, np.float32(sae["epsilon"]))
        dense = np.maximum((normalized - sae["b_pre"]) @ sae["enc_w"], 0)
        order = np.argsort(-dense, axis=1, kind="stable")[:, :k]
        ids[start:stop] = order
        values[start:stop] = np.take_along_axis(dense, order, axis=1)
    return ids, values


def rank_features(feature_ids, values, target_mask, n_features=4096):
    """Rank all features by single-time, global cyclone-union support F1.

    Positive retained activations define support. Ties follow descending IoU,
    recall, precision, then ascending feature ID. The single-time ranking is
    distinct from the paper's ranking by medians over the 2018-2019 record.
    """
    ids, values = np.asarray(feature_ids), np.asarray(values)
    target = np.asarray(target_mask, dtype=bool)
    if ids.shape != values.shape or ids.ndim != 2 or target.shape != (len(ids),):
        raise ValueError("Sparse codes and target mask must share the node axis")
    if not target.any():
        raise ValueError("F1 ranking requires at least one cyclone target node")
    active = values > 0
    count = np.bincount(ids[active], minlength=n_features)
    overlap = np.bincount(ids[active & target[:, None]], minlength=n_features)
    target_count = int(target.sum())
    f1 = 2 * overlap / (count + target_count)
    iou = overlap / (count + target_count - overlap)
    recall = overlap / target_count
    precision = np.divide(overlap, count, out=np.zeros(n_features), where=count > 0)
    order = np.lexsort((np.arange(n_features), -precision, -recall, -iou, -f1))
    return [dict(rank=rank, feature_id=int(feature), f1=float(f1[feature]),
                 iou=float(iou[feature]), recall=float(recall[feature]),
                 precision=float(precision[feature]), active_count=int(count[feature]),
                 overlap_count=int(overlap[feature]))
            for rank, feature in enumerate(order, 1)]


def select_feature(feature_ids, values, feature_id):
    """Recover one feature's [nodes] activation map from sparse codes."""
    return np.where(np.asarray(feature_ids) == feature_id, values, 0).sum(axis=1)
