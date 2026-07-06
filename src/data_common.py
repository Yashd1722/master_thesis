"""
src/data_common.py

Single source of truth for how a residual time series is turned into a model
input — used by BOTH training (src/dataset_loader.py, training/train.py) and
empirical inference (src/rolling_window.py).

Why this file exists
--------------------
Bury et al. (2021) train their classifiers on *censored* series: each training
sample gets a random amount of left zero-padding, so the model learns to detect
a transition from only the tail of a record. At inference on short sediment
cores (57-365 points, far shorter than ts_len=500/1500) the record is likewise
left-padded with zeros to ts_len. If training only ever sees clean full-length
signals, the padded empirical input is completely out-of-distribution and the
model collapses to a constant prediction (AUC ~= 0.5) or inverts.

The two operations that MUST be identical on both sides are:
    1. normalisation  — divide the visible signal by its mean absolute value.
    2. left-padding    — zeros on the left, real signal (ending at the
                          transition) right-aligned.

Keeping them here guarantees train/inference parity for every current and
future model.
"""
from __future__ import annotations

import numpy as np


def normalize_mean_abs(x: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Divide a 1-D series by its mean absolute value (-> mean|x| == 1).

    This is the exact normalisation used at inference in
    rolling_window.prepare_dl_input, so training must use it too. A near-flat
    series (mean|x| <= eps) is returned unchanged to avoid divide-by-zero.
    """
    x = np.asarray(x, dtype=np.float64)
    denom = np.mean(np.abs(x))
    if denom > eps:
        x = x / denom
    return x.astype(np.float32)


def left_pad_to(x: np.ndarray, length: int) -> np.ndarray:
    """Right-align `x` in a zero vector of size `length` (pad on the LEFT).

    If `x` is longer than `length`, keep its last `length` points (the tail,
    which for our data holds the approach to the transition).
    """
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    if n >= length:
        return x[-length:].astype(np.float32)
    out = np.zeros(length, dtype=np.float32)
    out[-n:] = x
    return out


def random_left_censor(
    x: np.ndarray,
    length: int,
    rng: np.random.Generator,
    pad_max_frac: float = 0.9,
    min_visible: int = 30,
) -> np.ndarray:
    """Bury-style left-censoring augmentation for ONE full-length training series.

    Steps (mirrors what inference produces for a short real record):
        1. pick a random pad length  pad = int(U(0, pad_max_frac) * length)
        2. keep the visible tail  x[-visible:]   (transition stays at the end)
        3. normalise the visible tail to mean|x| == 1
        4. right-align it in a zero vector of size `length`

    `min_visible` floors the visible length so extremely short (all-zero-ish)
    inputs are not produced. Returns a (length,) float32 vector.
    """
    x = np.asarray(x, dtype=np.float64)
    L = length
    pad = int(rng.uniform(0.0, pad_max_frac) * L)
    visible = L - pad
    if visible < min_visible:
        visible = min_visible
    tail = x[-visible:] if len(x) >= visible else x
    tail = normalize_mean_abs(tail)
    return left_pad_to(tail, L)


def make_model_input(x: np.ndarray, length: int) -> np.ndarray:
    """Clean (un-augmented) model input: normalise then left-pad to `length`.

    Used for val/test and for the empirical inference of a full visible record.
    Equivalent to random_left_censor with pad == 0.
    """
    return left_pad_to(normalize_mean_abs(x), length)


def make_fixed_window(residuals: np.ndarray, end_pos: int, length: int) -> np.ndarray:
    """Fixed-length rolling-window input ending at `end_pos`.

    Takes the last `length` residuals before `end_pos`, normalises them, and
    left-pads to `length`. For records shorter than `length` (all our empirical
    cores) this is identical to normalising residuals[:end_pos] and left-padding
    — i.e. it matches the historical prepare_dl_input behaviour — but it is also
    correct when a record is longer than `length` (fixed window, not growing
    history). This is the ONE function inference should call.
    """
    seg = np.asarray(residuals[:end_pos], dtype=np.float64)
    if len(seg) > length:
        seg = seg[-length:]
    return make_model_input(seg, length)
