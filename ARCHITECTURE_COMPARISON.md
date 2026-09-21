# Architecture Comparison — Reference Implementations vs. This Repo (PyTorch)

Three deep-learning classifiers were added to the roster: **ResNet**, **TCN**, and
**RNN-FCN (LSTM-FCN)**. None of them exist as a PyTorch model that matches this
repo's interface (`nn.Module` with `forward(x: [B, 1, T]) -> [B, num_classes]`
and `predict_proba`), so each was re-implemented from the original paper and
cross-checked **twice**: once against the paper, once against a widely-used
reference implementation.

Input to every DL model in this repo is the single-channel right-aligned,
left-padded residual `x` of shape `[B, 1, T]` (T = 500 or 1500). The 5-channel
EWS feature stack is applied to TSC models only, not to DL models.

Summary of fidelity:

| model | reference | verdict |
|---|---|---|
| ResNet | Wang et al. 2017; Fawaz `dl-4-tsc/classifiers/resnet.py` (Keras) | **exact** — layer-for-layer identical |
| TCN | Bai et al. 2018; `locuslab/TCN` (PyTorch) | **exact** block structure + weight init; depth/width are config choices (documented) |
| RNN-FCN | Karim et al. 2018 (LSTM-FCN, IEEE Access); `titu1994/LSTM-FCN` (Keras); `tsai.models.RNN_FCN` (PyTorch) | **exact** — including the dimension-shuffle LSTM; LSTM cell count is a paper-sanctioned config (128) |

---

## 1. ResNet

**Paper:** Z. Wang, W. Yan, T. Oates, *"Time Series Classification from Scratch
with Deep Neural Networks: A Strong Baseline"*, IJCNN 2017.
**Reference implementation:** `hfawaz/dl-4-tsc`, `classifiers/resnet.py` (Keras),
also `aeon.classification.deep_learning.ResNetClassifier` (Keras).
**This repo:** `models/resnet.py`.

### Layer-by-layer

| stage | Fawaz / Wang (Keras) | `models/resnet.py` (PyTorch) | match |
|---|---|---|---|
| Block 1 conv a | Conv1D(64, k=8, pad=same) → BN → ReLU | Conv1d(1→64, k=8, pad=same, bias=False) → BN → ReLU | ✔ |
| Block 1 conv b | Conv1D(64, k=5, pad=same) → BN → ReLU | Conv1d(64→64, k=5) → BN → ReLU | ✔ |
| Block 1 conv c | Conv1D(64, k=3, pad=same) → BN | Conv1d(64→64, k=3) → BN | ✔ |
| Block 1 shortcut | Conv1D(64, k=1, pad=same) → BN | Conv1d(1→64, k=1) → BN (in≠out) | ✔ |
| Block 1 merge | Add → ReLU | `relu(y + shortcut(x))` | ✔ |
| Block 2 | filters 128, k=8/5/3, shortcut Conv1D(128,k=1)→BN | `_ResidualBlock(64→128)`, shortcut Conv1d(64→128,k=1)→BN | ✔ |
| Block 3 | filters 128, k=8/5/3, shortcut = **BN only** (channels match) | `_ResidualBlock(128→128)`, shortcut = `BatchNorm1d` (in==out) | ✔ |
| head | GlobalAveragePooling1D → Dense(n_classes, softmax) | `AdaptiveAvgPool1d(1)` → `Linear(128, n_classes)` (softmax in `predict_proba`) | ✔ |

Feature-map schedule **{64, 128, 128}**, kernel schedule **{8, 5, 3}**, no
pooling between blocks, ReLU after the residual add, `conv_c` has BN but no
pre-add activation — all identical to the reference.

### Differences

- `bias=False` on convolutions (Keras default is `bias=True`). Immaterial: every
  conv is followed by BatchNorm, which cancels a constant bias.
- Weight init: PyTorch default (Kaiming-uniform) vs Keras default (Glorot-uniform).
  Not specified in the paper; both are standard. Not corrected.
- `padding="same"` with even kernel 8 emits a benign PyTorch warning about an
  asymmetric zero-pad copy (same as the existing `models/cnn_lstm.py`, k=12).

### Training config (`config.yaml: training.resnet`)

Paper uses Adam, `ReduceLROnPlateau(factor=0.5, patience=50, min_lr=1e-4)`,
batch 64, 1500 epochs. This repo: Adam, `lr=1e-3`, `ReduceLROnPlateau(factor=0.5,
patience=15, min_lr=1e-6)`, batch 128, `epochs=500`, `patience=40` early stop —
tuned to the training-corpus size and the shared DL harness, not the UCR
per-dataset regime.

Verified: `python models/resnet.py` → `[8, 4]` logits, softmax sums to 1,
503,364 params, identical for T=500 and T=1500 (GAP removes length dependence).

---

## 2. TCN (Temporal Convolutional Network)

**Paper:** S. Bai, J. Z. Kolter, V. Koltun, *"An Empirical Evaluation of Generic
Convolutional and Recurrent Networks for Sequence Modeling"*, arXiv:1803.01271, 2018.
**Reference implementation:** `locuslab/TCN`, `TCN/tcn.py` (PyTorch).
**This repo:** `models/tcn.py`.

### Layer-by-layer

| component | `locuslab/TCN` | `models/tcn.py` | match |
|---|---|---|---|
| `Chomp1d` | `x[:, :, :-chomp_size]` | identical (`chomp_size > 0` guard added) | ✔ |
| `TemporalBlock` conv 1 | `weight_norm(Conv1d(k, dilation=d, padding=(k-1)*d))` | identical | ✔ |
| after conv 1 | Chomp1d → ReLU → Dropout | identical | ✔ |
| `TemporalBlock` conv 2 | `weight_norm(Conv1d(...))` → Chomp → ReLU → Dropout | identical | ✔ |
| residual | `Conv1d(k=1)` only if `n_in != n_out`, else identity | identical (`self.downsample`) | ✔ |
| block output | `relu(y + res)` | identical | ✔ |
| weight init | `conv.weight.data.normal_(0, 0.01)` on both convs + downsample | identical (`init` in `__init__`) | ✔ |
| stack | dilation `2**i`, channel list, input ch = prev output ch | identical | ✔ |
| classification head | reference returns full sequence; classification scripts take `out[:, :, -1]` → Linear | `Linear(channels, n_classes)` on `x[:, :, -1]` | ✔ (matches reference usage) |

### Differences

- **Depth / width are config choices** (the paper varies them per task):
  this repo uses `levels=8`, `channels=64`, `kernel_size=7`, `dropout=0.2`.
  Receptive field = `1 + 2·(k−1)·Σ 2^i` = **3061** ≥ 1500, so the last timestep
  sees the full series for both T=500 and T=1500.
- Head uses the last timestep (`x[:, :, -1]`), consistent with left-padded
  right-aligned input (transition signal at the tail) and with `models/cnn_lstm.py`
  / `models/lstm.py`, which also read `x[:, -1, :]`.
- `weight_norm` raises a `FutureWarning` (deprecated in favour of
  `parametrizations.weight_norm`); kept because the reference uses this exact call
  and the numerics are unchanged.

### Training config (`config.yaml: training.tcn`)

Paper/reference: Adam, `lr≈2e-3`, gradient clip `0.35–0.4`. This repo: Adam,
`lr=2e-3`, `grad_clip=0.4`, `weight_decay=0`, batch 128, `epochs=400`,
`patience=30`, `ReduceLROnPlateau(factor=0.5, patience=12)`.

Verified: `python models/tcn.py` → `[8, 4]` logits, softmax sums to 1,
432,964 params, identical for T=500 and T=1500.

---

## 3. RNN-FCN / LSTM-FCN

**Paper:** F. Karim, S. Majumdar, H. Darabi, S. Chen, *"LSTM Fully Convolutional
Networks for Time Series Classification"*, IEEE Access, 2018.
**Reference implementations:** `titu1994/LSTM-FCN` (Keras, `generate_lstmfcn`);
`tsai.models.RNN_FCN._RNN_FCN_Base` (PyTorch, `_cell=nn.LSTM`).
**This repo:** `models/rnn_fcn.py`.

### The dimension shuffle (the defining detail)

Karim's design feeds the LSTM the series as **one timestep of dimension T**, not
T timesteps of dimension 1. In Keras the input is shaped `(batch, 1, T)` and the
LSTM consumes it directly; the FCN branch permutes to `(batch, T, 1)`.
`tsai` does the same and comments it explicitly:
`self.rnn = LSTM(seq_len if shuffle else c_in, hidden_size)` with
`shuffle=True` default, `# You would normally permute x. Authors did the opposite.`

This repo matches: `self.lstm = nn.LSTM(ts_len, lstm_hidden, batch_first=True)`
and `forward` passes `x` of shape `[B, 1, T]` straight in → LSTM sees 1 step of
`ts_len` features → `lstm_out[:, -1, :]`. Consequence, and a correctness check:
the LSTM weight matrix scales with `ts_len`, so the parameter count differs
between datasets — **587,780 (T=500)** vs **1,099,780 (T=1500)** — exactly as in
the reference.

### Layer-by-layer

| branch | Karim (Keras) / tsai (PyTorch) | `models/rnn_fcn.py` | match |
|---|---|---|---|
| LSTM input | dimension-shuffled → `LSTM(units)` over 1 step of T features | `nn.LSTM(ts_len, 128)` over `[B, 1, T]` | ✔ |
| LSTM output | last step → `Dropout(0.8)` | `lstm_out[:, -1, :]` → `Dropout(0.8)` | ✔ |
| FCN conv 1 | Conv1D(128, k=8, pad=same) → BN → ReLU | Conv1d(1→128, k=8) → BN → ReLU | ✔ |
| FCN conv 2 | Conv1D(256, k=5) → BN → ReLU | Conv1d(128→256, k=5) → BN → ReLU | ✔ |
| FCN conv 3 | Conv1D(128, k=3) → BN → ReLU | Conv1d(256→128, k=3) → BN → ReLU | ✔ |
| FCN pool | GlobalAveragePooling1D | `AdaptiveAvgPool1d(1)` | ✔ |
| merge | `concatenate([lstm, fcn])` → Dense(n_classes, softmax) | `Linear(128+128, n_classes)` on `cat([lstm_feat, conv_feat])` | ✔ |
| squeeze-excite | **absent** in univariate LSTM-FCN (present only in MLSTM-FCN) | absent | ✔ |

### Differences

- **LSTM cell count = 128.** Karim trains and reports `{8, 64, 128}` per dataset
  and picks the best; 128 is the largest paper configuration. `tsai` default is
  100. Fixed at 128 here (no per-dataset tuning). This is a hyperparameter, not
  an architecture change.
- **Kernel sizes {8, 5, 3}** — matches Karim's paper and Fawaz's FCN. `tsai`
  uses {7, 5, 3} to avoid the even-kernel `same`-padding pad-copy; the warning is
  benign here (as in `cnn_lstm.py`).
- `bias=False` on FCN convs (cancelled by the following BatchNorm).
- Keras `he_uniform` conv init vs PyTorch default Kaiming-uniform — both
  He-style; not corrected.

### Training config (`config.yaml: training.rnn_fcn`)

Paper: Adam, batch 128, up to 2000 epochs, `ReduceLROnPlateau`. This repo: Adam,
`lr=1e-3`, batch 128, `epochs=500`, `patience=40`,
`ReduceLROnPlateau(factor=0.5, patience=15, min_lr=1e-6)`, `weight_decay=1e-4`.

Verified: `python models/rnn_fcn.py` → `[8, 4]` logits, softmax sums to 1;
param counts 587,780 (T=500) / 1,099,780 (T=1500).

---

## What "comparable results" means here

The **architectures** reproduce the references layer-for-layer (ResNet, RNN-FCN)
or block-for-block plus the published weight init (TCN). Absolute accuracy will
still differ from the papers' UCR/UEA numbers because:

1. Different data — a synthetic bifurcation SDE corpus + Mediterranean sediment
   cores, not the UCR archive.
2. Different input conditioning — single-channel right-aligned left-padded
   residuals with AR(1) left-censoring augmentation (shared DL harness).
3. Different training regime — one shared schedule (`epochs`, `patience`,
   scheduler) across all DL models rather than the papers' per-dataset tuning.

The point of matching the architectures is that **relative** comparison within
this thesis (ResNet vs. TCN vs. RNN-FCN vs. CNN-LSTM vs. InceptionTime vs.
PatchTST) is on the same footing as the literature, and any gap to published
numbers is attributable to data/harness, not to a mis-built model.
