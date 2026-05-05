"""
models/cnn_multihead.py
=======================
Multi-head CNN classifier — Ma et al. (2025).

Architecture:
    4 parallel convolutional branches with kernel sizes [3, 5, 11, 21].
    Each branch captures patterns at a different timescale:
      k=3  → very local (3 timesteps)
      k=5  → local (5 timesteps)
      k=11 → medium (11 timesteps)
      k=21 → long-range (21 timesteps)

    Each branch:
        Conv1D(64, kernel=k, padding=same)
        → BatchNorm1D → ReLU
        → GlobalAvgPool1D  (reduces to 64 scalars regardless of T)

    Concatenate all branches: 64 × 4 = 256 features
    → Dropout(0.50)
    → Linear(256 → 128) → ReLU
    → Linear(128 → num_classes)

Key advantage over single-kernel CNN:
    Multi-scale feature extraction is more robust to varying sampling rates
    and geological noise. Ma 2025 found this architecture performed best
    on 64PE406E1 and MS66 cores.

Input  : (B, 1, T)
Output : (B, num_classes) — raw logits
"""
MODEL_NAME  = "multihead_cnn"
MODEL_CLASS = "MultiHeadCNN"
IS_SKLEARN  = False


import torch
import torch.nn as nn


class ConvBranch(nn.Module):
    """Single convolutional branch for one kernel size."""

    def __init__(self, kernel_size: int, n_filters: int = 64):
        super().__init__()
        self.conv  = nn.Conv1d(
            in_channels  = 1,
            out_channels = n_filters,
            kernel_size  = kernel_size,
            padding      = "same",
        )
        self.bn    = nn.BatchNorm1d(n_filters)
        self.relu  = nn.ReLU()
        # Global average pool: (B, n_filters, T) → (B, n_filters)
        self.pool  = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn(self.conv(x)))   # (B, n_filters, T)
        x = self.pool(x).squeeze(-1)            # (B, n_filters)
        return x


class MultiHeadCNN(nn.Module):

    def __init__(self, ts_len: int = 500, num_classes: int = 4,
                 kernel_sizes: tuple = (3, 5, 11, 21),
                 n_filters: int = 64):
        super().__init__()
        self.ts_len      = ts_len
        self.num_classes = num_classes
        self.kernel_sizes = kernel_sizes
        self.n_filters    = n_filters

        # ── Parallel branches ─────────────────────────────────────────────────
        self.branches = nn.ModuleList([
            ConvBranch(k, n_filters) for k in kernel_sizes
        ])

        # ── Shared head ───────────────────────────────────────────────────────
        concat_dim = n_filters * len(kernel_sizes)   # 64 × 4 = 256

        self.dropout = nn.Dropout(p=0.50)
        self.fc1     = nn.Linear(concat_dim, 128)
        self.relu    = nn.ReLU()
        self.fc2     = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 1, T)

        Returns
        -------
        logits : (B, num_classes)
        """
        # Run all branches in parallel, concatenate outputs
        branch_outs = [branch(x) for branch in self.branches]  # [(B, 64), ...]
        x = torch.cat(branch_outs, dim=1)                        # (B, 256)

        # Shared dense head
        x = self.dropout(x)
        x = self.relu(self.fc1(x))                               # (B, 128)
        return self.fc2(x)                                        # (B, num_classes)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns softmax probabilities. Used by evaluate.py."""
        return torch.softmax(self.forward(x), dim=-1)


if __name__ == "__main__":
    for ts_len, n_cls in [(500, 4), (1500, 4), (500, 2)]:
        model  = MultiHeadCNN(ts_len=ts_len, num_classes=n_cls)
        x      = torch.randn(8, 1, ts_len)
        logits = model(x)
        probs  = model.predict_proba(x)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"MultiHeadCNN | ts_len={ts_len:4d} | num_classes={n_cls} | "
            f"logits={tuple(logits.shape)} | "
            f"params={params:,}"
        )
