"""
models/inceptiontime.py

InceptionTime — modern DL baseline for time series classification.
From Fawaz et al. (2020) "InceptionTime: Finding AlexNet for Time Series
Classification", Data Mining and Knowledge Discovery.

Architecture: 3 stacked Inception modules + Global Average Pooling + Linear.
Each Inception module applies filters of 3 different lengths in parallel
plus a bottleneck + MaxPool passthrough, then concatenates all outputs.

This provides a fair modern DL comparison against the Bury (2021) CNN-LSTM.
Input (B, 1, T) -> logits (B, num_classes).
"""
import torch
import torch.nn as nn


class _InceptionModule(nn.Module):
    """One Inception module: 3 kernel-size branches + bottleneck MaxPool branch."""

    def __init__(self, in_channels: int, n_filters: int = 32,
                 kernel_sizes=(9, 19, 39), bottleneck_size: int = 32):
        super().__init__()

        # Bottleneck: reduce channels before large convolutions (saves compute)
        self.bottleneck = nn.Conv1d(in_channels, bottleneck_size,
                                    kernel_size=1, bias=False)

        # Three parallel convolutions at different scales
        self.convs = nn.ModuleList([
            nn.Conv1d(bottleneck_size, n_filters, kernel_size=k,
                      padding=k // 2, bias=False)
            for k in kernel_sizes
        ])

        # MaxPool branch: passthrough with bottleneck-shaped projection
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.mp_proj = nn.Conv1d(in_channels, n_filters, kernel_size=1, bias=False)

        self.bn   = nn.BatchNorm1d(n_filters * (len(kernel_sizes) + 1))
        self.relu = nn.ReLU()

    def forward(self, x):
        # Bottleneck path
        x_bn = self.bottleneck(x)
        branches = [conv(x_bn) for conv in self.convs]

        # MaxPool path (no bottleneck for the passthrough)
        branches.append(self.mp_proj(self.maxpool(x)))

        out = torch.cat(branches, dim=1)   # (B, n_filters*(K+1), T)
        return self.relu(self.bn(out))


class _ResidualBlock(nn.Module):
    """Shortcut connection every 3 Inception modules (Fawaz 2020 Fig. 2)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x, residual):
        return self.relu(x + self.shortcut(residual))


class InceptionTime(nn.Module):
    """
    InceptionTime — 3 Inception blocks with residual shortcuts.
    Fawaz et al. (2020), DOI: 10.1007/s10618-020-00710-y
    """

    def __init__(self, ts_len: int = 500, num_classes: int = 4,
                 n_filters: int = 32, depth: int = 3):
        super().__init__()
        self.ts_len      = ts_len
        self.num_classes = num_classes

        n_out = n_filters * 4   # 3 kernel branches + 1 maxpool branch

        modules = []
        in_ch = 1
        for _ in range(depth):
            modules.append(_InceptionModule(in_ch, n_filters=n_filters))
            in_ch = n_out

        self.inception_stack = nn.ModuleList(modules)

        # Residual shortcut (applied after all inception blocks)
        self.residual = _ResidualBlock(1, n_out)

        self.gap = nn.AdaptiveAvgPool1d(1)   # Global Average Pooling -> (B, n_out, 1)
        self.fc  = nn.Linear(n_out, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual_input = x
        for module in self.inception_stack:
            x = module(x)

        # Add shortcut from original input
        x = self.residual(x, residual_input)

        x = self.gap(x).squeeze(-1)   # (B, n_out)
        return self.fc(x)             # (B, num_classes)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns softmax probabilities. Used by evaluate.py."""
        return torch.softmax(self.forward(x), dim=-1)


if __name__ == "__main__":
    for ts_len, n_cls in [(500, 4), (1500, 4)]:
        model  = InceptionTime(ts_len=ts_len, num_classes=n_cls)
        x      = torch.randn(8, 1, ts_len)
        logits = model(x)
        probs  = model.predict_proba(x)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"InceptionTime | ts_len={ts_len} | logits={tuple(logits.shape)} | "
              f"params={params:,}")
