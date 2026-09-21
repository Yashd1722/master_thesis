import torch
import torch.nn as nn


class _ResidualBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=8, padding="same", bias=False)
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding="same", bias=False)
        self.bn2   = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding="same", bias=False)
        self.bn3   = nn.BatchNorm1d(out_channels)
        if in_channels == out_channels:
            self.shortcut = nn.BatchNorm1d(out_channels)
        else:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.relu(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(y))
        return self.relu(y + self.shortcut(x))


class ResNet(nn.Module):

    def __init__(self, ts_len: int = 500, num_classes: int = 4, n_feature_maps: int = 64):
        super().__init__()
        self.ts_len      = ts_len
        self.num_classes = num_classes
        self.block1 = _ResidualBlock(1, n_feature_maps)
        self.block2 = _ResidualBlock(n_feature_maps, n_feature_maps * 2)
        self.block3 = _ResidualBlock(n_feature_maps * 2, n_feature_maps * 2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc  = nn.Linear(n_feature_maps * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x), dim=-1)


if __name__ == "__main__":
    for ts_len in (500, 1500):
        net = ResNet(ts_len=ts_len, num_classes=4)
        out = net(torch.randn(8, 1, ts_len))
        assert out.shape == (8, 4), out.shape
        assert torch.allclose(net.predict_proba(torch.randn(3, 1, ts_len)).sum(-1),
                              torch.ones(3), atol=1e-5)
        print(f"ResNet ts_len={ts_len} ok  params={sum(p.numel() for p in net.parameters()):,}")
