import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class _Chomp1d(nn.Module):

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x


class _TemporalBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size,
                                           padding=pad, dilation=dilation))
        self.chomp1 = _Chomp1d(pad)
        self.relu1  = nn.ReLU()
        self.drop1  = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size,
                                           padding=pad, dilation=dilation))
        self.chomp2 = _Chomp1d(pad)
        self.relu2  = nn.ReLU()
        self.drop2  = nn.Dropout(dropout)
        self.downsample = (nn.Conv1d(in_channels, out_channels, 1)
                           if in_channels != out_channels else None)
        self.relu = nn.ReLU()
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y = self.drop1(self.relu1(self.chomp1(self.conv1(x))))
        y = self.drop2(self.relu2(self.chomp2(self.conv2(y))))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(y + res)


class TCN(nn.Module):

    def __init__(self, ts_len: int = 500, num_classes: int = 4,
                 channels: int = 64, levels: int = 8, kernel_size: int = 7,
                 dropout: float = 0.2):
        super().__init__()
        self.ts_len      = ts_len
        self.num_classes = num_classes
        layers = []
        in_ch = 1
        for i in range(levels):
            layers.append(_TemporalBlock(in_ch, channels, kernel_size,
                                         dilation=2 ** i, dropout=dropout))
            in_ch = channels
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.network(x)
        return self.fc(x[:, :, -1])

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x), dim=-1)


if __name__ == "__main__":
    for ts_len in (500, 1500):
        net = TCN(ts_len=ts_len, num_classes=4)
        out = net(torch.randn(8, 1, ts_len))
        assert out.shape == (8, 4), out.shape
        assert torch.allclose(net.predict_proba(torch.randn(3, 1, ts_len)).sum(-1),
                              torch.ones(3), atol=1e-5)
        print(f"TCN ts_len={ts_len} ok  params={sum(p.numel() for p in net.parameters()):,}")
