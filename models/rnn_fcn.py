import torch
import torch.nn as nn


class RNN_FCN(nn.Module):

    def __init__(self, ts_len: int = 500, num_classes: int = 4,
                 lstm_hidden: int = 128, lstm_dropout: float = 0.8):
        super().__init__()
        self.ts_len      = ts_len
        self.num_classes = num_classes

        self.lstm = nn.LSTM(ts_len, lstm_hidden, batch_first=True)
        self.lstm_drop = nn.Dropout(lstm_dropout)

        self.conv1 = nn.Conv1d(1, 128, kernel_size=8, padding="same", bias=False)
        self.bn1   = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding="same", bias=False)
        self.bn2   = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 128, kernel_size=3, padding="same", bias=False)
        self.bn3   = nn.BatchNorm1d(128)
        self.relu  = nn.ReLU()
        self.gap   = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(lstm_hidden + 128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        lstm_feat = self.lstm_drop(lstm_out[:, -1, :])

        c = self.relu(self.bn1(self.conv1(x)))
        c = self.relu(self.bn2(self.conv2(c)))
        c = self.relu(self.bn3(self.conv3(c)))
        conv_feat = self.gap(c).squeeze(-1)

        return self.fc(torch.cat([lstm_feat, conv_feat], dim=1))

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x), dim=-1)


if __name__ == "__main__":
    for ts_len in (500, 1500):
        net = RNN_FCN(ts_len=ts_len, num_classes=4)
        out = net(torch.randn(8, 1, ts_len))
        assert out.shape == (8, 4), out.shape
        assert torch.allclose(net.predict_proba(torch.randn(3, 1, ts_len)).sum(-1),
                              torch.ones(3), atol=1e-5)
        print(f"RNN_FCN ts_len={ts_len} ok  params={sum(p.numel() for p in net.parameters()):,}")
