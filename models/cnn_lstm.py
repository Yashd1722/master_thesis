"""
models/cnn_lstm.py — CNN-LSTM hybrid, reproduction of Bury et al. (PNAS 2021).

Conv1D(50, k=12, same) -> ReLU -> Dropout(.1) -> MaxPool(2)
-> LSTM(50) -> Dropout(.1) -> LSTM(10) -> Dropout(.1) -> Linear(num_classes)
Input (B, 1, T) -> logits (B, num_classes). Dropout is 10% (Bury: higher hurt F1).
"""
import torch
import torch.nn as nn


class CNNLSTM(nn.Module):

    def __init__(self, ts_len: int = 500, num_classes: int = 4):
        super().__init__()
        self.ts_len      = ts_len
        self.num_classes = num_classes
        self.conv  = nn.Conv1d(1, 50, kernel_size=12, padding="same")
        self.relu  = nn.ReLU()
        self.drop1 = nn.Dropout(0.10)
        self.pool  = nn.MaxPool1d(2)
        self.lstm1 = nn.LSTM(50, 50, batch_first=True)
        self.drop2 = nn.Dropout(0.10)
        self.lstm2 = nn.LSTM(50, 10, batch_first=True)
        self.drop3 = nn.Dropout(0.10)
        self.fc    = nn.Linear(10, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.drop1(self.relu(self.conv(x))))  # (B, 50, T//2)
        x = x.permute(0, 2, 1)                              # (B, T//2, 50)
        x, _ = self.lstm1(x)
        x = self.drop2(x)
        x, _ = self.lstm2(x)
        x = self.drop3(x)
        return self.fc(x[:, -1, :])                         # last step -> logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x), dim=-1)
