"""
models/lstm.py — standalone LSTM classifier (Ma et al. 2025).

Linear(1->128) -> LSTM(128) -> Dropout(.5) -> LSTM(64) -> Dropout(.5)
-> Linear(128, ReLU) -> Linear(num_classes). Input (B, 1, T) -> logits.
No CNN front-end; higher dropout (0.5) than the CNN-LSTM.
"""
import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):

    def __init__(self, ts_len: int = 500, num_classes: int = 4):
        super().__init__()
        self.ts_len      = ts_len
        self.num_classes = num_classes
        self.input_proj = nn.Linear(1, 128)
        self.lstm1 = nn.LSTM(128, 128, batch_first=True)
        self.drop1 = nn.Dropout(0.50)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True)
        self.drop2 = nn.Dropout(0.50)
        self.fc1   = nn.Linear(64, 128)
        self.relu  = nn.ReLU()
        self.fc2   = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x.permute(0, 2, 1))   # (B, T, 128)
        x, _ = self.lstm1(x)
        x = self.drop1(x)
        x, _ = self.lstm2(x)
        x = self.drop2(x)
        x = self.relu(self.fc1(x[:, -1, :]))
        return self.fc2(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x), dim=-1)
