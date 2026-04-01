"""
models/lstm.py
==============
Standalone LSTM classifier.

Architecture matches Ma et al. (2025) exactly:
    Dense (input projection: 1 → 128)
    → LSTM(128 hidden)
    → Dropout(0.50)
    → LSTM(64 hidden)          ← second LSTM layer
    → Dense(128, ReLU)
    → Dense(num_classes, sigmoid/softmax)

Key differences from CNNLSTM:
  - No CNN layer — LSTM reads raw residuals directly
  - Higher dropout (0.50 vs 0.10) — Ma 2025 used this for binary classification
  - Input projection layer maps scalar input to 128-dim before LSTM
  - Two LSTM layers for depth without the CNN front-end

Input  : (B, 1, T)        — batch, channel, time
Output : (B, num_classes) — raw logits
"""

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):

    def __init__(self, ts_len: int = 500, num_classes: int = 4):
        super().__init__()
        self.ts_len      = ts_len
        self.num_classes = num_classes

        # Project scalar input to 128-dim (Ma 2025: "dense layer from input")
        self.input_proj = nn.Linear(1, 128)

        # First LSTM
        self.lstm1 = nn.LSTM(
            input_size  = 128,
            hidden_size = 128,
            batch_first = True,
        )
        self.drop1 = nn.Dropout(p=0.50)

        # Second LSTM
        self.lstm2 = nn.LSTM(
            input_size  = 128,
            hidden_size = 64,
            batch_first = True,
        )
        self.drop2 = nn.Dropout(p=0.50)

        # Output head
        self.fc1  = nn.Linear(64, 128)
        self.relu = nn.ReLU()
        self.fc2  = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 1, T)

        Returns
        -------
        logits : (B, num_classes)
        """
        # Reshape for LSTM: (B, 1, T) → (B, T, 1)
        x = x.permute(0, 2, 1)            # (B, T, 1)

        # Input projection: (B, T, 1) → (B, T, 128)
        x = self.input_proj(x)

        # LSTM layers
        x, _ = self.lstm1(x)              # (B, T, 128)
        x    = self.drop1(x)
        x, _ = self.lstm2(x)              # (B, T, 64)
        x    = self.drop2(x)

        # Take last time step
        x = x[:, -1, :]                   # (B, 64)

        # Output head
        x = self.relu(self.fc1(x))        # (B, 128)
        return self.fc2(x)                # (B, num_classes)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns softmax probabilities. Used by evaluate.py."""
        return torch.softmax(self.forward(x), dim=-1)


if __name__ == "__main__":
    for ts_len, n_cls in [(500, 4), (1500, 4), (500, 2)]:
        model  = LSTMClassifier(ts_len=ts_len, num_classes=n_cls)
        x      = torch.randn(8, 1, ts_len)
        logits = model(x)
        probs  = model.predict_proba(x)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"LSTM     | ts_len={ts_len:4d} | num_classes={n_cls} | "
            f"logits={tuple(logits.shape)} | "
            f"probs={tuple(probs.shape)} | "
            f"params={params:,}"
        )
