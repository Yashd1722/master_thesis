"""
models/cnn_lstm.py
==================
CNN-LSTM hybrid — exact reproduction of Bury et al. (PNAS 2021).

Architecture (Fig. 3 of Bury 2021):
    Conv1D(50 filters, kernel=12, padding=same)
    → ReLU
    → Dropout(0.10)
    → MaxPool1D(kernel=2)
    → LSTM(50 hidden)
    → Dropout(0.10)
    → LSTM(10 hidden)
    → Dropout(0.10)
    → Linear(num_classes)

Key design choices from the paper:
  - Conv1D reads local temporal patterns (kernel=12 spans ~2.4% of ts_500)
  - MaxPool halves the sequence length before LSTM
  - Two stacked LSTMs: first captures broad patterns, second refines
  - Dropout=10% (much lower than typical — Bury found higher dropout hurt F1)
  - No batch normalisation (Bury did not use it)
  - Trained with 1500 epochs, lr=0.0005, Adam optimiser

Input  : (B, 1, T)        — batch, channel, time
Output : (B, num_classes) — raw logits, use CrossEntropyLoss during training
"""

import torch
import torch.nn as nn


class CNNLSTM(nn.Module):

    def __init__(self, ts_len: int = 500, num_classes: int = 4):
        super().__init__()
        self.ts_len      = ts_len
        self.num_classes = num_classes

        # ── CNN block ─────────────────────────────────────────────────────────
        # padding="same" keeps sequence length = ts_len after convolution
        self.conv = nn.Conv1d(
            in_channels  = 1,
            out_channels = 50,
            kernel_size  = 12,
            padding      = "same",   # output length = input length
        )
        self.relu    = nn.ReLU()
        self.drop1   = nn.Dropout(p=0.10)
        self.pool    = nn.MaxPool1d(kernel_size=2)
        # After pool: sequence length = ts_len // 2, features = 50

        # ── LSTM block ────────────────────────────────────────────────────────
        self.lstm1 = nn.LSTM(
            input_size  = 50,
            hidden_size = 50,
            batch_first = True,      # input: (B, seq_len, features)
        )
        self.drop2 = nn.Dropout(p=0.10)

        self.lstm2 = nn.LSTM(
            input_size  = 50,
            hidden_size = 10,
            batch_first = True,
        )
        self.drop3 = nn.Dropout(p=0.10)

        # ── Output ────────────────────────────────────────────────────────────
        self.fc = nn.Linear(10, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 1, T)

        Returns
        -------
        logits : (B, num_classes)
        """
        # CNN block: (B, 1, T) → (B, 50, T//2)
        x = self.conv(x)                   # (B, 50, T)
        x = self.relu(x)
        x = self.drop1(x)
        x = self.pool(x)                   # (B, 50, T//2)

        # Permute for LSTM: (B, 50, T//2) → (B, T//2, 50)
        x = x.permute(0, 2, 1)

        # LSTM block
        x, _ = self.lstm1(x)               # (B, T//2, 50)
        x    = self.drop2(x)
        x, _ = self.lstm2(x)               # (B, T//2, 10)
        x    = self.drop3(x)

        # Take last time step
        x = x[:, -1, :]                    # (B, 10)

        return self.fc(x)                  # (B, num_classes)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns softmax probabilities.
        Used by evaluate.py during inference — never during training.

        Parameters
        ----------
        x : (B, 1, T)

        Returns
        -------
        probs : (B, num_classes)  values sum to 1 along dim=1
        """
        return torch.softmax(self.forward(x), dim=-1)


if __name__ == "__main__":
    # Quick architecture check
    for ts_len, n_cls in [(500, 4), (1500, 4), (500, 2)]:
        model  = CNNLSTM(ts_len=ts_len, num_classes=n_cls)
        x      = torch.randn(8, 1, ts_len)
        logits = model(x)
        probs  = model.predict_proba(x)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"CNNLSTM | ts_len={ts_len:4d} | num_classes={n_cls} | "
            f"logits={tuple(logits.shape)} | "
            f"probs={tuple(probs.shape)} | "
            f"params={params:,}"
        )
