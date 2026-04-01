"""
models/cnn.py
=============
CNN classifier — exact architecture from Ma et al. (2025) Methods section:

    "We use a CNN with two 1D convolutional layers with 64 filters,
     a kernel size of three, and the ReLU activation function.
     They are followed by a dropout layer with a rate of 0.5
     and a MaxPooling layer to reduce the dimension.
     Then, we use a dense layer with the ReLU activation function
     and 128 neurons. Finally, we use a dense layer to the output
     classes with a sigmoid activation function."

Architecture:
    Conv1D(64, kernel=3, padding=same) → ReLU
    → Conv1D(64, kernel=3, padding=same) → ReLU
    → Dropout(0.50)
    → MaxPool1D(kernel=2)
    → Flatten
    → Linear(64 × T//2 → 128) → ReLU
    → Linear(128 → num_classes)

Key difference from CNNLSTM (Bury):
  - Two Conv layers instead of one (kernel=3 vs kernel=12)
  - No LSTM — purely convolutional + dense
  - Higher dropout (0.50 vs 0.10)
  - Kernel=3 captures very local patterns (3 timesteps at a time)

Input  : (B, 1, T)
Output : (B, num_classes) — raw logits
"""

import torch
import torch.nn as nn


class CNNClassifier(nn.Module):

    def __init__(self, ts_len: int = 500, num_classes: int = 4):
        super().__init__()
        self.ts_len      = ts_len
        self.num_classes = num_classes

        # ── Two Conv1D layers (Ma 2025) ───────────────────────────────────────
        self.conv1 = nn.Conv1d(
            in_channels  = 1,
            out_channels = 64,
            kernel_size  = 3,
            padding      = "same",   # output length stays = T
        )
        self.conv2 = nn.Conv1d(
            in_channels  = 64,
            out_channels = 64,
            kernel_size  = 3,
            padding      = "same",
        )
        self.relu  = nn.ReLU()
        self.drop  = nn.Dropout(p=0.50)
        self.pool  = nn.MaxPool1d(kernel_size=2)
        # After pool: (B, 64, T//2)

        # ── Dense head ────────────────────────────────────────────────────────
        # Flatten (B, 64, T//2) → (B, 64 × T//2)
        flat_dim  = 64 * (ts_len // 2)

        self.fc1   = nn.Linear(flat_dim, 128)
        self.fc2   = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 1, T)

        Returns
        -------
        logits : (B, num_classes)
        """
        # Conv block
        x = self.relu(self.conv1(x))     # (B, 64, T)
        x = self.relu(self.conv2(x))     # (B, 64, T)
        x = self.drop(x)
        x = self.pool(x)                 # (B, 64, T//2)

        # Flatten
        x = x.flatten(start_dim=1)      # (B, 64 × T//2)

        # Dense head
        x = self.relu(self.fc1(x))      # (B, 128)
        return self.fc2(x)              # (B, num_classes)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns softmax probabilities. Used by evaluate.py."""
        return torch.softmax(self.forward(x), dim=-1)


if __name__ == "__main__":
    for ts_len, n_cls in [(500, 4), (1500, 4), (500, 2)]:
        model  = CNNClassifier(ts_len=ts_len, num_classes=n_cls)
        x      = torch.randn(8, 1, ts_len)
        logits = model(x)
        probs  = model.predict_proba(x)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"CNN      | ts_len={ts_len:4d} | num_classes={n_cls} | "
            f"logits={tuple(logits.shape)} | "
            f"probs={tuple(probs.shape)} | "
            f"params={params:,}"
        )
