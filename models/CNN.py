import torch
import torch.nn as nn


class CNNClassifier(nn.Module):
    """
    1D CNN classifier for sequence classification.

    Input:
      x: (batch_size, seq_len, input_size) = (B, T, F)

    Internally:
      Conv1d expects (B, C, L) so we permute to (B, F, T).

    Output:
      logits: (batch_size, num_classes)
    """

    def __init__(
        self,
        input_size=2,
        num_classes=4,
        channels=(64, 128),
        kernel_size=7,
        dropout=0.2,
        use_batchnorm=True,
        global_pool="avg",  # "avg" or "max"
    ):
        super().__init__()

        if isinstance(channels, (list, tuple)):
            channels = list(channels)
        else:
            # allow passing a single int
            channels = [int(channels)]

        layers = []
        in_ch = input_size

        for out_ch in channels:
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            layers.append(nn.MaxPool1d(kernel_size=2))
            in_ch = out_ch

        self.features = nn.Sequential(*layers)

        if global_pool not in ("avg", "max"):
            raise ValueError("global_pool must be 'avg' or 'max'")
        self.global_pool = global_pool

        self.classifier = nn.Sequential(
            nn.Linear(in_ch, in_ch // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_ch // 2, num_classes),
        )

    def forward(self, x):
        # x: (B, T, F) -> (B, F, T)
        x = x.permute(0, 2, 1)

        x = self.features(x)  # (B, C, L')

        if self.global_pool == "avg":
            x = x.mean(dim=2)  # (B, C)
        else:
            x = x.amax(dim=2)  # (B, C)

        return self.classifier(x)


if __name__ == "__main__":
    model = CNNClassifier(input_size=2, num_classes=4)
    x = torch.randn(8, 500, 2)
    y = model(x)
    print("Output shape:", y.shape)  # (8, 4)
