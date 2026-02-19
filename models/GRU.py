import torch
import torch.nn as nn


class GRUClassifier(nn.Module):
    """
    GRU classifier for sequence classification.

    Input:
      x: (batch_size, seq_len, input_size) = (B, T, F)

    Output:
      logits: (batch_size, num_classes)
    """

    def __init__(
        self,
        input_size=2,
        num_classes=4,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        bidirectional=False,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        out_size = hidden_size * (2 if bidirectional else 1)

        self.classifier = nn.Sequential(
            nn.Linear(out_size, out_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_size // 2, num_classes),
        )

    def forward(self, x):
        # x: (B, T, F)
        out, h_n = self.gru(x)

        # h_n: (num_layers * num_directions, B, hidden_size)
        if self.gru.bidirectional:
            h_last = torch.cat((h_n[-2], h_n[-1]), dim=1)  # (B, 2*hidden)
        else:
            h_last = h_n[-1]  # (B, hidden)

        return self.classifier(h_last)


if __name__ == "__main__":
    model = GRUClassifier(input_size=2, num_classes=4)
    x = torch.randn(8, 500, 2)
    y = model(x)
    print("Output shape:", y.shape)  # (8, 4)
