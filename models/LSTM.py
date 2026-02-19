import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """
    Simple LSTM classifier for sequence classification.

    Input:
      x: (batch_size, seq_len, input_size)

    Output:
      logits: (batch_size, num_classes)
    """

    def __init__(
        self,
        input_size=2,
        hidden_size=128,
        num_layers=2,
        num_classes=4,
        dropout=0.2,
        bidirectional=False,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,         # input shape: (B, T, F)
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_out_size = hidden_size * (2 if bidirectional else 1)

        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_size, lstm_out_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_size // 2, num_classes),
        )

    def forward(self, x):
        # x shape: (B, T, F)
        out, (h_n, c_n) = self.lstm(x)

        # h_n shape: (num_layers * num_directions, B, hidden_size)
        # Use the last layer's hidden state
        if self.lstm.bidirectional:
            # last layer forward is -2, backward is -1
            h_last = torch.cat((h_n[-2], h_n[-1]), dim=1)  # (B, hidden*2)
        else:
            h_last = h_n[-1]  # (B, hidden)

        logits = self.classifier(h_last)
        return logits


# Optional quick test: run this file directly
if __name__ == "__main__":
    model = LSTMClassifier(input_size=2, num_classes=4)
    x = torch.randn(8, 500, 2)  # (batch, time, features)
    y = model(x)
    print("Output shape:", y.shape)  # should be (8, 4)
