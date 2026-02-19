import torch
import torch.nn as nn

class CNNLSTMClassifier(nn.Module):
    """
    CNN + LSTM classifier for sequence classification.
    The input should have shape (batch_size, seq_len, input_size) = (B, T, F).

    Default settings replicate the Keras example from the deep‑early‑warnings‑pnas repository:
    - One convolution layer with 50 filters and kernel_size=12 followed by dropout and max‑pooling.
    - Two LSTM layers with 50 and 10 hidden units respectively, with dropout between layers.

    You can customise the architecture via conv_channels, conv_kernel_size, pool_size,
    lstm_hidden_sizes, dropout and bidirectional flags.
    """

    def __init__(
        self,
        input_size: int = 2,
        num_classes: int = 4,
        conv_channels=(50,),       # list/tuple of output channels for each conv layer
        conv_kernel_size: int = 12,
        pool_size: int = 2,
        lstm_hidden_sizes=(50, 10),
        dropout: float = 0.10,
        bidirectional: bool = False,
    ):
        super().__init__()
        # Build convolutional front end (1D CNN)
        conv_blocks = []
        in_channels = input_size
        for out_channels in conv_channels:
            conv_blocks.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=conv_kernel_size,
                    padding=conv_kernel_size // 2,
                )
            )
            conv_blocks.append(nn.ReLU())
            conv_blocks.append(nn.Dropout(dropout))
            conv_blocks.append(nn.MaxPool1d(kernel_size=pool_size))
            in_channels = out_channels
        self.conv = nn.Sequential(*conv_blocks)

        # Build LSTM layers
        self.lstm_layers = nn.ModuleList()
        lstm_input_size = in_channels
        for hidden_size in lstm_hidden_sizes:
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=lstm_input_size,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                    dropout=dropout if hidden_size > 0 else 0.0,
                    bidirectional=bidirectional,
                )
            )
            lstm_input_size = hidden_size * (2 if bidirectional else 1)

        # Classifier: simple 2‑layer MLP on final hidden state
        fc_hidden = max(lstm_input_size // 2, num_classes)  # avoid 0 when hidden_size small
        self.classifier = nn.Sequential(
            nn.Linear(lstm_input_size, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes),
        )

    def forward(self, x):
        """
        x: (B, T, F)  -> conv expects (B, F, T), LSTM expects (B, T', C)
        Returns logits of shape (B, num_classes).
        """
        # (B, T, F) -> (B, F, T)
        x = x.permute(0, 2, 1)
        x = self.conv(x)  # (B, C, L')
        # Prepare for LSTM: (B, C, L') -> (B, L', C)
        x = x.permute(0, 2, 1)

        # Pass through LSTM layers
        for idx, lstm in enumerate(self.lstm_layers):
            x, (h_n, c_n) = lstm(x)
            # x remains (B, L', hidden)
            # We keep the full sequence for intermediate layers

        # h_n: (num_directions, B, hidden_size). Take last layer's hidden state.
        # If bidirectional, concatenate forward/backward last states.
        if self.lstm_layers[-1].bidirectional:
            h_last = torch.cat((h_n[-2], h_n[-1]), dim=1)  # (B, 2*hidden)
        else:
            h_last = h_n[-1]  # (B, hidden)

        return self.classifier(h_last)
