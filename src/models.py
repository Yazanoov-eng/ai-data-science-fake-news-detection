import torch
import torch.nn as nn

class CNNLSTMClassifier(nn.Module):
    """
    Model 1 (required): CNN + LSTM on top of BERT embeddings.
    Input x shape: (B, T, E) = (batch, seq_len, embed_dim)
    """
    def __init__(self, embed_dim: int, num_filters: int = 128, kernel_sizes=(3,4,5),
                 lstm_hidden_size: int = 128, lstm_num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k)
            for k in kernel_sizes
        ])

        self.lstm = nn.LSTM(
            input_size=num_filters * len(kernel_sizes),
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=False
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden_size, 2)

    def forward(self, x):
        # (B, T, E) -> (B, E, T)
        x = x.transpose(1, 2)

        pooled = []
        for conv in self.convs:
            y = torch.relu(conv(x))          # (B, F, T-k+1)
            y = torch.max(y, dim=2).values   # (B, F) global max pool
            pooled.append(y)

        feats = torch.cat(pooled, dim=1)     # (B, F*num_kernels)
        feats = feats.unsqueeze(1)           # (B, 1, F*num_kernels)

        out, _ = self.lstm(feats)            # (B, 1, H)
        h = out[:, -1, :]                    # (B, H)

        h = self.dropout(h)
        return self.fc(h)                    # (B, 2)
