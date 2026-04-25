"""LSTM sequence classifier for soccer event detection."""
from __future__ import annotations

import torch
import torch.nn as nn
from ml.features import FEATURE_DIM, NUM_CLASSES


class LSTMEventClassifier(nn.Module):
    """Two-layer bidirectional LSTM → LayerNorm → Dropout → Linear.

    Input:  (batch, seq_len, input_size)
    Output: (batch, num_classes) — raw logits
    """

    def __init__(
        self,
        input_size: int = FEATURE_DIM,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        lstm_out_dim = hidden_size * self.num_directions
        self.norm = nn.LayerNorm(lstm_out_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_out_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, features)
        lstm_out, _ = self.lstm(x)
        # Take the last time step's output
        last_out = lstm_out[:, -1, :]
        out = self.norm(last_out)
        out = self.dropout(out)
        return self.fc(out)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities. Use for inference only."""
        with torch.no_grad():
            return torch.softmax(self.forward(x), dim=-1)
