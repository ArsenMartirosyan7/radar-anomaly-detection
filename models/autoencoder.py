from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class FeedForwardAutoencoder(nn.Module):
    """
    Pointwise feed-forward autoencoder.

    Input:
        x: (batch, input_dim)

    Output:
        reconstruction: (batch, input_dim)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        if not hidden_dims:
            raise ValueError("hidden_dims must not be empty")

        encoder_layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        reversed_dims = list(reversed(hidden_dims[:-1])) + [input_dim]
        prev_dim = hidden_dims[-1]

        for i, hidden_dim in enumerate(reversed_dims):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))

            if i != len(reversed_dims) - 1:
                decoder_layers.append(nn.ReLU())
                decoder_layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction