from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)

        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape:
            (batch, window, d_model)
        """
        return x + self.pe[:, : x.size(1)]


class TranAD(nn.Module):
    """
    TranAD-style Transformer reconstruction model.

    Input:
        x: (batch, window_size, input_dim)

    Output:
        rec1_last: first-stage reconstruction of the last timestep
        rec2_last: second-stage anomaly-focused reconstruction of the last timestep

    The second stage receives an anomaly-conditioning signal based on
    first-stage reconstruction error.
    """

    def __init__(
        self,
        input_dim: int,
        window_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 1,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")

        self.input_dim = input_dim
        self.window_size = window_size
        self.d_model = d_model

        self.encoder_input_projection = nn.Linear(input_dim * 2, d_model)
        self.decoder_input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=window_size + 10)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        decoder_layer_1 = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        decoder_layer_2 = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.decoder_1 = nn.TransformerDecoder(
            decoder_layer=decoder_layer_1,
            num_layers=num_layers,
        )

        self.decoder_2 = nn.TransformerDecoder(
            decoder_layer=decoder_layer_2,
            num_layers=num_layers,
        )

        self.output_layer_1 = nn.Linear(d_model, input_dim)
        self.output_layer_2 = nn.Linear(d_model, input_dim)

    def _encode(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        encoder_input = torch.cat([x, context], dim=-1)
        encoder_input = self.encoder_input_projection(encoder_input)
        encoder_input = self.positional_encoding(encoder_input)
        memory = self.encoder(encoder_input)
        return memory

    def _decode_stage_1(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        decoder_input = self.decoder_input_projection(x)
        decoder_input = self.positional_encoding(decoder_input)
        decoded = self.decoder_1(tgt=decoder_input, memory=memory)
        return self.output_layer_1(decoded)

    def _decode_stage_2(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        decoder_input = self.decoder_input_projection(x)
        decoder_input = self.positional_encoding(decoder_input)
        decoded = self.decoder_2(tgt=decoder_input, memory=memory)
        return self.output_layer_2(decoded)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        zero_context = torch.zeros_like(x)

        memory_1 = self._encode(x, zero_context)
        rec1_full = self._decode_stage_1(x, memory_1)

        anomaly_context = torch.square(rec1_full.detach() - x)

        memory_2 = self._encode(x, anomaly_context)
        rec2_full = self._decode_stage_2(x, memory_2)

        rec1_last = rec1_full[:, -1, :]
        rec2_last = rec2_full[:, -1, :]

        return rec1_last, rec2_last