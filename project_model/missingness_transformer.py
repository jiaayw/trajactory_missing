from __future__ import annotations

import torch
from torch import nn

from utils.trajectory_ops import constant_velocity_forecast


class MissingnessAwareTransformer(nn.Module):
    """Compact non-LSTM model over x, y, mask, and time-gap features."""

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        pred_len: int = 12,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.pred_len = pred_len
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 8, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.residual_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len * 2),
        )
        self.gate_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len),
        )
        final_gate = self.gate_head[-1]
        if isinstance(final_gate, nn.Linear):
            nn.init.zeros_(final_gate.weight)
            nn.init.constant_(final_gate.bias, -2.0)

    def forward(
        self,
        features: torch.Tensor,
        last_pos: torch.Tensor,
        target: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
        mask: torch.Tensor | None = None,
        cv_obs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del last_pos, target, teacher_forcing_ratio
        obs_len = features.shape[1]
        if obs_len > self.pos_embedding.shape[1]:
            raise ValueError(
                f"obs_len={obs_len} exceeds learned positional capacity {self.pos_embedding.shape[1]}"
            )

        if mask is None:
            mask = features[..., 2:3]
        if cv_obs is None:
            cv_obs = features[..., :2]
        cv_base = constant_velocity_forecast(cv_obs, mask, self.pred_len)
        encoded = self.input_proj(features) + self.pos_embedding[:, :obs_len, :]
        encoded = self.encoder(encoded)
        pooled = torch.cat([encoded.mean(dim=1), encoded[:, -1, :]], dim=-1)
        residual = self.residual_head(pooled).reshape(features.shape[0], self.pred_len, 2)
        gate = torch.sigmoid(self.gate_head(pooled)).reshape(features.shape[0], self.pred_len, 1)
        return cv_base + gate * residual
