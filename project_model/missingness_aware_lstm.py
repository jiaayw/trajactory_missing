from __future__ import annotations

from baseline_model.models import VanillaLSTMEncoderDecoder
from utils.trajectory_ops import constant_velocity_forecast


class MissingnessAwareLSTM(VanillaLSTMEncoderDecoder):
    """LSTM that consumes x, y, observed-mask, and time-gap features."""

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 1,
        pred_len: int = 12,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            encoder_input_dim=input_dim,
            output_dim=2,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            pred_len=pred_len,
            dropout=dropout,
        )

    def forward(
        self,
        features,
        last_pos,
        target=None,
        teacher_forcing_ratio: float = 0.0,
        mask=None,
        cv_obs=None,
    ):
        obs = cv_obs if cv_obs is not None else features[..., :2]
        if mask is None:
            mask = features[..., 2:3]
        cv_base = constant_velocity_forecast(obs, mask, self.pred_len)
        residual_target = target - cv_base if target is not None else None
        residuals = super().forward(
            features,
            last_pos.new_zeros(last_pos.shape),
            target=residual_target,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        return cv_base + residuals
