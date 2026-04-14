from __future__ import annotations

import torch
from torch import nn


class ConstantVelocityPredictor:
    def __init__(self, pred_len: int = 12) -> None:
        self.pred_len = pred_len

    def predict(self, obs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            mask = torch.ones(obs.shape[0], obs.shape[1], 1, device=obs.device)
        preds = []
        for seq, seq_mask in zip(obs, mask):
            observed_idx = torch.where(seq_mask[:, 0] > 0)[0]
            if len(observed_idx) >= 2:
                last_idx = observed_idx[-1]
                prev_idx = observed_idx[-2]
                velocity = (seq[last_idx] - seq[prev_idx]) / (last_idx - prev_idx).clamp(min=1)
                last = seq[last_idx]
            else:
                last = seq[observed_idx[-1]] if len(observed_idx) else seq[-1]
                velocity = torch.zeros_like(last)
            steps = torch.arange(1, self.pred_len + 1, device=obs.device).float().unsqueeze(1)
            preds.append(last.unsqueeze(0) + steps * velocity.unsqueeze(0))
        return torch.stack(preds, dim=0)


class VanillaLSTMEncoderDecoder(nn.Module):
    """Vanilla LSTM encoder-decoder adapted from lkulowski/LSTM_encoder_decoder.

    The encoder reads the observed trajectory. The decoder starts from the last
    observed coordinate and recursively predicts future coordinates, with
    optional teacher forcing during training.
    """

    def __init__(
        self,
        encoder_input_dim: int = 2,
        output_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 1,
        pred_len: int = 12,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.output_dim = output_dim
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=encoder_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.decoder = nn.LSTM(
            input_size=output_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        features: torch.Tensor,
        last_pos: torch.Tensor,
        target: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        _, state = self.encoder(features)
        decoder_input = last_pos
        predictions = []

        for step in range(self.pred_len):
            decoder_output, state = self.decoder(decoder_input.unsqueeze(1), state)
            pred = self.output_layer(decoder_output.squeeze(1))
            predictions.append(pred)

            use_teacher_forcing = (
                self.training
                and target is not None
                and teacher_forcing_ratio > 0
                and torch.rand((), device=features.device).item() < teacher_forcing_ratio
            )
            decoder_input = target[:, step, :] if use_teacher_forcing else pred

        return torch.stack(predictions, dim=1)
