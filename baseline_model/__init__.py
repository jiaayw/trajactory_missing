"""Baseline trajectory prediction models."""

from baseline_model.models import ConstantVelocityPredictor, VanillaLSTMEncoderDecoder

__all__ = ["ConstantVelocityPredictor", "VanillaLSTMEncoderDecoder"]
