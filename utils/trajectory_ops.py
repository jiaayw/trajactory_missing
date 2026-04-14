from __future__ import annotations

import torch


def constant_velocity_forecast(obs: torch.Tensor, mask: torch.Tensor | None, pred_len: int) -> torch.Tensor:
    """Differentiable constant-velocity forecast with shape [batch, pred_len, 2]."""
    if obs.ndim != 3 or obs.shape[-1] != 2:
        raise ValueError("obs must have shape [batch, obs_len, 2]")
    if mask is None:
        mask = torch.ones(obs.shape[0], obs.shape[1], 1, device=obs.device, dtype=obs.dtype)
    if mask.shape[:2] != obs.shape[:2]:
        raise ValueError("mask must match obs batch and time dimensions")

    batch, obs_len, _ = obs.shape
    observed = mask[..., 0] > 0
    times = torch.arange(obs_len, device=obs.device, dtype=obs.dtype).unsqueeze(0).expand(batch, -1)
    observed_times = torch.where(observed, times, torch.full_like(times, -1.0))
    last_idx = observed_times.argmax(dim=1)
    last_valid_time = observed_times.gather(1, last_idx.unsqueeze(1)).squeeze(1)

    before_last = observed & (times < last_valid_time.unsqueeze(1))
    prev_times = torch.where(before_last, times, torch.full_like(times, -1.0))
    prev_idx = prev_times.argmax(dim=1)
    prev_valid_time = prev_times.gather(1, prev_idx.unsqueeze(1)).squeeze(1)

    batch_idx = torch.arange(batch, device=obs.device)
    last = obs[batch_idx, last_idx]
    prev = obs[batch_idx, prev_idx]
    has_prev = prev_valid_time >= 0
    dt = (last_valid_time - prev_valid_time).clamp(min=1.0).unsqueeze(1)
    velocity = torch.where(has_prev.unsqueeze(1), (last - prev) / dt, torch.zeros_like(last))

    steps = torch.arange(1, pred_len + 1, device=obs.device, dtype=obs.dtype).view(1, pred_len, 1)
    return last.unsqueeze(1) + steps * velocity.unsqueeze(1)
