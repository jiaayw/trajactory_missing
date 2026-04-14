from __future__ import annotations

import torch


def make_observation_mask(
    obs: torch.Tensor,
    mode: str = "complete",
    drop_rate: float = 0.0,
    contiguous_len: int = 3,
) -> torch.Tensor:
    """Return a [batch, obs_len, 1] mask where 1 means observed."""
    if obs.ndim != 3:
        raise ValueError("obs must have shape [batch, obs_len, 2]")
    batch, obs_len, _ = obs.shape
    device = obs.device
    mask = torch.ones(batch, obs_len, 1, device=device)

    if mode == "complete" or drop_rate <= 0:
        return mask
    if mode == "random":
        mask = (torch.rand(batch, obs_len, 1, device=device) > drop_rate).float()
        mask[:, 0, :] = 1.0
        return mask
    if mode == "contiguous":
        length = max(1, min(contiguous_len, obs_len - 1))
        max_start = max(1, obs_len - length)
        starts = torch.randint(1, max_start + 1, (batch,), device=device)
        for row, start in enumerate(starts.tolist()):
            mask[row, start : start + length, :] = 0.0
        return mask
    if mode == "partial":
        missing = max(1, min(int(round(obs_len * drop_rate)), obs_len - 1))
        mask[:, obs_len - missing :, :] = 0.0
        return mask

    raise ValueError("mode must be one of: complete, random, contiguous, partial")


def carry_forward(obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Fill missing observed points with the latest available coordinate."""
    if mask.shape[:2] != obs.shape[:2]:
        raise ValueError("mask must match obs batch and time dimensions")

    filled = obs.clone()
    last = filled[:, 0, :]
    for t in range(obs.shape[1]):
        observed = mask[:, t, :].bool()
        last = torch.where(observed, obs[:, t, :], last)
        filled[:, t, :] = last
    return filled


def missing_gap_features(mask: torch.Tensor) -> torch.Tensor:
    """Return normalized time-since-last-observed values with shape [batch, obs_len, 1]."""
    gaps = torch.zeros_like(mask)
    running = torch.zeros(mask.shape[0], 1, device=mask.device)
    denom = max(1, mask.shape[1] - 1)
    for t in range(mask.shape[1]):
        running = torch.where(mask[:, t, :] > 0, torch.zeros_like(running), running + 1.0)
        gaps[:, t, :] = running / denom
    return gaps


def build_model_inputs(
    obs: torch.Tensor,
    mode: str = "complete",
    drop_rate: float = 0.0,
    contiguous_len: int = 3,
    missing_aware: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = make_observation_mask(obs, mode, drop_rate, contiguous_len)
    filled = carry_forward(obs, mask)
    if missing_aware:
        features = torch.cat([filled, mask, missing_gap_features(mask)], dim=-1)
    else:
        features = filled
    last_pos = filled[:, -1, :]
    return features, last_pos, mask


def motion_features(filled: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Build relative position, velocity, acceleration, mask, and gap features."""
    if filled.ndim != 3 or filled.shape[-1] != 2:
        raise ValueError("filled must have shape [batch, obs_len, 2]")
    if mask.shape[:2] != filled.shape[:2]:
        raise ValueError("mask must match filled batch and time dimensions")

    rel_pos = filled - filled[:, -1:, :]

    velocity = torch.zeros_like(filled)
    velocity[:, 1:, :] = filled[:, 1:, :] - filled[:, :-1, :]

    acceleration = torch.zeros_like(filled)
    acceleration[:, 1:, :] = velocity[:, 1:, :] - velocity[:, :-1, :]

    return torch.cat(
        [rel_pos, velocity, acceleration, mask, missing_gap_features(mask)],
        dim=-1,
    )


def build_motion_model_inputs(
    obs: torch.Tensor,
    mode: str = "complete",
    drop_rate: float = 0.0,
    contiguous_len: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return 8-D motion features plus absolute filled observations for CV base."""
    mask = make_observation_mask(obs, mode, drop_rate, contiguous_len)
    filled = carry_forward(obs, mask)
    features = motion_features(filled, mask)
    last_pos = filled[:, -1, :]
    return features, last_pos, mask, filled
