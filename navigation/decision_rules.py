from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class NavigationDecision:
    action: str
    reason: str
    min_distance: float


def decide_navigation_action(
    pred: torch.Tensor,
    safety_half_width: float = 1.0,
    safety_depth: float = 3.0,
    side_close_depth: float = 4.0,
) -> NavigationDecision:
    """Map one predicted trajectory [pred_len, 2] to a high-level robot action."""
    if pred.ndim != 2 or pred.shape[-1] != 2:
        raise ValueError("pred must have shape [pred_len, 2]")

    pred_cpu = pred.detach().cpu()
    x = pred_cpu[:, 0]
    y = pred_cpu[:, 1]
    distances = torch.linalg.norm(pred_cpu, dim=-1)
    min_distance = float(distances.min().item())

    in_front_zone = (x.abs() <= safety_half_width) & (y >= 0) & (y <= safety_depth)
    if bool(in_front_zone.any()):
        return NavigationDecision("stop", "predicted path enters front safety zone", min_distance)

    close_ahead = (y >= 0) & (y <= side_close_depth)
    if bool(close_ahead.any()):
        mean_x = float(x[close_ahead].mean().item())
        if mean_x < -safety_half_width:
            return NavigationDecision("turn_right", "predicted path is close on the left", min_distance)
        if mean_x > safety_half_width:
            return NavigationDecision("turn_left", "predicted path is close on the right", min_distance)

    return NavigationDecision("go", "predicted path stays outside safety zone", min_distance)
