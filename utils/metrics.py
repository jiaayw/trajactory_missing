from __future__ import annotations

import torch


def ade(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(pred - target, dim=-1).mean()


def fde(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(pred[:, -1, :] - target[:, -1, :], dim=-1).mean()


def metric_dict(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    return {"ADE": float(ade(pred, target).item()), "FDE": float(fde(pred, target).item())}

