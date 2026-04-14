from __future__ import annotations

import os
from pathlib import Path

import torch


def save_trajectory_plot(
    path: str | Path,
    obs: torch.Tensor,
    mask: torch.Tensor,
    target: torch.Tensor,
    pred: torch.Tensor,
    title: str = "Trajectory prediction",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mpl_config_dir = path.parent / ".mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(mpl_config_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    obs = obs.detach().cpu()
    mask = mask.detach().cpu().squeeze(-1).bool()
    target = target.detach().cpu()
    pred = pred.detach().cpu()

    plt.figure(figsize=(6, 5))
    plt.plot(target[:, 0], target[:, 1], "g-o", label="future target")
    plt.plot(pred[:, 0], pred[:, 1], "r-o", label="prediction")
    plt.plot(obs[mask, 0], obs[mask, 1], "b-o", label="observed")
    if (~mask).any():
        plt.scatter(obs[~mask, 0], obs[~mask, 1], c="gray", marker="x", label="missing")
    plt.axis("equal")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_navigation_plot(
    path: str | Path,
    obs: torch.Tensor,
    mask: torch.Tensor,
    target: torch.Tensor,
    pred: torch.Tensor,
    action: str,
    safety_half_width: float = 1.0,
    safety_depth: float = 3.0,
    title: str = "Navigation decision demo",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mpl_config_dir = path.parent / ".mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(mpl_config_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    obs = obs.detach().cpu()
    mask = mask.detach().cpu().squeeze(-1).bool()
    target = target.detach().cpu()
    pred = pred.detach().cpu()

    plt.figure(figsize=(7, 6))
    ax = plt.gca()
    safety_zone = Rectangle(
        (-safety_half_width, 0),
        2 * safety_half_width,
        safety_depth,
        facecolor="red",
        alpha=0.12,
        edgecolor="red",
        label="safety zone",
    )
    ax.add_patch(safety_zone)
    plt.scatter([0], [0], c="black", marker="^", s=90, label="robot")
    plt.plot(target[:, 0], target[:, 1], "g-o", label="future target")
    plt.plot(pred[:, 0], pred[:, 1], "r-o", label="prediction")
    plt.plot(obs[mask, 0], obs[mask, 1], "b-o", label="observed")
    if (~mask).any():
        plt.scatter(obs[~mask, 0], obs[~mask, 1], c="gray", marker="x", label="missing")
    plt.axhline(0, color="black", linewidth=0.8, alpha=0.35)
    plt.axvline(0, color="black", linewidth=0.8, alpha=0.35)
    plt.axis("equal")
    plt.title(f"{title}: {action}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_grid_navigation_plot(
    path: str | Path,
    obs: torch.Tensor,
    mask: torch.Tensor,
    target: torch.Tensor,
    pred: torch.Tensor,
    action: str,
    safety_half_width: float = 1.0,
    safety_depth: float = 3.0,
    title: str = "Grid navigation demo",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mpl_config_dir = path.parent / ".mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(mpl_config_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    obs = obs.detach().cpu()
    mask = mask.detach().cpu().squeeze(-1).bool()
    target = target.detach().cpu()
    pred = pred.detach().cpu()

    all_points = torch.cat([obs, target, pred, torch.zeros(1, 2)], dim=0)
    min_x = min(float(all_points[:, 0].min().item()), -safety_half_width) - 1.0
    max_x = max(float(all_points[:, 0].max().item()), safety_half_width) + 1.0
    min_y = min(float(all_points[:, 1].min().item()), 0.0) - 1.0
    max_y = max(float(all_points[:, 1].max().item()), safety_depth) + 1.0
    span = max(max_x - min_x, max_y - min_y, 6.0)
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    half = span / 2.0
    min_x, max_x = cx - half, cx + half
    min_y, max_y = cy - half, cy + half

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_facecolor("#f7f8fb")
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect("equal", adjustable="box")

    x_start = int(min_x) - 1
    x_end = int(max_x) + 1
    y_start = int(min_y) - 1
    y_end = int(max_y) + 1
    for ix in range(x_start, x_end):
        for iy in range(y_start, y_end):
            color = "#eef1f5" if (ix + iy) % 2 == 0 else "#ffffff"
            ax.add_patch(Rectangle((ix, iy), 1, 1, facecolor=color, edgecolor="none", zorder=0))
    ax.grid(color="#c8ced8", linewidth=0.8, alpha=0.75)

    safety_zone = Rectangle(
        (-safety_half_width, 0),
        2 * safety_half_width,
        safety_depth,
        facecolor="#ff4d4d",
        alpha=0.16,
        edgecolor="#d91f26",
        linewidth=2,
        label="safety zone",
        zorder=1,
    )
    ax.add_patch(safety_zone)
    ax.scatter([0], [0], c="#111111", marker="^", s=170, label="robot/camera", zorder=5)

    ax.plot(target[:, 0], target[:, 1], color="#2ca02c", marker="o", linewidth=2.4, label="true future", zorder=3)
    ax.plot(pred[:, 0], pred[:, 1], color="#d62728", marker="o", linewidth=2.4, label="predicted future", zorder=4)
    if mask.any():
        ax.plot(obs[mask, 0], obs[mask, 1], color="#1f77b4", marker="o", linewidth=2.4, label="observed", zorder=4)
    if (~mask).any():
        ax.scatter(obs[~mask, 0], obs[~mask, 1], c="#666666", marker="x", s=90, linewidths=2.5, label="missing", zorder=6)

    action_colors = {
        "stop": "#d91f26",
        "go": "#238823",
        "turn_left": "#c26d00",
        "turn_right": "#c26d00",
    }
    badge_color = action_colors.get(action, "#333333")
    ax.text(
        0.02,
        0.98,
        f"Action: {action}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        color="white",
        fontsize=13,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": badge_color, "edgecolor": "none"},
    )
    ax.set_title(title)
    ax.set_xlabel("x position")
    ax.set_ylabel("y position")
    ax.legend(loc="lower right", framealpha=0.92)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
