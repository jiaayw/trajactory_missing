from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


SCENE_SPLITS = ("eth", "hotel", "univ", "zara1", "zara2")
PHASES = ("train", "val", "test")


@dataclass(frozen=True)
class TrajectoryWindow:
    obs: np.ndarray
    pred: np.ndarray
    frames: np.ndarray
    ped_id: float
    source: str


def resolve_dataset_root(root: str | Path) -> Path:
    root_path = Path(root)
    if root_path.exists():
        return root_path

    typo_path = Path(str(root_path).replace("datasets_LMTrajectrory", "datasets_LMTrajectory"))
    if typo_path.exists():
        return typo_path

    raise FileNotFoundError(
        f"Dataset root not found: {root_path}. Expected a folder containing {SCENE_SPLITS}."
    )


def _load_txt(path: Path) -> np.ndarray:
    data = np.loadtxt(path, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 4:
        raise ValueError(f"{path} must have at least 4 columns: frame_id ped_id x y")
    return data[:, :4]


def _is_regular_window(frames: np.ndarray) -> bool:
    if len(frames) <= 2:
        return True
    diffs = np.diff(frames)
    return bool(np.all(diffs == diffs[0]))


def load_windows(
    dataset_root: str | Path,
    split: str,
    phase: str,
    obs_len: int = 8,
    pred_len: int = 12,
    stride: int = 1,
) -> list[TrajectoryWindow]:
    dataset_root = resolve_dataset_root(dataset_root)
    if split not in SCENE_SPLITS:
        raise ValueError(f"Unknown split {split!r}. Expected one of {SCENE_SPLITS}.")
    if phase not in PHASES:
        raise ValueError(f"Unknown phase {phase!r}. Expected one of {PHASES}.")

    phase_dir = dataset_root / split / phase
    if not phase_dir.exists():
        raise FileNotFoundError(f"Missing phase directory: {phase_dir}")

    total_len = obs_len + pred_len
    windows: list[TrajectoryWindow] = []
    for path in sorted(phase_dir.glob("*.txt")):
        rows = _load_txt(path)
        for ped_id in np.unique(rows[:, 1]):
            ped_rows = rows[rows[:, 1] == ped_id]
            order = np.argsort(ped_rows[:, 0], kind="mergesort")
            ped_rows = ped_rows[order]
            frames = ped_rows[:, 0]
            coords = ped_rows[:, 2:4]
            if len(coords) < total_len:
                continue

            for start in range(0, len(coords) - total_len + 1, stride):
                end = start + total_len
                frame_window = frames[start:end]
                if not _is_regular_window(frame_window):
                    continue
                windows.append(
                    TrajectoryWindow(
                        obs=coords[start : start + obs_len].copy(),
                        pred=coords[start + obs_len : end].copy(),
                        frames=frame_window.copy(),
                        ped_id=float(ped_id),
                        source=path.name,
                    )
                )
    return windows


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        phase: str,
        obs_len: int = 8,
        pred_len: int = 12,
        stride: int = 1,
    ) -> None:
        self.windows = load_windows(dataset_root, split, phase, obs_len, pred_len, stride)
        self.obs_len = obs_len
        self.pred_len = pred_len
        if not self.windows:
            raise ValueError(
                f"No trajectory windows found for split={split}, phase={phase}, "
                f"obs_len={obs_len}, pred_len={pred_len}."
            )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | float]:
        window = self.windows[index]
        return {
            "obs": torch.from_numpy(window.obs).float(),
            "pred": torch.from_numpy(window.pred).float(),
            "frames": torch.from_numpy(window.frames).float(),
            "ped_id": window.ped_id,
            "source": window.source,
        }

