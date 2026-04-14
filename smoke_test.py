from __future__ import annotations

import argparse
from pathlib import Path

import torch

from baseline_model import ConstantVelocityPredictor, VanillaLSTMEncoderDecoder
from data.missingness import build_model_inputs, build_motion_model_inputs
from data.trajectory_dataset import TrajectoryDataset
from project_model import MissingnessAwareLSTM, MissingnessAwareTransformer
from utils.metrics import metric_dict


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "preprocessed" / "datasets_LMTrajectory"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick non-training checks for milestone 1.")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--split", default="zara1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = TrajectoryDataset(Path(args.dataset_root), args.split, "train")
    sample = dataset[0]
    assert tuple(sample["obs"].shape) == (8, 2)
    assert tuple(sample["pred"].shape) == (12, 2)
    assert torch.all(sample["frames"][1:] > sample["frames"][:-1])

    obs = torch.stack([dataset[i]["obs"] for i in range(min(16, len(dataset)))])
    target = torch.stack([dataset[i]["pred"] for i in range(min(16, len(dataset)))])

    features, last_pos, mask = build_model_inputs(obs, mode="random", drop_rate=0.3, missing_aware=True)
    assert tuple(features.shape) == (obs.shape[0], 8, 4)
    assert tuple(mask.shape) == (obs.shape[0], 8, 1)
    motion_features, motion_last_pos, motion_mask, filled = build_motion_model_inputs(
        obs,
        mode="random",
        drop_rate=0.3,
    )
    assert tuple(motion_features.shape) == (obs.shape[0], 8, 8)
    assert tuple(motion_mask.shape) == (obs.shape[0], 8, 1)
    assert tuple(filled.shape) == (obs.shape[0], 8, 2)

    cv_pred = ConstantVelocityPredictor(pred_len=12).predict(features[..., :2], mask)
    vanilla_pred = VanillaLSTMEncoderDecoder(pred_len=12)(obs, obs[:, -1, :])
    proposed_pred = MissingnessAwareLSTM(pred_len=12)(features, last_pos)
    transformer_pred = MissingnessAwareTransformer(hidden_dim=32, num_layers=2, num_heads=4, pred_len=12)(features, last_pos)
    motion_transformer_pred = MissingnessAwareTransformer(input_dim=8, hidden_dim=32, num_layers=2, num_heads=4, pred_len=12)(
        motion_features,
        motion_last_pos,
        mask=motion_mask,
        cv_obs=filled,
    )
    assert tuple(cv_pred.shape) == tuple(target.shape)
    assert tuple(vanilla_pred.shape) == tuple(target.shape)
    assert tuple(proposed_pred.shape) == tuple(target.shape)
    assert tuple(transformer_pred.shape) == tuple(target.shape)
    assert tuple(motion_transformer_pred.shape) == tuple(target.shape)

    print(
        {
            "windows": len(dataset),
            "sample_obs_shape": tuple(sample["obs"].shape),
            "sample_pred_shape": tuple(sample["pred"].shape),
            "constant_velocity": metric_dict(cv_pred, target),
            "vanilla_lstm_forward_shape": tuple(vanilla_pred.shape),
            "proposed_forward_shape": tuple(proposed_pred.shape),
            "transformer_forward_shape": tuple(transformer_pred.shape),
            "motion_feature_shape": tuple(motion_features.shape),
            "motion_transformer_forward_shape": tuple(motion_transformer_pred.shape),
        }
    )


if __name__ == "__main__":
    main()
