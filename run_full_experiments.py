from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader

from baseline_model import ConstantVelocityPredictor, VanillaLSTMEncoderDecoder
from data.missingness import build_model_inputs, build_motion_model_inputs
from data.trajectory_dataset import SCENE_SPLITS, TrajectoryDataset
from navigation import decide_navigation_action
from project_model import MissingnessAwareLSTM, MissingnessAwareTransformer
from utils.metrics import metric_dict
from utils.plotting import save_grid_navigation_plot
from utils.trajectory_ops import constant_velocity_forecast


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "preprocessed" / "datasets_LMTrajectory"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results"

MISSINGNESS_CONDITIONS = {
    "complete": {"mode": "complete", "drop_rate": 0.0, "contiguous_len": 3},
    "random_0.1": {"mode": "random", "drop_rate": 0.1, "contiguous_len": 3},
    "random_0.3": {"mode": "random", "drop_rate": 0.3, "contiguous_len": 3},
    "random_0.5": {"mode": "random", "drop_rate": 0.5, "contiguous_len": 3},
    "contiguous": {"mode": "contiguous", "drop_rate": 0.0, "contiguous_len": 3},
    "partial": {"mode": "partial", "drop_rate": 0.3, "contiguous_len": 3},
}
MIXED_TRAIN_CONDITIONS = ("random_0.1", "random_0.3", "random_0.5", "partial")
MISSING_AWARE_VAL_CONDITIONS = ("random_0.3", "random_0.5", "partial")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run robustness and navigation demo experiments.")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--splits", nargs="+", default=["zara1", "zara2"], choices=SCENE_SPLITS)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["constant_velocity", "vanilla_lstm", "missing_lstm", "missing_transformer"],
        choices=("constant_velocity", "vanilla_lstm", "missing_lstm", "missing_transformer"),
    )
    parser.add_argument("--conditions", nargs="+", default=list(MISSINGNESS_CONDITIONS))
    parser.add_argument("--obs-len", type=int, default=8)
    parser.add_argument("--pred-len", type=int, default=12)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--loss", default="mse", choices=("mse", "huber"))
    parser.add_argument("--residual-weight", type=float, default=0.01)
    parser.add_argument("--feature-mode", default="motion", choices=("basic", "motion"))
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--teacher-forcing-ratio", type=float, default=0.5)
    parser.add_argument("--teacher-forcing-decay", action="store_true")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-plot-rows", type=int, default=8)
    parser.add_argument("--plot-sample", default="median_ade", choices=("first", "median_ade", "worst_ade", "stop_case"))
    return parser.parse_args()


def make_loader(args: argparse.Namespace, split: str, phase: str, shuffle: bool) -> DataLoader:
    dataset = TrajectoryDataset(
        dataset_root=args.dataset_root,
        split=split,
        phase=phase,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        stride=args.stride,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        drop_last=False,
    )


def make_model(args: argparse.Namespace, model_name: str):
    feature_mode = getattr(args, "feature_mode", "basic")
    if model_name == "constant_velocity":
        return ConstantVelocityPredictor(pred_len=args.pred_len)
    if model_name == "vanilla_lstm":
        return VanillaLSTMEncoderDecoder(
            encoder_input_dim=2,
            output_dim=2,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            pred_len=args.pred_len,
            dropout=args.dropout,
        ).to(args.device)
    if model_name == "missing_lstm":
        return MissingnessAwareLSTM(
            input_dim=8 if feature_mode == "motion" else 4,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            pred_len=args.pred_len,
            dropout=args.dropout,
        ).to(args.device)
    if model_name == "missing_transformer":
        return MissingnessAwareTransformer(
            input_dim=8 if feature_mode == "motion" else 4,
            hidden_dim=args.hidden_dim,
            num_layers=args.transformer_layers,
            num_heads=args.num_heads,
            pred_len=args.pred_len,
            dropout=args.dropout,
        ).to(args.device)
    raise ValueError(f"Unknown model: {model_name}")


def condition_namespace(args: argparse.Namespace, condition_name: str) -> SimpleNamespace:
    params = MISSINGNESS_CONDITIONS[condition_name]
    return SimpleNamespace(
        missing_mode=params["mode"],
        drop_rate=params["drop_rate"],
        contiguous_len=params["contiguous_len"],
        device=args.device,
        output_dir=args.output_dir,
    )


def missingness_params(condition_name: str) -> dict[str, float | int | str]:
    return MISSINGNESS_CONDITIONS[condition_name]


def model_inputs(
    obs: torch.Tensor,
    mode: str,
    drop_rate: float,
    contiguous_len: int,
    missing_aware: bool,
    feature_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if missing_aware and feature_mode == "motion":
        return build_motion_model_inputs(obs, mode, drop_rate, contiguous_len)
    features, last_pos, mask = build_model_inputs(
        obs,
        mode,
        drop_rate,
        contiguous_len,
        missing_aware=missing_aware,
    )
    return features, last_pos, mask, features[..., :2]


def prediction_loss_with_regularization(
    pred: torch.Tensor,
    target: torch.Tensor,
    cv_obs: torch.Tensor,
    mask: torch.Tensor,
    pred_len: int,
    loss_fn,
    residual_weight: float,
) -> torch.Tensor:
    loss = loss_fn(pred, target)
    if residual_weight <= 0:
        return loss
    cv_base = constant_velocity_forecast(cv_obs, mask, pred_len)
    residual_penalty = (pred - cv_base).abs().mean()
    return loss + residual_weight * residual_penalty


def validation_score(model, val_loader: DataLoader, args: argparse.Namespace, model_name: str) -> float:
    if model_name not in {"missing_lstm", "missing_transformer"}:
        val_metrics, _ = evaluate_model(model, val_loader, args, model_name, "complete", save_plot=False)
        return float(val_metrics["ADE"])
    scores = []
    for condition_name in MISSING_AWARE_VAL_CONDITIONS:
        val_metrics, _ = evaluate_model(model, val_loader, args, model_name, condition_name, save_plot=False)
        scores.append(float(val_metrics["ADE"]))
    return sum(scores) / len(scores)


def train_neural_model(args: argparse.Namespace, model, train_loader: DataLoader, val_loader: DataLoader, model_name: str):
    missing_aware = model_name in {"missing_lstm", "missing_transformer"}

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss() if args.loss == "huber" else nn.MSELoss()
    best_state = None
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        if args.teacher_forcing_decay and args.epochs > 1:
            teacher_forcing_ratio = args.teacher_forcing_ratio * (1.0 - (epoch - 1) / (args.epochs - 1))
        else:
            teacher_forcing_ratio = args.teacher_forcing_ratio

        model.train()
        for batch in train_loader:
            obs = batch["obs"].to(args.device)
            target = batch["pred"].to(args.device)
            train_condition = random.choice(MIXED_TRAIN_CONDITIONS) if missing_aware else "complete"
            train_params = missingness_params(train_condition)
            features, last_pos, mask, cv_obs = model_inputs(
                obs,
                train_params["mode"],
                train_params["drop_rate"],
                train_params["contiguous_len"],
                missing_aware=missing_aware,
                feature_mode=getattr(args, "feature_mode", "basic"),
            )
            if missing_aware:
                pred = model(
                    features,
                    last_pos,
                    target=target,
                    teacher_forcing_ratio=teacher_forcing_ratio,
                    mask=mask,
                    cv_obs=cv_obs,
                )
            else:
                pred = model(features, last_pos, target=target, teacher_forcing_ratio=teacher_forcing_ratio)
            if missing_aware:
                loss = prediction_loss_with_regularization(
                    pred,
                    target,
                    cv_obs,
                    mask,
                    args.pred_len,
                    loss_fn,
                    args.residual_weight,
                )
            else:
                loss = loss_fn(pred, target)
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        val_score = validation_score(model, val_loader, args, model_name)
        if val_score < best_val:
            best_val = val_score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def evaluate_model(
    model,
    loader: DataLoader,
    args: argparse.Namespace,
    model_name: str,
    condition_name: str,
    save_plot: bool,
    split: str = "unknown",
) -> tuple[dict[str, float | str], dict[str, torch.Tensor | str] | None]:
    missing_aware = model_name in {"missing_lstm", "missing_transformer"}
    params = condition_namespace(args, condition_name)
    if isinstance(model, nn.Module):
        model.eval()

    total_ade = 0.0
    total_fde = 0.0
    count = 0
    sample = None
    action_counts = {"stop": 0, "go": 0, "turn_left": 0, "turn_right": 0}
    correct_actions = 0
    true_stop = 0
    pred_stop = 0
    true_positive_stop = 0
    plot_candidates = []

    for batch in loader:
        obs = batch["obs"].to(args.device)
        target = batch["pred"].to(args.device)
        features, last_pos, mask, cv_obs = model_inputs(
            obs,
            params.missing_mode,
            params.drop_rate,
            params.contiguous_len,
            missing_aware=missing_aware,
            feature_mode=getattr(args, "feature_mode", "basic"),
        )
        if isinstance(model, ConstantVelocityPredictor):
            pred = model.predict(cv_obs, mask)
        elif missing_aware:
            pred = model(features, last_pos, mask=mask, cv_obs=cv_obs)
        else:
            pred = model(features, last_pos)

        metrics = metric_dict(pred, target)
        per_sample_ade = torch.linalg.norm(pred - target, dim=-1).mean(dim=1)
        batch_size = obs.shape[0]
        total_ade += metrics["ADE"] * batch_size
        total_fde += metrics["FDE"] * batch_size
        count += batch_size

        for row, (pred_row, target_row) in enumerate(zip(pred, target)):
            pred_action = decide_navigation_action(pred_row).action
            true_action = decide_navigation_action(target_row).action
            action_counts[pred_action] += 1
            correct_actions += int(pred_action == true_action)
            true_stop += int(true_action == "stop")
            pred_stop += int(pred_action == "stop")
            true_positive_stop += int(pred_action == "stop" and true_action == "stop")
            if save_plot:
                candidate = {
                    "obs": obs[row].detach().cpu(),
                    "mask": mask[row].detach().cpu(),
                    "target": target_row.detach().cpu(),
                    "pred": pred_row.detach().cpu(),
                    "action": pred_action,
                    "pred_action": pred_action,
                    "true_action": true_action,
                    "ade": float(per_sample_ade[row].detach().cpu().item()),
                }
                plot_candidates.append(candidate)
                if sample is None:
                    sample = candidate

    result = {
        "split": split,
        "model": model_name,
        "missingness": condition_name,
        "ADE": total_ade / count,
        "FDE": total_fde / count,
        "n": count,
        "stop": action_counts["stop"],
        "go": action_counts["go"],
        "turn_left": action_counts["turn_left"],
        "turn_right": action_counts["turn_right"],
        "action": max(action_counts, key=action_counts.get),
        "action_accuracy": correct_actions / count,
        "stop_precision": true_positive_stop / pred_stop if pred_stop else 0.0,
        "stop_recall": true_positive_stop / true_stop if true_stop else 0.0,
        "true_stop": true_stop,
        "pred_stop": pred_stop,
    }

    if save_plot and plot_candidates:
        plot_sample = getattr(args, "plot_sample", "median_ade")
        if plot_sample == "first":
            sample = plot_candidates[0]
        elif plot_sample == "worst_ade":
            sample = max(plot_candidates, key=lambda item: item["ade"])
        elif plot_sample == "stop_case":
            stop_candidates = [
                item
                for item in plot_candidates
                if item["pred_action"] == "stop" or item["true_action"] == "stop"
            ]
            if stop_candidates:
                sample = min(stop_candidates, key=lambda item: item["ade"])
            else:
                sorted_candidates = sorted(plot_candidates, key=lambda item: item["ade"])
                sample = sorted_candidates[len(sorted_candidates) // 2]
        else:
            sorted_candidates = sorted(plot_candidates, key=lambda item: item["ade"])
            sample = sorted_candidates[len(sorted_candidates) // 2]

        plot_path = (
            Path(args.output_dir)
            / "plots"
            / f"{split}_{model_name}_{condition_name}_navigation.png"
        )
        save_grid_navigation_plot(
            plot_path,
            sample["obs"],
            sample["mask"],
            sample["target"],
            sample["pred"],
            sample["action"],
        )
        result["plot"] = str(plot_path)
    else:
        result["plot"] = ""

    return result, sample


def write_results(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "model",
        "missingness",
        "ADE",
        "FDE",
        "n",
        "stop",
        "go",
        "turn_left",
        "turn_right",
        "action",
        "action_accuracy",
        "stop_precision",
        "stop_recall",
        "true_stop",
        "pred_stop",
        "plot",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | str]] = []
    saved_plots = 0
    for split in args.splits:
        train_loader = make_loader(args, split, "train", shuffle=True)
        val_loader = make_loader(args, split, "val", shuffle=False)
        test_loader = make_loader(args, split, "test", shuffle=False)

        for model_name in args.models:
            model = make_model(args, model_name)
            if not isinstance(model, ConstantVelocityPredictor):
                model = train_neural_model(args, model, train_loader, val_loader, model_name)

            for condition_name in args.conditions:
                save_plot = saved_plots < args.max_plot_rows
                result, _ = evaluate_model(
                    model,
                    test_loader,
                    args,
                    model_name,
                    condition_name,
                    save_plot=save_plot,
                    split=split,
                )
                if save_plot:
                    saved_plots += 1
                rows.append(result)
                print(result)

    output_path = output_dir / "full_experiment_results.csv"
    write_results(output_path, rows)
    print(f"Saved {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
