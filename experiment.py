from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from baseline_model import ConstantVelocityPredictor, VanillaLSTMEncoderDecoder
from data.missingness import build_model_inputs, build_motion_model_inputs
from data.trajectory_dataset import SCENE_SPLITS, TrajectoryDataset
from project_model import MissingnessAwareLSTM, MissingnessAwareTransformer
from utils.metrics import metric_dict
from utils.plotting import save_trajectory_plot
from utils.trajectory_ops import constant_velocity_forecast


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "preprocessed" / "datasets_LMTrajectory"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results"
MIXED_TRAIN_CONDITIONS = (
    {"mode": "random", "drop_rate": 0.1, "contiguous_len": 3},
    {"mode": "random", "drop_rate": 0.3, "contiguous_len": 3},
    {"mode": "random", "drop_rate": 0.5, "contiguous_len": 3},
    {"mode": "partial", "drop_rate": 0.3, "contiguous_len": 3},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate trajectory prediction models.")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--split", default="zara1", choices=SCENE_SPLITS)
    parser.add_argument(
        "--model",
        default="vanilla_lstm",
        choices=("constant_velocity", "vanilla_lstm", "lstm", "missing_lstm", "missing_transformer"),
    )
    parser.add_argument("--obs-len", type=int, default=8)
    parser.add_argument("--pred-len", type=int, default=12)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
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
    parser.add_argument("--missing-mode", default="complete", choices=("complete", "random", "contiguous", "partial"))
    parser.add_argument("--drop-rate", type=float, default=0.0)
    parser.add_argument("--contiguous-len", type=int, default=3)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--mixed-missingness", action="store_true")
    return parser.parse_args()


def canonical_model_name(model_name: str) -> str:
    return "vanilla_lstm" if model_name == "lstm" else model_name


def make_loader(args: argparse.Namespace, phase: str, shuffle: bool) -> DataLoader:
    dataset = TrajectoryDataset(
        dataset_root=args.dataset_root,
        split=args.split,
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


def model_inputs(
    obs: torch.Tensor,
    args: argparse.Namespace,
    missing_aware: bool,
    mode: str | None = None,
    drop_rate: float | None = None,
    contiguous_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mode = args.missing_mode if mode is None else mode
    drop_rate = args.drop_rate if drop_rate is None else drop_rate
    contiguous_len = args.contiguous_len if contiguous_len is None else contiguous_len
    if missing_aware and args.feature_mode == "motion":
        return build_motion_model_inputs(obs, mode, drop_rate, contiguous_len)
    features, last_pos, mask = build_model_inputs(
        obs,
        mode,
        drop_rate,
        contiguous_len,
        missing_aware=missing_aware,
    )
    return features, last_pos, mask, features[..., :2]


@torch.no_grad()
def evaluate(model, loader: DataLoader, args: argparse.Namespace, missing_aware: bool) -> dict[str, float]:
    if isinstance(model, nn.Module):
        model.eval()
    total_ade = 0.0
    total_fde = 0.0
    count = 0
    sample = None

    for batch in loader:
        obs = batch["obs"].to(args.device)
        target = batch["pred"].to(args.device)
        features, last_pos, mask, cv_obs = model_inputs(obs, args, missing_aware)
        if isinstance(model, ConstantVelocityPredictor):
            pred = model.predict(cv_obs, mask)
        elif missing_aware:
            pred = model(features, last_pos, mask=mask, cv_obs=cv_obs)
        else:
            pred = model(features, last_pos)
        metrics = metric_dict(pred, target)
        batch_size = obs.shape[0]
        total_ade += metrics["ADE"] * batch_size
        total_fde += metrics["FDE"] * batch_size
        count += batch_size
        if sample is None:
            sample = (obs[0], mask[0], target[0], pred[0])

    results = {"ADE": total_ade / count, "FDE": total_fde / count, "n": count}
    if args.plot and sample is not None:
        output_path = Path(args.output_dir) / f"{args.split}_{canonical_model_name(args.model)}_{args.missing_mode}.png"
        save_trajectory_plot(output_path, *sample, title=json.dumps(results))
        results["plot"] = str(output_path)
    return results


def train(args: argparse.Namespace) -> dict[str, float]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = canonical_model_name(args.model)
    train_loader = make_loader(args, "train", shuffle=True)
    val_loader = make_loader(args, "val", shuffle=False)
    test_loader = make_loader(args, "test", shuffle=False)

    if model_name == "constant_velocity":
        model = ConstantVelocityPredictor(pred_len=args.pred_len)
        results = evaluate(model, test_loader, args, missing_aware=False)
        print(json.dumps({"phase": "test", "split": args.split, "model": model_name, **results}, indent=2))
        return results

    missing_aware = model_name in {"missing_lstm", "missing_transformer"}
    if model_name == "missing_lstm":
        model = MissingnessAwareLSTM(
            input_dim=8 if args.feature_mode == "motion" else 4,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            pred_len=args.pred_len,
            dropout=args.dropout,
        ).to(args.device)
    elif model_name == "missing_transformer":
        model = MissingnessAwareTransformer(
            input_dim=8 if args.feature_mode == "motion" else 4,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers if args.num_layers > 1 else 2,
            num_heads=args.num_heads,
            pred_len=args.pred_len,
            dropout=args.dropout,
        ).to(args.device)
    else:
        model = VanillaLSTMEncoderDecoder(
            encoder_input_dim=2,
            output_dim=2,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            pred_len=args.pred_len,
            dropout=args.dropout,
        ).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss() if args.loss == "huber" else nn.MSELoss()
    best_val = float("inf")
    best_path = output_dir / f"{args.split}_{model_name}_best.pt"

    for epoch in range(1, args.epochs + 1):
        if args.teacher_forcing_decay and args.epochs > 1:
            teacher_forcing_ratio = args.teacher_forcing_ratio * (1.0 - (epoch - 1) / (args.epochs - 1))
        else:
            teacher_forcing_ratio = args.teacher_forcing_ratio
        model.train()
        total_loss = 0.0
        count = 0
        for batch in train_loader:
            obs = batch["obs"].to(args.device)
            target = batch["pred"].to(args.device)
            if missing_aware and args.mixed_missingness:
                train_params = random.choice(MIXED_TRAIN_CONDITIONS)
                train_mode = train_params["mode"]
                train_drop_rate = train_params["drop_rate"]
                train_contiguous_len = train_params["contiguous_len"]
            else:
                train_mode = args.missing_mode if missing_aware else "complete"
                train_drop_rate = args.drop_rate if missing_aware else 0.0
                train_contiguous_len = args.contiguous_len
            features, last_pos, mask, cv_obs = model_inputs(
                obs,
                args,
                missing_aware,
                mode=train_mode,
                drop_rate=train_drop_rate,
                contiguous_len=train_contiguous_len,
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
            loss = loss_fn(pred, target)
            if missing_aware and args.residual_weight > 0:
                cv_base = constant_velocity_forecast(cv_obs, mask, args.pred_len)
                loss = loss + args.residual_weight * (pred - cv_base).abs().mean()
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += float(loss.item()) * obs.shape[0]
            count += obs.shape[0]

        val_results = evaluate(model, val_loader, args, missing_aware=missing_aware)
        train_loss = total_loss / count
        print(
            json.dumps(
                {
                    "phase": "val",
                    "epoch": epoch,
                    "split": args.split,
                    "model": model_name,
                    "train_loss": train_loss,
                    "teacher_forcing_ratio": teacher_forcing_ratio,
                    **val_results,
                }
            )
        )
        if val_results["ADE"] < best_val:
            best_val = val_results["ADE"]
            torch.save({"model_state": model.state_dict(), "args": vars(args)}, best_path)

    checkpoint = torch.load(best_path, map_location=args.device)
    model.load_state_dict(checkpoint["model_state"])
    results = evaluate(model, test_loader, args, missing_aware=missing_aware)
    print(json.dumps({"phase": "test", "split": args.split, "model": model_name, **results}, indent=2))
    return results


if __name__ == "__main__":
    train(parse_args())
