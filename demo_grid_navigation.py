from __future__ import annotations

import argparse
import math
import time
from collections import deque
from pathlib import Path

import torch

from baseline_model import ConstantVelocityPredictor
from data.missingness import build_model_inputs, build_motion_model_inputs
from data.trajectory_dataset import SCENE_SPLITS, TrajectoryDataset
from demo_webcam_navigation import (
    build_live_inputs,
    detect_motion_center,
    detect_obstacle_zones,
    pixel_to_world,
)
from navigation import decide_navigation_action
from project_model import MissingnessAwareLSTM, MissingnessAwareTransformer


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "preprocessed" / "datasets_LMTrajectory"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "results" / "colab_zara2_missing_transformer_best.pt"
DEFAULT_SAVE_PATH = PROJECT_ROOT / "results" / "plots" / "grid_demo.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live webcam-to-chessboard navigation demo. Dataset replay is available with --replay-dataset."
    )
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--split", default="zara2", choices=SCENE_SPLITS)
    parser.add_argument("--phase", default="test", choices=("train", "val", "test"))
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--missing-mode", default="random", choices=("complete", "random", "contiguous", "partial"))
    parser.add_argument("--drop-rate", type=float, default=0.3)
    parser.add_argument("--contiguous-len", type=int, default=3)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--model", default="missing_transformer", choices=("constant_velocity", "missing_transformer", "missing_lstm"))
    parser.add_argument("--feature-mode", default="auto", choices=("auto", "basic", "motion"))
    parser.add_argument("--obs-len", type=int, default=8)
    parser.add_argument("--pred-len", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-path", default=str(DEFAULT_SAVE_PATH))
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--window-size", type=int, default=820)
    parser.add_argument("--fps-delay-ms", type=int, default=180)
    parser.add_argument("--live-webcam", action="store_true", help="Run the live camera-to-chessboard demo.")
    parser.add_argument("--replay-dataset", action="store_true", help="Replay one saved dataset trajectory on the grid.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--show-camera-inset", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pixels-per-meter", type=float, default=0.0)
    parser.add_argument("--min-contour-area", type=float, default=900.0)
    parser.add_argument("--warmup-frames", type=int, default=20)
    parser.add_argument("--live-sample-interval-sec", type=float, default=1.0)
    parser.add_argument("--prediction-refresh-sec", type=float, default=1.0)
    parser.add_argument("--live-smoothing-alpha", type=float, default=0.35)
    parser.add_argument("--min-motion-distance", type=float, default=0.20)
    parser.add_argument("--min-moving-samples", type=int, default=2)
    parser.add_argument("--motion-confidence-threshold", type=float, default=0.60)
    parser.add_argument("--stationary-default-action", default="stop", choices=("stop", "go"))
    parser.add_argument("--enable-obstacle-zones", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--obstacle-roi-start", type=float, default=0.55)
    parser.add_argument("--obstacle-threshold", type=float, default=0.025)
    parser.add_argument("--obstacle-min-area", type=float, default=1200.0)
    return parser.parse_args()


def checkpoint_state(path: Path) -> tuple[dict[str, torch.Tensor], dict]:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        return checkpoint["model_state"], checkpoint.get("args", checkpoint.get("config", {}))
    return checkpoint, {}


def infer_transformer_layers(state: dict[str, torch.Tensor], fallback: int) -> int:
    layers = set()
    for key in state:
        if key.startswith("encoder.layers."):
            parts = key.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                layers.add(int(parts[2]))
    return max(layers) + 1 if layers else fallback


def infer_lstm_layers(state: dict[str, torch.Tensor], fallback: int) -> int:
    layers = set()
    for key in state:
        if key.startswith("encoder.weight_ih_l"):
            layers.add(int(key.rsplit("l", 1)[-1]))
    return max(layers) + 1 if layers else fallback


def make_model(args: argparse.Namespace):
    if args.model == "constant_velocity":
        return ConstantVelocityPredictor(pred_len=args.pred_len), args.pred_len, "basic"

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state, checkpoint_args = checkpoint_state(checkpoint_path)
    pred_len = int(checkpoint_args.get("pred_len", args.pred_len))
    feature_mode = args.feature_mode

    if args.model == "missing_transformer":
        input_weight = state.get("input_proj.weight")
        input_dim = int(input_weight.shape[1]) if input_weight is not None else 4
        hidden_dim = int(input_weight.shape[0]) if input_weight is not None else args.hidden_dim
        if feature_mode == "auto":
            feature_mode = "motion" if input_dim == 8 else "basic"
        final_weight = state.get("residual_head.4.weight")
        if final_weight is not None:
            pred_len = int(final_weight.shape[0] // 2)
        model = MissingnessAwareTransformer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=infer_transformer_layers(state, args.transformer_layers),
            num_heads=args.num_heads,
            pred_len=pred_len,
            dropout=args.dropout,
        )
    else:
        hidden_weight = state.get("encoder.weight_hh_l0")
        input_weight = state.get("encoder.weight_ih_l0")
        hidden_dim = int(hidden_weight.shape[1]) if hidden_weight is not None else args.hidden_dim
        input_dim = int(input_weight.shape[1]) if input_weight is not None else 4
        if feature_mode == "auto":
            feature_mode = "motion" if input_dim == 8 else "basic"
        model = MissingnessAwareLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=infer_lstm_layers(state, args.num_layers),
            pred_len=pred_len,
            dropout=args.dropout,
        )

    model.load_state_dict(state, strict=False)
    model.to(args.device)
    model.eval()
    return model, pred_len, feature_mode


def build_inputs(obs: torch.Tensor, args: argparse.Namespace, feature_mode: str):
    if feature_mode == "motion":
        return build_motion_model_inputs(
            obs,
            mode=args.missing_mode,
            drop_rate=args.drop_rate,
            contiguous_len=args.contiguous_len,
        )
    features, last_pos, mask = build_model_inputs(
        obs,
        mode=args.missing_mode,
        drop_rate=args.drop_rate,
        contiguous_len=args.contiguous_len,
        missing_aware=args.model != "constant_velocity",
    )
    return features, last_pos, mask, features[..., :2]


@torch.no_grad()
def predict_sample(args: argparse.Namespace):
    dataset = TrajectoryDataset(
        args.dataset_root,
        args.split,
        args.phase,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
    )
    index = max(0, min(args.sample_index, len(dataset) - 1))
    sample = dataset[index]
    obs = sample["obs"].unsqueeze(0).to(args.device)
    target = sample["pred"].unsqueeze(0).to(args.device)
    model, pred_len, feature_mode = make_model(args)
    features, last_pos, mask, cv_obs = build_inputs(obs, args, feature_mode)

    if isinstance(model, ConstantVelocityPredictor):
        pred = model.predict(cv_obs, mask)
    else:
        pred = model(features, last_pos, mask=mask, cv_obs=cv_obs)

    action = decide_navigation_action(pred[0]).action
    true_action = decide_navigation_action(target[0]).action
    return {
        "index": index,
        "obs": obs[0].detach().cpu(),
        "mask": mask[0].detach().cpu(),
        "target": target[0].detach().cpu(),
        "pred": pred[0].detach().cpu(),
        "action": action,
        "true_action": true_action,
        "pred_len": pred_len,
        "feature_mode": feature_mode,
    }


def world_bounds(obs: torch.Tensor, target: torch.Tensor, pred: torch.Tensor) -> tuple[float, float, float, float]:
    points = torch.cat([obs, target, pred, torch.zeros(1, 2)], dim=0)
    min_x = min(float(points[:, 0].min().item()), -1.0) - 1.0
    max_x = max(float(points[:, 0].max().item()), 1.0) + 1.0
    min_y = min(float(points[:, 1].min().item()), 0.0) - 1.0
    max_y = max(float(points[:, 1].max().item()), 3.0) + 1.0
    span = max(max_x - min_x, max_y - min_y, 6.0)
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    half = span / 2.0
    return cx - half, cx + half, cy - half, cy + half


def live_world_bounds() -> tuple[float, float, float, float]:
    return -4.0, 4.0, -1.5, 6.5


def make_mapper(bounds: tuple[float, float, float, float], size: int, margin: int = 70):
    min_x, max_x, min_y, max_y = bounds
    scale = (size - 2 * margin) / max(max_x - min_x, max_y - min_y)

    def to_pixel(point: torch.Tensor | tuple[float, float]) -> tuple[int, int]:
        if isinstance(point, torch.Tensor):
            x_world = float(point[0])
            y_world = float(point[1])
        else:
            x_world, y_world = point
        x_pixel = int(round(margin + (x_world - min_x) * scale))
        y_pixel = int(round(size - margin - (y_world - min_y) * scale))
        return x_pixel, y_pixel

    return to_pixel


def draw_grid_frame(result: dict, size: int, reveal_step: int | None = None):
    import cv2
    import numpy as np

    obs = result["obs"]
    mask = result["mask"].squeeze(-1).bool()
    target = result.get("target")
    pred = result["pred"]
    action = result["action"]
    true_action = result.get("true_action")
    bounds = result.get("bounds") or world_bounds(obs, target, pred)
    to_pixel = make_mapper(bounds, size)

    frame = np.full((size, size, 3), 248, dtype=np.uint8)
    min_x, max_x, min_y, max_y = bounds
    for ix in range(int(min_x) - 1, int(max_x) + 1):
        for iy in range(int(min_y) - 1, int(max_y) + 1):
            p1 = to_pixel((ix, iy + 1))
            p2 = to_pixel((ix + 1, iy))
            color = (238, 241, 245) if (ix + iy) % 2 == 0 else (255, 255, 255)
            cv2.rectangle(frame, p1, p2, color, -1)
            cv2.rectangle(frame, p1, p2, (205, 211, 220), 1)

    overlay = frame.copy()
    cv2.rectangle(overlay, to_pixel((-1.0, 3.0)), to_pixel((1.0, 0.0)), (70, 70, 255), -1)
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
    cv2.rectangle(frame, to_pixel((-1.0, 3.0)), to_pixel((1.0, 0.0)), (20, 20, 220), 2)
    cv2.drawMarker(frame, to_pixel((0.0, 0.0)), (20, 20, 20), cv2.MARKER_TRIANGLE_UP, 24, 2)
    cv2.putText(frame, "robot/camera", (to_pixel((0.0, 0.0))[0] + 10, to_pixel((0.0, 0.0))[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)

    obs_limit = len(obs) if reveal_step is None else min(len(obs), max(0, reveal_step + 1))
    if target is None:
        future_limit = 0
    elif reveal_step is None:
        future_limit = len(target)
    else:
        future_limit = max(0, reveal_step - len(obs) + 1)
    pred_limit = len(pred) if reveal_step is None or reveal_step >= len(obs) - 1 else 0

    def draw_path(points: torch.Tensor, color: tuple[int, int, int], limit: int, radius: int = 5):
        if limit <= 0:
            return
        pixels = [to_pixel(point) for point in points[:limit]]
        for start, end in zip(pixels[:-1], pixels[1:]):
            cv2.line(frame, start, end, color, 2)
        for pixel in pixels:
            cv2.circle(frame, pixel, radius, color, -1)

    observed_points = obs[:obs_limit][mask[:obs_limit]]
    draw_path(observed_points, (220, 90, 20), len(observed_points), radius=5)
    if target is not None:
        draw_path(target, (55, 160, 65), future_limit, radius=5)
    draw_path(pred, (40, 40, 230), pred_limit, radius=4)

    for point in obs[:obs_limit][~mask[:obs_limit]]:
        x, y = to_pixel(point)
        cv2.line(frame, (x - 7, y - 7), (x + 7, y + 7), (105, 105, 105), 2)
        cv2.line(frame, (x - 7, y + 7), (x + 7, y - 7), (105, 105, 105), 2)

    action_color = {
        "stop": (20, 20, 220),
        "go": (35, 140, 35),
        "turn_left": (0, 140, 220),
        "turn_right": (0, 140, 220),
    }.get(action, (60, 60, 60))
    cv2.rectangle(frame, (20, 18), (360, 92), action_color, -1)
    cv2.putText(frame, f"Action: {action}", (34, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.86, (255, 255, 255), 2, cv2.LINE_AA)
    subtitle = f"True future action: {true_action}" if true_action is not None else result.get("subtitle", "")
    if subtitle:
        cv2.putText(frame, subtitle, (34, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    legend = "blue=observed  red=prediction"
    if target is not None:
        legend += "  green=true future"
    legend += "  gray=missing"
    cv2.putText(frame, legend, (20, size - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (35, 35, 35), 1, cv2.LINE_AA)
    return frame


def add_camera_inset(board_frame, camera_frame, detection=None) -> None:
    import cv2

    if camera_frame is None:
        return
    inset_width = max(150, board_frame.shape[1] // 5)
    inset_height = int(inset_width * camera_frame.shape[0] / camera_frame.shape[1])
    inset = cv2.resize(camera_frame, (inset_width, inset_height))
    if detection is not None:
        cx, cy, x, y, w, h = detection
        scale_x = inset_width / camera_frame.shape[1]
        scale_y = inset_height / camera_frame.shape[0]
        x0 = int(x * scale_x)
        y0 = int(y * scale_y)
        x1 = int((x + w) * scale_x)
        y1 = int((y + h) * scale_y)
        cc = (int(cx * scale_x), int(cy * scale_y))
        cv2.rectangle(inset, (x0, y0), (x1, y1), (255, 180, 0), 2)
        cv2.circle(inset, cc, 4, (255, 180, 0), -1)

    x_offset = board_frame.shape[1] - inset_width - 18
    y_offset = 18
    board_frame[y_offset : y_offset + inset_height, x_offset : x_offset + inset_width] = inset
    cv2.rectangle(
        board_frame,
        (x_offset, y_offset),
        (x_offset + inset_width, y_offset + inset_height),
        (40, 40, 40),
        2,
    )
    cv2.putText(
        board_frame,
        "camera",
        (x_offset + 8, y_offset + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def history_to_tensors(history: deque[tuple[float, float] | None], obs_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    obs_values = []
    mask_values = []
    last = (0.0, 0.0)
    for point in history:
        if point is None:
            obs_values.append(last)
            mask_values.append(0.0)
        else:
            last = point
            obs_values.append(point)
            mask_values.append(1.0)
    while len(obs_values) < obs_len:
        obs_values.insert(0, (0.0, 0.0))
        mask_values.insert(0, 0.0)
    obs = torch.tensor(obs_values[-obs_len:], dtype=torch.float32)
    mask = torch.tensor(mask_values[-obs_len:], dtype=torch.float32).view(obs_len, 1)
    return obs, mask


def point_distance(a: tuple[float, float] | None, b: tuple[float, float] | None) -> float:
    if a is None or b is None:
        return 0.0
    return math.hypot(a[0] - b[0], a[1] - b[1])


def motion_confidence(
    detected_point: tuple[float, float] | None,
    previous_point: tuple[float, float] | None,
    recent_moving: deque[bool],
    min_motion_distance: float,
    min_moving_samples: int,
) -> tuple[float, float, bool, int]:
    if detected_point is None:
        return 0.0, 0.0, False, sum(recent_moving)

    distance = point_distance(detected_point, previous_point)
    moving_now = previous_point is not None and distance >= min_motion_distance
    moving_count = sum(recent_moving) + int(moving_now)
    distance_score = min(distance / max(min_motion_distance, 1e-6), 1.0)
    sustained_score = min(moving_count / max(min_moving_samples, 1), 1.0)
    confidence = 0.25 + 0.35 * distance_score + 0.40 * sustained_score
    return min(confidence, 1.0), distance, moving_now, moving_count


@torch.no_grad()
def run_live_webcam(args: argparse.Namespace) -> None:
    import cv2

    model, pred_len, feature_mode = make_model(args)
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {args.camera_index}. "
            "Close other demo windows or try --camera-index 1."
        )

    subtractor = cv2.createBackgroundSubtractorMOG2(history=180, varThreshold=36, detectShadows=True)
    history: deque[tuple[float, float] | None] = deque(maxlen=args.obs_len)
    frame_index = 0
    predicted = torch.empty(0, 2)
    action = "warming_up"
    trajectory_action = "warming_up"
    obstacle_reason = "not_checked"
    pixels_per_meter = args.pixels_per_meter
    sample_interval = max(0.05, args.live_sample_interval_sec)
    prediction_refresh = max(0.05, args.prediction_refresh_sec)
    smoothing_alpha = min(max(args.live_smoothing_alpha, 0.0), 1.0)
    min_motion_distance = max(0.0, args.min_motion_distance)
    min_moving_samples = max(1, args.min_moving_samples)
    confidence_threshold = min(max(args.motion_confidence_threshold, 0.0), 1.0)
    next_sample_time = time.monotonic()
    next_prediction_time = 0.0
    smoothed_world_point: tuple[float, float] | None = None
    last_candidate_point: tuple[float, float] | None = None
    recent_moving: deque[bool] = deque(maxlen=max(args.obs_len, min_moving_samples))
    movement_confidence = 0.0
    sampled_distance = 0.0
    movement_status = "waiting"

    print("Mode: live webcam chessboard demo.")
    print("Press q to quit. Move one person/object in front of the webcam.")
    print("Use only one camera demo at a time; close other webcam windows first.")
    print(f"Loaded {args.model} checkpoint with pred_len={pred_len}; feature_mode={feature_mode}")
    print(f"Sampling one live point every {sample_interval:.1f}s; prediction refresh {prediction_refresh:.1f}s.")
    print(
        f"Motion gate: distance>={min_motion_distance:.2f}, "
        f"samples>={min_moving_samples}, confidence>={confidence_threshold:.2f}."
    )

    try:
        while True:
            ok, camera_frame = cap.read()
            if not ok:
                break
            if args.mirror:
                camera_frame = cv2.flip(camera_frame, 1)

            frame_index += 1
            height, width = camera_frame.shape[:2]
            if pixels_per_meter <= 0:
                pixels_per_meter = min(width, height) / 6.0

            obstacle_decision = None
            if args.enable_obstacle_zones:
                obstacle_decision = detect_obstacle_zones(
                    camera_frame,
                    roi_start=args.obstacle_roi_start,
                    occupancy_threshold=args.obstacle_threshold,
                    min_area=args.obstacle_min_area,
                )
                obstacle_reason = obstacle_decision.reason
            else:
                obstacle_reason = "disabled"

            now = time.monotonic()
            detection, _ = detect_motion_center(camera_frame, subtractor, args.min_contour_area)
            detected_world_point = None
            if detection is not None:
                cx, cy, *_ = detection
                world_point = pixel_to_world((cx, cy), width, height, pixels_per_meter)
                if smoothed_world_point is None:
                    smoothed_world_point = world_point
                else:
                    smoothed_world_point = (
                        smoothing_alpha * world_point[0] + (1.0 - smoothing_alpha) * smoothed_world_point[0],
                        smoothing_alpha * world_point[1] + (1.0 - smoothing_alpha) * smoothed_world_point[1],
                    )
                detected_world_point = smoothed_world_point

            appended_sample = False
            if frame_index > args.warmup_frames and now >= next_sample_time:
                (
                    movement_confidence,
                    sampled_distance,
                    moving_now,
                    moving_count,
                ) = motion_confidence(
                    detected_world_point,
                    last_candidate_point,
                    recent_moving,
                    min_motion_distance,
                    min_moving_samples,
                )
                if detected_world_point is not None:
                    last_candidate_point = detected_world_point
                recent_moving.append(moving_now)

                moving_confident = (
                    detected_world_point is not None
                    and moving_count >= min_moving_samples
                    and movement_confidence >= confidence_threshold
                )
                if moving_confident:
                    history.append(detected_world_point)
                    movement_status = "moving"
                else:
                    history.append(None)
                    movement_status = "not_confident" if detected_world_point is not None else "no_detection"
                    predicted = torch.empty(0, 2)
                    trajectory_action = "not_confident"
                appended_sample = True
                next_sample_time = now + sample_interval

            if (
                appended_sample
                and movement_status == "moving"
                and len(history) == args.obs_len
                and any(point is not None for point in history)
                and now >= next_prediction_time
            ):
                features, last_pos, mask, cv_obs = build_live_inputs(history, args.device, feature_mode)
                pred = model(features, last_pos, mask=mask, cv_obs=cv_obs)[0].detach().cpu()
                predicted = pred
                trajectory_decision = decide_navigation_action(pred)
                trajectory_action = trajectory_decision.action
                next_prediction_time = now + prediction_refresh
            elif len(history) < args.obs_len:
                predicted = torch.empty(0, 2)
                trajectory_action = "warming_up"

            if obstacle_decision is not None and obstacle_decision.action is not None:
                action = obstacle_decision.action
            elif movement_status != "moving":
                action = args.stationary_default_action
            else:
                action = trajectory_action

            obs, mask = history_to_tensors(history, args.obs_len)

            result = {
                "obs": obs,
                "mask": mask,
                "target": None,
                "pred": predicted if len(predicted) else torch.zeros(0, 2),
                "action": action,
                "true_action": None,
                "subtitle": (
                    f"buffer {len(history)}/{args.obs_len} | moving {movement_confidence:.2f} | "
                    f"dist {sampled_distance:.2f} | {movement_status} | model {trajectory_action}"
                ),
                "bounds": live_world_bounds(),
            }
            board_frame = draw_grid_frame(result, args.window_size)
            if args.show_camera_inset:
                add_camera_inset(board_frame, camera_frame, detection)

            cv2.imshow("Live webcam chessboard navigation demo", board_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    import cv2

    args = parse_args()
    if args.live_webcam and args.replay_dataset:
        raise ValueError("Choose either --live-webcam or --replay-dataset, not both.")

    replay_dataset = args.replay_dataset or args.headless
    if args.live_webcam or not replay_dataset:
        run_live_webcam(args)
        return

    print("Mode: dataset replay on chessboard grid.")
    result = predict_sample(args)
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if args.headless:
        frame = draw_grid_frame(result, args.window_size)
        cv2.imwrite(str(save_path), frame)
        print(f"Saved grid demo: {save_path}")
        print(
            {
                "sample_index": result["index"],
                "action": result["action"],
                "true_action": result["true_action"],
                "feature_mode": result["feature_mode"],
            }
        )
        return

    total_steps = args.obs_len + result["pred_len"]
    step = 0
    print("Press q to quit. The grid replays observed motion, prediction, and true future.")
    try:
        while True:
            frame = draw_grid_frame(result, args.window_size, reveal_step=step)
            cv2.imshow("Chessboard trajectory navigation demo", frame)
            key = cv2.waitKey(args.fps_delay_ms) & 0xFF
            if key == ord("q"):
                break
            step = (step + 1) % total_steps
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
