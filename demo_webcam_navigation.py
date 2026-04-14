from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import torch

from data.missingness import carry_forward, missing_gap_features, motion_features
from navigation import decide_navigation_action
from project_model import MissingnessAwareLSTM, MissingnessAwareTransformer


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "results" / "colab_zara2_missing_transformer_best.pt"


@dataclass(frozen=True)
class ObstacleZoneDecision:
    action: str | None
    reason: str
    blocked: dict[str, bool]
    occupancy: dict[str, float]
    roi: tuple[int, int, int, int]
    zones: dict[str, tuple[int, int, int, int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Raw webcam debug overlay for trajectory navigation. Use demo_grid_navigation.py for the class demo."
    )
    parser.add_argument("--camera-index", type=int, default=1)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--model", default="missing_transformer", choices=("missing_transformer", "missing_lstm"))
    parser.add_argument("--obs-len", type=int, default=8)
    parser.add_argument("--pred-len", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--feature-mode", default="auto", choices=("auto", "basic", "motion"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pixels-per-meter", type=float, default=0.0)
    parser.add_argument("--min-contour-area", type=float, default=900.0)
    parser.add_argument("--warmup-frames", type=int, default=20)
    parser.add_argument("--mirror", action="store_true", help="Mirror the webcam view for a selfie-style display.")
    parser.add_argument("--enable-obstacle-zones", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--obstacle-roi-start", type=float, default=0.55)
    parser.add_argument("--obstacle-threshold", type=float, default=0.025)
    parser.add_argument("--obstacle-min-area", type=float, default=1200.0)
    parser.add_argument("--disable-trajectory", action="store_true", help="Run only RGB obstacle-zone rules.")
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
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Train a model first or pass --checkpoint."
        )

    state, checkpoint_args = checkpoint_state(checkpoint_path)
    pred_len = int(checkpoint_args.get("pred_len", args.pred_len))
    if args.model == "missing_transformer":
        input_dim = int(state.get("input_proj.weight", torch.empty(args.hidden_dim, 4)).shape[1])
        feature_mode = args.feature_mode
        if feature_mode == "auto":
            feature_mode = "motion" if input_dim == 8 else "basic"
        hidden_dim = int(state.get("input_proj.weight", torch.empty(args.hidden_dim, 4)).shape[0])
        final_weight = state.get("residual_head.4.weight", state.get("head.4.weight"))
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
        hidden_dim = int(hidden_weight.shape[1]) if hidden_weight is not None else args.hidden_dim
        input_weight = state.get("encoder.weight_ih_l0")
        input_dim = int(input_weight.shape[1]) if input_weight is not None else 4
        feature_mode = args.feature_mode
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


def pixel_to_world(point: tuple[int, int], width: int, height: int, pixels_per_meter: float) -> tuple[float, float]:
    x_pixel, y_pixel = point
    x_world = (x_pixel - width / 2.0) / pixels_per_meter
    y_world = (height - y_pixel) / pixels_per_meter
    return x_world, y_world


def world_to_pixel(point: torch.Tensor, width: int, height: int, pixels_per_meter: float) -> tuple[int, int]:
    x_world = float(point[0])
    y_world = float(point[1])
    return int(round(x_world * pixels_per_meter + width / 2.0)), int(round(height - y_world * pixels_per_meter))


def detect_motion_center(frame, subtractor, min_contour_area: float):
    import cv2

    mask = subtractor.apply(frame)
    mask = cv2.medianBlur(mask, 5)
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < min_contour_area:
        return None, mask

    x, y, w, h = cv2.boundingRect(contour)
    return (x + w // 2, y + h // 2, x, y, w, h), mask


def detect_obstacle_zones(
    frame,
    roi_start: float = 0.55,
    occupancy_threshold: float = 0.025,
    min_area: float = 1200.0,
) -> ObstacleZoneDecision:
    import cv2

    height, width = frame.shape[:2]
    roi_start = min(max(roi_start, 0.0), 0.95)
    y0 = int(height * roi_start)
    roi = frame[y0:height, :]
    roi_height = max(1, height - y0)
    zone_width = width // 3
    zone_bounds = {
        "left": (0, y0, zone_width, roi_height),
        "center": (zone_width, y0, zone_width, roi_height),
        "right": (zone_width * 2, y0, width - zone_width * 2, roi_height),
    }

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 45, 135)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    obstacle_mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    obstacle_mask = cv2.dilate(obstacle_mask, kernel, iterations=1)

    blocked: dict[str, bool] = {}
    occupancy: dict[str, float] = {}
    for name, (x, _, zone_w, _) in zone_bounds.items():
        zone_mask = obstacle_mask[:, x : x + zone_w]
        contours, _ = cv2.findContours(zone_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        large_area = sum(cv2.contourArea(contour) for contour in contours if cv2.contourArea(contour) >= min_area)
        ratio = float(cv2.countNonZero(zone_mask)) / float(max(1, zone_mask.size))
        occupancy[name] = ratio
        blocked[name] = ratio >= occupancy_threshold or large_area >= min_area

    action = None
    reason = "clear"
    if blocked["center"]:
        action = "stop"
        reason = "center_blocked"
    elif blocked["left"]:
        action = "turn_right"
        reason = "left_blocked"
    elif blocked["right"]:
        action = "turn_left"
        reason = "right_blocked"

    return ObstacleZoneDecision(
        action=action,
        reason=reason,
        blocked=blocked,
        occupancy=occupancy,
        roi=(0, y0, width, roi_height),
        zones=zone_bounds,
    )


def build_live_inputs(history: deque[tuple[float, float] | None], device: str, feature_mode: str = "basic"):
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

    obs = torch.tensor(obs_values, dtype=torch.float32, device=device).unsqueeze(0)
    mask = torch.tensor(mask_values, dtype=torch.float32, device=device).view(1, -1, 1)
    filled = carry_forward(obs, mask)
    if feature_mode == "motion":
        features = motion_features(filled, mask)
    else:
        features = torch.cat([filled, mask, missing_gap_features(mask)], dim=-1)
    last_pos = filled[:, -1, :]
    return features, last_pos, mask, filled


def draw_polyline(frame, points, color, thickness: int = 2) -> None:
    import cv2

    if len(points) < 2:
        return
    for start, end in zip(points[:-1], points[1:]):
        cv2.line(frame, start, end, color, thickness)
    for point in points:
        cv2.circle(frame, point, 4, color, -1)


def draw_safety_zone(frame, width: int, height: int, pixels_per_meter: float) -> None:
    import cv2

    left = world_to_pixel(torch.tensor([-1.0, 3.0]), width, height, pixels_per_meter)
    right = world_to_pixel(torch.tensor([1.0, 0.0]), width, height, pixels_per_meter)
    overlay = frame.copy()
    cv2.rectangle(overlay, left, right, (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.16, frame, 0.84, 0, frame)
    cv2.rectangle(frame, left, right, (0, 0, 255), 2)
    robot = world_to_pixel(torch.tensor([0.0, 0.0]), width, height, pixels_per_meter)
    cv2.drawMarker(frame, robot, (0, 0, 0), markerType=cv2.MARKER_TRIANGLE_UP, markerSize=18, thickness=2)


def draw_obstacle_zones(frame, decision: ObstacleZoneDecision) -> None:
    import cv2

    overlay = frame.copy()
    colors = {
        "left": (0, 255, 255),
        "center": (0, 0, 255),
        "right": (0, 255, 255),
    }
    clear_color = (0, 160, 0)
    for name, (x, y, w, h) in decision.zones.items():
        color = colors[name] if decision.blocked[name] else clear_color
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            frame,
            f"{name}: {decision.occupancy[name]:.2f}",
            (x + 8, y + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)


def main() -> None:
    import cv2

    args = parse_args()
    model = None
    pred_len = args.pred_len
    feature_mode = args.feature_mode
    if not args.disable_trajectory:
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
    final_action = "warming_up"
    trajectory_action = "warming_up"
    trajectory_reason = "waiting for trajectory buffer"
    obstacle_reason = "not_checked"
    predicted = None
    pixels_per_meter = args.pixels_per_meter

    print("Mode: raw webcam debug overlay.")
    print("For the class demo, run demo_grid_navigation.py --live-webcam --mirror instead.")
    print("Use only one camera demo at a time; close other webcam windows first.")
    print("Press q to quit. Move one person/object in front of the webcam after warmup.")
    if model is not None:
        print(f"Loaded {args.model} checkpoint with pred_len={pred_len}: {args.checkpoint}")
    else:
        print("Trajectory model disabled; running RGB obstacle-zone rules only.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)

            frame_index += 1
            height, width = frame.shape[:2]
            if pixels_per_meter <= 0:
                pixels_per_meter = min(width, height) / 6.0

            obstacle_decision = None
            if args.enable_obstacle_zones:
                obstacle_decision = detect_obstacle_zones(
                    frame,
                    roi_start=args.obstacle_roi_start,
                    occupancy_threshold=args.obstacle_threshold,
                    min_area=args.obstacle_min_area,
                )
                draw_obstacle_zones(frame, obstacle_decision)
                obstacle_reason = obstacle_decision.reason
            else:
                obstacle_reason = "disabled"

            detection, _ = detect_motion_center(frame, subtractor, args.min_contour_area)
            if detection is not None:
                cx, cy, x, y, w, h = detection
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 180, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (255, 180, 0), -1)
                world_point = pixel_to_world((cx, cy), width, height, pixels_per_meter)
                if frame_index > args.warmup_frames:
                    history.append(world_point)
            elif history and frame_index > args.warmup_frames:
                history.append(None)

            draw_safety_zone(frame, width, height, pixels_per_meter)

            observed_points = [
                world_to_pixel(torch.tensor(point), width, height, pixels_per_meter)
                for point in history
                if point is not None
            ]
            draw_polyline(frame, observed_points, (255, 80, 20), thickness=2)

            if model is not None and len(history) == args.obs_len and any(point is not None for point in history):
                features, last_pos, mask, cv_obs = build_live_inputs(history, args.device, feature_mode)
                with torch.no_grad():
                    pred = model(features, last_pos, mask=mask, cv_obs=cv_obs)[0].detach().cpu()
                predicted = pred
                decision = decide_navigation_action(pred)
                trajectory_action = decision.action
                trajectory_reason = decision.reason

            if obstacle_decision is not None and obstacle_decision.action is not None:
                final_action = obstacle_decision.action
            elif model is not None:
                final_action = trajectory_action
            else:
                final_action = "go"

            if predicted is not None:
                pred_pixels = [world_to_pixel(point, width, height, pixels_per_meter) for point in predicted]
                draw_polyline(frame, pred_pixels, (0, 0, 255), thickness=2)

            cv2.putText(
                frame,
                f"Debug Action: {final_action}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Obstacle Rule: {obstacle_reason}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Trajectory Rule: {trajectory_action} ({trajectory_reason})",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Trajectory buffer: {len(history)}/{args.obs_len}",
                (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Raw webcam navigation debug overlay", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
