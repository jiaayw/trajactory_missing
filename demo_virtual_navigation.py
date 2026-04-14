from __future__ import annotations

import argparse
import math
import random
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import torch

from baseline_model import ConstantVelocityPredictor
from demo_grid_navigation import DEFAULT_CHECKPOINT, make_model
from demo_webcam_navigation import build_live_inputs
from navigation import decide_navigation_action


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SAVE_PATH = PROJECT_ROOT / "results" / "plots" / "virtual_demo.png"
WORLD_BOUNDS = (-6.0, 6.0, -2.0, 8.0)
SCENARIOS = (
    "crossing_left",
    "crossing_right",
    "approaching_robot",
    "passing_safe",
    "stop_case",
    "random",
    "diagonal_crossing",
    "sudden_stop",
    "curved_crossing",
    "two_stage_crossing",
)
KEY_SCENARIOS = {
    ord("1"): "crossing_left",
    ord("2"): "crossing_right",
    ord("3"): "approaching_robot",
    ord("4"): "passing_safe",
    ord("5"): "stop_case",
    ord("6"): "diagonal_crossing",
    ord("7"): "curved_crossing",
    ord("8"): "sudden_stop",
    ord("9"): "random",
}
GOAL_PRESETS = (
    (-1.2, 7.2),
    (0.0, 7.2),
    (1.2, 7.2),
    (-5.0, 3.0),
    (5.0, 3.0),
    (-1.2, 5.4),
    (1.2, 5.4),
)
SCENARIO_GOALS = {
    "crossing_left": (1.2, 6.8),
    "crossing_right": (-1.2, 6.8),
    "approaching_robot": (0.0, 7.2),
    "passing_safe": (1.2, 6.6),
    "stop_case": (-1.2, 6.6),
    "random": (5.0, 3.0),
    "diagonal_crossing": (-5.0, 3.0),
    "sudden_stop": (1.2, 6.4),
    "curved_crossing": (-1.2, 6.2),
    "two_stage_crossing": (5.0, 3.0),
}
LANE_CENTERS = (-1.2, 0.0, 1.2)
LANE_CENTER_X = 0.0
LANE_HEADING = math.pi / 2.0
LEFT_HEADING = math.pi
RIGHT_HEADING = 0.0
SIDE_GOAL_X = 5.0


@dataclass
class DemoState:
    robot: tuple[float, float, float]
    goal: tuple[float, float]
    driving_goal: tuple[float, float]
    route_phase: str
    target_lane: int
    pedestrian_history: deque[tuple[float, float]]
    robot_history: deque[tuple[float, float]]
    background_peds: list[dict[str, float]]
    active_action: str = "warming_up"
    pending_action: str = "warming_up"
    pending_count: int = 0
    stop_cooldown: int = 0
    step: int = 0
    paused: bool = False
    show_true_future: bool = True
    show_background: bool = True
    goal_reached: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Virtual moving-robot pedestrian navigation demo.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--model", default="missing_transformer", choices=("constant_velocity", "missing_transformer", "missing_lstm"))
    parser.add_argument("--feature-mode", default="auto", choices=("auto", "basic", "motion"))
    parser.add_argument("--scenario", default="crossing_left", choices=SCENARIOS)
    parser.add_argument("--num-background-pedestrians", type=int, default=4)
    parser.add_argument("--obs-len", type=int, default=8)
    parser.add_argument("--pred-len", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--robot-speed", type=float, default=0.12)
    parser.add_argument("--robot-turn-rate", type=float, default=8.0)
    parser.add_argument("--window-size", type=int, default=820)
    parser.add_argument("--fps-delay-ms", type=int, default=180)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--save-path", default=str(DEFAULT_SAVE_PATH))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--show-true-future", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-status-panel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-robot-trail", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--action-smoothing-steps", type=int, default=2)
    parser.add_argument("--stop-cooldown-steps", type=int, default=4)
    parser.add_argument("--robot-radius", type=float, default=0.25)
    parser.add_argument("--risk-distance", type=float, default=0.65)
    parser.add_argument("--turn-clearance-distance", type=float, default=1.10)
    parser.add_argument("--stop-release-distance", type=float, default=1.25)
    parser.add_argument("--goal-x", type=float, default=None)
    parser.add_argument("--goal-y", type=float, default=None)
    parser.add_argument("--goal-radius", type=float, default=0.45)
    parser.add_argument("--auto-reset-on-goal", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--goal-turn-threshold-deg", type=float, default=8.0)
    parser.add_argument("--goal-turn-move-scale", type=float, default=0.35)
    parser.add_argument("--lane-change-speed", type=float, default=0.06)
    parser.add_argument("--intersection-y", type=float, default=3.0)
    parser.add_argument("--turn-complete-threshold-deg", type=float, default=8.0)
    parser.add_argument("--route-debug", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    args.custom_goal = any(arg == "--goal-x" or arg.startswith("--goal-x=") for arg in sys.argv) or any(
        arg == "--goal-y" or arg.startswith("--goal-y=") for arg in sys.argv
    )
    if args.custom_goal:
        args.goal_x = 0.0 if args.goal_x is None else args.goal_x
        args.goal_y = 7.2 if args.goal_y is None else args.goal_y
        args.goal_source = "manual"
    else:
        set_scenario_goal(args)
    return args


def set_scenario_goal(args: argparse.Namespace, source: str = "scenario") -> None:
    args.goal_x, args.goal_y = SCENARIO_GOALS.get(args.scenario, (0.0, 7.2))
    args.goal_source = source


def print_goal(args: argparse.Namespace) -> None:
    driving_goal = snap_goal_to_road((args.goal_x, args.goal_y), args)
    print(
        f"Road goal: requested=({args.goal_x:.2f}, {args.goal_y:.2f}); "
        f"driving=({driving_goal[0]:.2f}, {driving_goal[1]:.2f}); source={args.goal_source}"
    )


def nearest_lane_index(x: float) -> int:
    return min(range(len(LANE_CENTERS)), key=lambda idx: abs(LANE_CENTERS[idx] - x))


def is_side_goal(goal: tuple[float, float], args: argparse.Namespace) -> bool:
    return abs(goal[0]) > max(abs(x) for x in LANE_CENTERS) + 0.8 and abs(goal[1] - args.intersection_y) < 0.35


def snap_goal_to_road(goal: tuple[float, float], args: argparse.Namespace) -> tuple[float, float]:
    x, y = goal
    if abs(x) > 2.4 or abs(y - args.intersection_y) < 0.8:
        return (SIDE_GOAL_X if x >= 0 else -SIDE_GOAL_X, args.intersection_y)
    return (LANE_CENTERS[nearest_lane_index(x)], min(max(y, args.intersection_y + 1.0), WORLD_BOUNDS[3] - 0.5))


def scenario_position(name: str, step: int) -> tuple[float, float]:
    t = float(step)
    if name == "crossing_left":
        return 4.8 - 0.18 * t, 2.5 + 0.08 * math.sin(t * 0.12)
    if name == "crossing_right":
        return -4.8 + 0.18 * t, 2.5 + 0.08 * math.sin(t * 0.12)
    if name == "approaching_robot":
        return 0.25 + 0.1 * math.sin(t * 0.08), 6.7 - 0.17 * t
    if name == "passing_safe":
        return 3.4 - 0.04 * t, 5.8 - 0.08 * t
    if name == "stop_case":
        return 0.2 * math.sin(t * 0.15), 2.2 + 0.2 * math.cos(t * 0.12)
    if name == "diagonal_crossing":
        return 4.8 - 0.14 * t, 6.6 - 0.13 * t
    if name == "sudden_stop":
        moving_t = min(t, 24.0)
        return -4.2 + 0.14 * moving_t, 3.0
    if name == "curved_crossing":
        return 4.0 * math.cos(0.045 * t), 3.1 + 2.1 * math.sin(0.045 * t)
    if name == "two_stage_crossing":
        if t < 22:
            return 4.5 - 0.16 * t, 2.25
        return 1.0 - 0.05 * (t - 22), 2.25 + 0.13 * (t - 22)
    return 3.6 * math.sin(t * 0.06) + 0.45 * math.sin(t * 0.21), 3.3 + 2.0 * math.cos(t * 0.043)


def make_background_pedestrians(count: int, seed: int) -> list[dict[str, float]]:
    rng = random.Random(seed)
    return [
        {
            "x": rng.uniform(-5.5, 5.5),
            "y": rng.uniform(-0.5, 7.5),
            "vx": rng.uniform(-0.045, 0.045),
            "vy": rng.uniform(-0.025, 0.025),
            "phase": rng.uniform(0.0, math.tau),
        }
        for _ in range(count)
    ]


def reset_state(args: argparse.Namespace) -> DemoState:
    driving_goal = snap_goal_to_road((args.goal_x, args.goal_y), args)
    return DemoState(
        robot=(0.0, -1.0, math.pi / 2.0),
        goal=(args.goal_x, args.goal_y),
        driving_goal=driving_goal,
        route_phase="start",
        target_lane=nearest_lane_index(driving_goal[0]) if not is_side_goal(driving_goal, args) else 1,
        pedestrian_history=deque(maxlen=args.obs_len),
        robot_history=deque([(0.0, -1.0)], maxlen=180),
        background_peds=make_background_pedestrians(args.num_background_pedestrians, args.seed),
        show_true_future=args.show_true_future,
        show_background=args.num_background_pedestrians > 0,
    )


def update_background_pedestrians(peds: list[dict[str, float]], step: int) -> None:
    min_x, max_x, min_y, max_y = WORLD_BOUNDS
    for ped in peds:
        ped["x"] += ped["vx"] + 0.012 * math.sin(step * 0.05 + ped["phase"])
        ped["y"] += ped["vy"] + 0.008 * math.cos(step * 0.04 + ped["phase"])
        if ped["x"] < min_x:
            ped["x"] = max_x
        elif ped["x"] > max_x:
            ped["x"] = min_x
        if ped["y"] < min_y:
            ped["y"] = max_y
        elif ped["y"] > max_y:
            ped["y"] = min_y


def forward_vector(heading: float) -> tuple[float, float]:
    return math.cos(heading), math.sin(heading)


def right_vector(heading: float) -> tuple[float, float]:
    fx, fy = forward_vector(heading)
    return fy, -fx


def local_to_world(point: tuple[float, float], robot: tuple[float, float, float]) -> tuple[float, float]:
    robot_x, robot_y, heading = robot
    right_x, right_y = right_vector(heading)
    forward_x, forward_y = forward_vector(heading)
    local_x, local_y = point
    return (
        robot_x + right_x * local_x + forward_x * local_y,
        robot_y + right_y * local_x + forward_y * local_y,
    )


def world_to_local(point: tuple[float, float], robot: tuple[float, float, float]) -> tuple[float, float]:
    robot_x, robot_y, heading = robot
    dx = point[0] - robot_x
    dy = point[1] - robot_y
    right_x, right_y = right_vector(heading)
    forward_x, forward_y = forward_vector(heading)
    return dx * right_x + dy * right_y, dx * forward_x + dy * forward_y


def clamp_robot(robot: tuple[float, float, float]) -> tuple[float, float, float]:
    min_x, max_x, min_y, max_y = WORLD_BOUNDS
    x, y, heading = robot
    return min(max(x, min_x + 0.4), max_x - 0.4), min(max(y, min_y + 0.4), max_y - 0.4), heading


def move_toward(value: float, target: float, step: float) -> float:
    if abs(target - value) <= step:
        return target
    return value + step if target > value else value - step


def update_robot(
    robot: tuple[float, float, float],
    action: str,
    state: DemoState,
    args: argparse.Namespace,
) -> tuple[float, float, float]:
    x, y, heading = robot
    speed = args.robot_speed
    turn_step = math.radians(args.robot_turn_rate)
    target_x = LANE_CENTERS[state.target_lane]

    if action == "stop":
        return clamp_robot((x, y, heading))

    if action in ("change_left", "change_right"):
        x = move_toward(x, target_x, args.lane_change_speed)
        y += speed * 0.55
        heading = move_toward(heading, LANE_HEADING, turn_step * 0.35)
        return clamp_robot((x, y, heading))

    if action == "turn_left":
        heading = move_toward(heading, LEFT_HEADING, turn_step)
        fx, fy = forward_vector(heading)
        return clamp_robot((x + fx * speed * args.goal_turn_move_scale, y + fy * speed * args.goal_turn_move_scale, heading))

    if action == "turn_right":
        heading = move_toward(heading, RIGHT_HEADING, turn_step)
        fx, fy = forward_vector(heading)
        return clamp_robot((x + fx * speed * args.goal_turn_move_scale, y + fy * speed * args.goal_turn_move_scale, heading))

    if action == "go":
        if state.route_phase in ("follow_side_left", "follow_side_right"):
            target_heading = LEFT_HEADING if state.driving_goal[0] < 0 else RIGHT_HEADING
            heading = move_toward(heading, target_heading, turn_step * 0.35)
            fx, fy = forward_vector(heading)
            return clamp_robot((x + fx * speed, y + fy * speed, heading))
        x = move_toward(x, target_x, args.lane_change_speed * 0.5)
        heading = move_toward(heading, LANE_HEADING, turn_step * 0.35)
        return clamp_robot((x, y + speed, heading))

    return clamp_robot((x, y, heading))


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % math.tau - math.pi


def distance_to_goal(robot: tuple[float, float, float], goal: tuple[float, float]) -> float:
    return math.hypot(goal[0] - robot[0], goal[1] - robot[1])


def route_action(
    state: DemoState,
    args: argparse.Namespace,
) -> tuple[str, str, float, bool]:
    robot = state.robot
    goal = state.driving_goal
    goal_distance = distance_to_goal(robot, goal)
    if goal_distance <= args.goal_radius:
        state.route_phase = "goal_reached"
        return "stop", "goal_reached", goal_distance, True

    side_goal = is_side_goal(goal, args)
    state.target_lane = 1 if side_goal else nearest_lane_index(goal[0])
    target_x = LANE_CENTERS[state.target_lane]
    if abs(robot[0] - target_x) > 0.08 and robot[1] < args.intersection_y - 0.15:
        state.route_phase = "change_left" if target_x < robot[0] else "change_right"
        return state.route_phase, state.route_phase, goal_distance, False

    if side_goal:
        if robot[1] < args.intersection_y - 0.08:
            state.route_phase = "approach_intersection"
            return "go", "approach_intersection", goal_distance, False
        turn_threshold = math.radians(args.turn_complete_threshold_deg)
        if goal[0] < 0 and abs(wrap_angle(LEFT_HEADING - robot[2])) > turn_threshold:
            state.route_phase = "turn_left"
            return "turn_left", "intersection_turn_left", goal_distance, False
        if goal[0] > 0 and abs(wrap_angle(RIGHT_HEADING - robot[2])) > turn_threshold:
            state.route_phase = "turn_right"
            return "turn_right", "intersection_turn_right", goal_distance, False
        state.route_phase = "follow_side_left" if goal[0] < 0 else "follow_side_right"
        return "go", state.route_phase, goal_distance, False

    state.route_phase = "drive_forward"
    if robot[1] >= goal[1] - args.goal_radius:
        state.route_phase = "goal_reached"
        return "stop", "goal_reached", abs(goal[1] - robot[1]), True
    return "go", "drive_to_top_goal", goal_distance, False


def apply_goal_navigation(
    refined_action: str,
    risk: str,
    state: DemoState,
    args: argparse.Namespace,
) -> tuple[str, str, float, bool]:
    goal_action, goal_reason, goal_distance, goal_reached = route_action(state, args)
    state.goal_reached = goal_reached
    if goal_reached:
        state.stop_cooldown = 0
        return "stop", goal_reason, goal_distance, True

    if refined_action == "stop" or risk in ("collision", "high"):
        return refined_action, "avoidance_priority", goal_distance, False
    if refined_action in ("turn_left", "turn_right") and risk == "medium":
        current_lane = nearest_lane_index(state.robot[0])
        if refined_action == "turn_left":
            state.target_lane = max(0, current_lane - 1)
            return "change_left", "avoidance_medium_lane_change", goal_distance, False
        state.target_lane = min(len(LANE_CENTERS) - 1, current_lane + 1)
        return "change_right", "avoidance_medium_lane_change", goal_distance, False
    return goal_action, goal_reason, goal_distance, False


def smooth_action(
    state: DemoState,
    raw_action: str,
    args: argparse.Namespace,
    immediate: bool = False,
    cooldown_on_stop: bool = True,
) -> str:
    if raw_action == "warming_up":
        return "go"
    if raw_action != "stop" and state.stop_cooldown > 0:
        state.stop_cooldown = 0
    if immediate:
        state.pending_action = raw_action
        state.pending_count = max(1, args.action_smoothing_steps)
        state.active_action = raw_action
        return state.active_action
    if state.stop_cooldown > 0:
        state.stop_cooldown -= 1
        state.active_action = "stop"
        return state.active_action
    if raw_action == state.pending_action:
        state.pending_count += 1
    else:
        state.pending_action = raw_action
        state.pending_count = 1
    if state.pending_count >= max(1, args.action_smoothing_steps):
        state.active_action = raw_action
        if raw_action == "stop" and cooldown_on_stop:
            state.stop_cooldown = max(0, args.stop_cooldown_steps)
    return state.active_action


def world_to_pixel(point: tuple[float, float], scene_size: int, margin: int = 70) -> tuple[int, int]:
    min_x, max_x, min_y, max_y = WORLD_BOUNDS
    scale = (scene_size - 2 * margin) / max(max_x - min_x, max_y - min_y)
    x = int(round(margin + (point[0] - min_x) * scale))
    y = int(round(scene_size - margin - (point[1] - min_y) * scale))
    return x, y


def draw_dashed_path(frame, points: list[tuple[float, float]], scene_size: int, color: tuple[int, int, int]) -> None:
    import cv2

    pixels = [world_to_pixel(point, scene_size) for point in points]
    for idx, (start, end) in enumerate(zip(pixels[:-1], pixels[1:])):
        if idx % 2 == 0:
            cv2.line(frame, start, end, color, 1)
    for idx, pixel in enumerate(pixels):
        if idx % 2 == 0:
            cv2.circle(frame, pixel, 3, color, -1)


def draw_path(
    frame,
    points: list[tuple[float, float]],
    scene_size: int,
    color: tuple[int, int, int],
    radius: int,
    thickness: int = 2,
) -> None:
    import cv2

    if not points:
        return
    pixels = [world_to_pixel(point, scene_size) for point in points]
    for start, end in zip(pixels[:-1], pixels[1:]):
        cv2.line(frame, start, end, color, thickness)
    for pixel in pixels:
        cv2.circle(frame, pixel, radius, color, -1)


def draw_fading_prediction(frame, points: list[tuple[float, float]], scene_size: int) -> None:
    import cv2

    if not points:
        return
    pixels = [world_to_pixel(point, scene_size) for point in points]
    for start, end in zip(pixels[:-1], pixels[1:]):
        cv2.line(frame, start, end, (55, 55, 210), 2)
    total = max(1, len(pixels) - 1)
    for idx, pixel in enumerate(pixels):
        strength = 1.0 - 0.55 * idx / total
        color = (int(80 * strength), int(70 * strength), 235)
        cv2.circle(frame, pixel, max(3, 7 - idx // 3), color, -1)


def risk_metrics(pred_local: torch.Tensor | None, args: argparse.Namespace) -> tuple[float | None, str]:
    if pred_local is None or len(pred_local) == 0:
        return None, "warming_up"
    distances = torch.linalg.norm(pred_local, dim=-1)
    min_distance = float(distances.min().item())
    if min_distance <= args.robot_radius:
        return min_distance, "collision"
    if min_distance <= args.risk_distance:
        return min_distance, "high"
    if min_distance <= args.risk_distance * 1.6:
        return min_distance, "medium"
    return min_distance, "low"


def path_enters_front_zone(
    pred_local: torch.Tensor | None,
    safety_half_width: float = 1.0,
    safety_depth: float = 3.0,
) -> bool:
    if pred_local is None or len(pred_local) == 0:
        return False
    x = pred_local[:, 0]
    y = pred_local[:, 1]
    return bool(((x.abs() <= safety_half_width) & (y >= 0) & (y <= safety_depth)).any())


def avoidance_action_from_path(pred_local: torch.Tensor | None) -> str:
    if pred_local is None or len(pred_local) == 0:
        return "stop"
    x = pred_local[:, 0]
    y = pred_local[:, 1]
    close_ahead = (y >= 0) & (y <= 4.0)
    if not bool(close_ahead.any()):
        return "go"
    mean_x = float(x[close_ahead].mean().item())
    if mean_x > 0.2:
        return "turn_left"
    if mean_x < -0.2:
        return "turn_right"
    return "stop"


def refine_robot_action(
    raw_action: str,
    pred_local: torch.Tensor | None,
    min_distance: float | None,
    risk: str,
    state: DemoState,
    args: argparse.Namespace,
) -> tuple[str, str]:
    if raw_action == "warming_up":
        return "go", "warmup_default_go"

    in_front_zone = path_enters_front_zone(pred_local)
    safe_distance = min_distance is None or min_distance > args.turn_clearance_distance
    stop_release_safe = min_distance is not None and min_distance > args.stop_release_distance and risk == "low"

    if raw_action in ("turn_left", "turn_right"):
        if risk == "low" or safe_distance:
            return "go", "turn_recovered_clear_path"
        return raw_action, "turn_kept_for_avoidance"

    if raw_action == "stop":
        if in_front_zone or risk in ("high", "collision"):
            return "stop", "stop_kept_for_front_risk"
        if stop_release_safe:
            state.stop_cooldown = 0
            return "go", "stop_released_clear_path"
        return "go", "stop_soft_recovery"

    if raw_action == "go":
        if risk in ("collision", "high"):
            return "stop", "go_overridden_high_risk"
        if risk == "medium" and in_front_zone:
            return avoidance_action_from_path(pred_local), "go_refined_medium_front_risk"
        return "go", "go_path_clear"

    return raw_action, "controller_passthrough"


def draw_world_markings(frame, scene_size: int) -> None:
    import cv2

    # Multi-lane road and intersection.
    cv2.rectangle(frame, world_to_pixel((-1.8, 7.8), scene_size), world_to_pixel((1.8, -1.8), scene_size), (235, 238, 242), -1)
    cv2.rectangle(frame, world_to_pixel((-5.8, 3.7), scene_size), world_to_pixel((5.8, 2.3), scene_size), (235, 238, 242), -1)
    for road_x in (-1.8, 1.8):
        cv2.line(frame, world_to_pixel((road_x, -1.8), scene_size), world_to_pixel((road_x, 7.8), scene_size), (105, 115, 125), 2)
    for road_y in (2.3, 3.7):
        cv2.line(frame, world_to_pixel((-5.8, road_y), scene_size), world_to_pixel((5.8, road_y), scene_size), (105, 115, 125), 2)
    for lane_x in (-0.6, 0.6):
        for y in [value * 0.8 - 1.6 for value in range(12)]:
            cv2.line(frame, world_to_pixel((lane_x, y), scene_size), world_to_pixel((lane_x, y + 0.35), scene_size), (165, 175, 185), 1)
    for x in [value * 0.9 - 5.4 for value in range(13)]:
        cv2.line(frame, world_to_pixel((x, 3.0), scene_size), world_to_pixel((x + 0.4, 3.0), scene_size), (165, 175, 185), 1)
    cv2.rectangle(frame, world_to_pixel((-1.8, 3.7), scene_size), world_to_pixel((1.8, 2.3), scene_size), (205, 210, 218), 2)
    cv2.putText(frame, "3-lane road", world_to_pixel((2.05, 7.35), scene_size), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 90, 100), 1, cv2.LINE_AA)
    cv2.putText(frame, "intersection", world_to_pixel((2.05, 3.55), scene_size), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 90, 100), 1, cv2.LINE_AA)

    # Crosswalk.
    for x in [-4.8, -3.8, -2.8, -1.8, -0.8, 0.2, 1.2, 2.2, 3.2, 4.2]:
        p1 = world_to_pixel((x, 2.1), scene_size)
        p2 = world_to_pixel((x + 0.55, 3.0), scene_size)
        cv2.rectangle(frame, p1, p2, (225, 225, 225), -1)
        cv2.rectangle(frame, p1, p2, (185, 185, 185), 1)
    cv2.putText(frame, "crosswalk", world_to_pixel((-5.3, 3.25), scene_size), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.putText(frame, "robot start", world_to_pixel((0.35, -1.15), scene_size), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1, cv2.LINE_AA)


def planned_route_points(state: DemoState, args: argparse.Namespace) -> list[tuple[float, float]]:
    robot_xy = (state.robot[0], state.robot[1])
    goal = state.driving_goal
    if is_side_goal(goal, args):
        return [robot_xy, (LANE_CENTERS[1], state.robot[1]), (LANE_CENTERS[1], args.intersection_y), goal]
    lane_x = LANE_CENTERS[nearest_lane_index(goal[0])]
    return [robot_xy, (lane_x, state.robot[1]), (lane_x, goal[1])]


def draw_goal(frame, state: DemoState, scene_size: int, args: argparse.Namespace) -> None:
    import cv2

    route_points = planned_route_points(state, args)
    draw_dashed_path(frame, route_points, scene_size, (150, 165, 140))
    requested_px = world_to_pixel(state.goal, scene_size)
    driving_px = world_to_pixel(state.driving_goal, scene_size)
    radius_px = max(8, int(round(args.goal_radius * 14)))
    fill = (80, 210, 185) if not state.goal_reached else (70, 190, 70)
    if requested_px != driving_px:
        cv2.circle(frame, requested_px, 7, (160, 190, 140), 1)
    cv2.circle(frame, driving_px, radius_px + 4, (245, 250, 240), -1)
    cv2.circle(frame, driving_px, radius_px + 4, (95, 145, 115), 1)
    cv2.circle(frame, driving_px, radius_px, fill, -1)
    flag_top = (driving_px[0] + 8, driving_px[1] - 28)
    cv2.line(frame, driving_px, flag_top, (40, 90, 65), 2)
    cv2.rectangle(frame, flag_top, (flag_top[0] + 22, flag_top[1] + 12), (70, 170, 100), -1)
    cv2.putText(frame, "road goal", (driving_px[0] + 18, driving_px[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (45, 100, 70), 1, cv2.LINE_AA)


def draw_status_panel(
    frame,
    scene_size: int,
    state: DemoState,
    args: argparse.Namespace,
    action: str,
    raw_action: str,
    refined_action: str,
    goal_action: str,
    reason: str,
    controller_reason: str,
    goal_reason: str,
    min_distance: float | None,
    risk: str,
    goal_distance: float,
    goal_reached: bool,
    paused: bool,
) -> None:
    import cv2

    x0 = scene_size
    cv2.rectangle(frame, (x0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), (245, 247, 250), -1)
    cv2.line(frame, (x0, 0), (x0, frame.shape[0]), (195, 200, 208), 1)
    rows = [
        ("scenario", args.scenario),
        ("model", args.model),
        ("raw", raw_action),
        ("avoid", refined_action),
        ("route act", goal_action),
        ("final", action),
        ("phase", state.route_phase),
        ("target lane", str(state.target_lane)),
        ("speed", f"{args.robot_speed:.2f}"),
        ("min dist", "--" if min_distance is None else f"{min_distance:.2f}"),
        ("risk", risk),
        ("goal", f"{state.driving_goal[0]:.1f},{state.driving_goal[1]:.1f}"),
        ("goal dist", f"{goal_distance:.2f}"),
        ("goal state", "reached" if goal_reached else "seeking"),
        ("goal src", args.goal_source),
        ("state", "paused" if paused else "running"),
    ]
    y = 42
    cv2.putText(frame, "Robot Navigation", (x0 + 18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (30, 30, 30), 2, cv2.LINE_AA)
    y += 38
    for label, value in rows:
        cv2.putText(frame, label, (x0 + 18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (90, 90, 90), 1, cv2.LINE_AA)
        cv2.putText(frame, value, (x0 + 105, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (25, 25, 25), 1, cv2.LINE_AA)
        y += 28
    y += 18
    cv2.putText(frame, "Controls", (x0 + 18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (30, 30, 30), 2, cv2.LINE_AA)
    y += 28
    for text in ["q quit", "r reset", "g robot reset", "0 cycle goal", "space pause", "t true path", "b background", "1-9 scenarios"]:
        cv2.putText(frame, text, (x0 + 18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 70, 70), 1, cv2.LINE_AA)
        y += 24
    y += 12
    cv2.putText(frame, controller_reason[:30], (x0 + 18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 80), 1, cv2.LINE_AA)
    y += 22
    cv2.putText(frame, goal_reason[:30], (x0 + 18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 80), 1, cv2.LINE_AA)
    y += 22
    cv2.putText(frame, reason[:30], (x0 + 18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 80), 1, cv2.LINE_AA)


def draw_virtual_frame(
    state: DemoState,
    main_position: tuple[float, float],
    pred_world: list[tuple[float, float]],
    true_future: list[tuple[float, float]],
    action: str,
    raw_action: str,
    refined_action: str,
    goal_action: str,
    reason: str,
    controller_reason: str,
    goal_reason: str,
    min_distance: float | None,
    risk: str,
    goal_distance: float,
    goal_reached: bool,
    args: argparse.Namespace,
) -> object:
    import cv2
    import numpy as np

    scene_size = args.window_size
    panel_width = 260 if args.show_status_panel else 0
    frame = np.full((scene_size, scene_size + panel_width, 3), 248, dtype=np.uint8)
    min_x, max_x, min_y, max_y = WORLD_BOUNDS

    for ix in range(math.floor(min_x), math.ceil(max_x)):
        for iy in range(math.floor(min_y), math.ceil(max_y)):
            p1 = world_to_pixel((ix, iy + 1), scene_size)
            p2 = world_to_pixel((ix + 1, iy), scene_size)
            color = (238, 241, 245) if (ix + iy) % 2 == 0 else (255, 255, 255)
            cv2.rectangle(frame, p1, p2, color, -1)
            cv2.rectangle(frame, p1, p2, (210, 216, 224), 1)

    draw_world_markings(frame, scene_size)
    draw_goal(frame, state, scene_size, args)

    safety_local = [(-1.0, 0.0), (1.0, 0.0), (1.0, 3.0), (-1.0, 3.0)]
    safety_world = [local_to_world(point, state.robot) for point in safety_local]
    safety_pixels = np.array([world_to_pixel(point, scene_size) for point in safety_world], dtype=np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [safety_pixels], (70, 70, 255))
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
    cv2.polylines(frame, [safety_pixels], True, (20, 20, 220), 2)
    cv2.putText(frame, "safety zone", world_to_pixel(safety_world[-1], scene_size), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (80, 80, 180), 1, cv2.LINE_AA)

    if state.show_background:
        for ped in state.background_peds:
            cv2.circle(frame, world_to_pixel((ped["x"], ped["y"]), scene_size), 5, (135, 135, 135), -1)

    if args.show_robot_trail:
        draw_path(frame, list(state.robot_history), scene_size, (70, 165, 165), radius=2, thickness=2)
    draw_path(frame, list(state.pedestrian_history), scene_size, (220, 90, 20), radius=4)
    if state.show_true_future:
        draw_dashed_path(frame, true_future, scene_size, (55, 160, 65))
    draw_fading_prediction(frame, pred_world, scene_size)

    cv2.circle(frame, world_to_pixel(main_position, scene_size), 10, (220, 90, 20), -1)
    cv2.circle(frame, world_to_pixel(main_position, scene_size), 12, (255, 255, 255), 2)

    robot_px = world_to_pixel((state.robot[0], state.robot[1]), scene_size)
    fx, fy = forward_vector(state.robot[2])
    nose = world_to_pixel((state.robot[0] + fx * 0.55, state.robot[1] + fy * 0.55), scene_size)
    cv2.circle(frame, robot_px, 13, (30, 120, 125), -1)
    cv2.circle(frame, robot_px, 15, (20, 20, 20), 2)
    cv2.arrowedLine(frame, robot_px, nose, (20, 20, 20), 2, tipLength=0.35)

    action_color = {
        "stop": (20, 20, 220),
        "go": (35, 140, 35),
        "turn_left": (0, 140, 220),
        "turn_right": (0, 140, 220),
        "change_left": (0, 160, 200),
        "change_right": (0, 160, 200),
        "warming_up": (90, 90, 90),
    }.get(action, (70, 70, 70))
    cv2.rectangle(frame, (20, 18), (440, 102), action_color, -1)
    cv2.putText(frame, f"Action: {action}", (34, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.86, (255, 255, 255), 2, cv2.LINE_AA)
    badge = f"{args.scenario} | risk {risk}"
    if goal_reached:
        badge = f"{args.scenario} | goal reached"
    cv2.putText(frame, badge, (34, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(
        frame,
        "blue=pedestrian  red=prediction  green=true future  teal=robot",
        (20, scene_size - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (35, 35, 35),
        1,
        cv2.LINE_AA,
    )

    if args.show_status_panel:
        draw_status_panel(
            frame,
            scene_size,
            state,
            args,
            action,
            raw_action,
            refined_action,
            goal_action,
            reason,
            controller_reason,
            goal_reason,
            min_distance,
            risk,
            goal_distance,
            goal_reached,
            state.paused,
        )
    return frame


@torch.no_grad()
def predict_local_future(model, feature_mode: str, local_history: deque[tuple[float, float]], device: str) -> torch.Tensor:
    features, last_pos, mask, cv_obs = build_live_inputs(local_history, device, feature_mode)
    if isinstance(model, ConstantVelocityPredictor):
        return model.predict(cv_obs, mask)[0].detach().cpu()
    return model(features, last_pos, mask=mask, cv_obs=cv_obs)[0].detach().cpu()


def predict_step(
    args: argparse.Namespace,
    model,
    feature_mode: str,
    state: DemoState,
) -> tuple[tuple[float, float], list[tuple[float, float]], list[tuple[float, float]], str, str, torch.Tensor | None]:
    main_position = scenario_position(args.scenario, state.step)
    state.pedestrian_history.append(main_position)
    pred_world: list[tuple[float, float]] = []
    pred_local = None
    raw_action = "warming_up"
    reason = "filling observation buffer"

    if len(state.pedestrian_history) == args.obs_len:
        local_history = deque((world_to_local(point, state.robot) for point in state.pedestrian_history), maxlen=args.obs_len)
        pred_local = predict_local_future(model, feature_mode, local_history, args.device)
        pred_world = [local_to_world((float(point[0]), float(point[1])), state.robot) for point in pred_local]
        decision = decide_navigation_action(pred_local)
        raw_action = decision.action
        reason = decision.reason

    true_future = [scenario_position(args.scenario, state.step + i + 1) for i in range(args.pred_len)]
    return main_position, pred_world, true_future, raw_action, reason, pred_local


def advance_state(args: argparse.Namespace, state: DemoState, goal_action: str, goal_reason: str) -> str:
    route_driven = goal_reason not in ("avoidance_priority", "avoidance_medium_priority")
    cooldown_on_stop = goal_reason in ("avoidance_priority", "avoidance_medium_priority")
    action = smooth_action(
        state,
        goal_action,
        args,
        immediate=route_driven and goal_action in ("go", "turn_left", "turn_right", "change_left", "change_right"),
        cooldown_on_stop=cooldown_on_stop,
    )
    state.robot = update_robot(state.robot, action, state, args)
    state.robot_history.append((state.robot[0], state.robot[1]))
    update_background_pedestrians(state.background_peds, state.step)
    state.step += 1
    return action


def cycle_goal(args: argparse.Namespace, state: DemoState) -> DemoState:
    current = (args.goal_x, args.goal_y)
    closest_index = min(
        range(len(GOAL_PRESETS)),
        key=lambda idx: math.hypot(GOAL_PRESETS[idx][0] - current[0], GOAL_PRESETS[idx][1] - current[1]),
    )
    next_goal = GOAL_PRESETS[(closest_index + 1) % len(GOAL_PRESETS)]
    args.goal_x, args.goal_y = next_goal
    args.goal_source = "cycled"
    state.goal = next_goal
    state.driving_goal = snap_goal_to_road(next_goal, args)
    state.target_lane = nearest_lane_index(state.driving_goal[0]) if not is_side_goal(state.driving_goal, args) else 1
    state.route_phase = "goal_cycled"
    state.goal_reached = False
    state.stop_cooldown = 0
    print_goal(args)
    return state


def handle_key(key: int, args: argparse.Namespace, state: DemoState) -> DemoState:
    if key == ord("r"):
        return reset_state(args)
    if key == ord("g"):
        return reset_state(args)
    if key == ord("0"):
        return cycle_goal(args, state)
    if key == ord("t"):
        state.show_true_future = not state.show_true_future
    elif key == ord("b"):
        state.show_background = not state.show_background
    elif key == ord(" "):
        state.paused = not state.paused
    elif key in KEY_SCENARIOS:
        args.scenario = KEY_SCENARIOS[key]
        if not args.custom_goal:
            set_scenario_goal(args)
        print_goal(args)
        return reset_state(args)
    return state


def run_demo(args: argparse.Namespace) -> None:
    import cv2

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model, pred_len, feature_mode = make_model(args)
    args.pred_len = pred_len
    state = reset_state(args)

    print("Mode: virtual moving-robot navigation demo.")
    print(f"Scenario: {args.scenario}; model={args.model}; feature_mode={feature_mode}; pred_len={pred_len}")
    print_goal(args)
    print("Controls: q quit, r reset, g reset robot, 0 cycle goal, space pause, t true future, b background, 1-9 scenarios.")

    headless_steps = max(args.obs_len + 18, 34)
    if is_side_goal(snap_goal_to_road((args.goal_x, args.goal_y), args), args):
        headless_steps = max(headless_steps, 70)
    frame = None
    while True:
        if state.paused and frame is not None:
            if not args.headless:
                cv2.imshow("Virtual moving-robot navigation demo", frame)
                key = cv2.waitKey(args.fps_delay_ms) & 0xFF
                if key == ord("q"):
                    break
                state = handle_key(key, args, state)
            continue

        main_position, pred_world, true_future, raw_action, reason, pred_local = predict_step(args, model, feature_mode, state)
        min_distance, risk = risk_metrics(pred_local, args)
        refined_action, controller_reason = refine_robot_action(
            raw_action,
            pred_local,
            min_distance,
            risk,
            state,
            args,
        )
        goal_action, goal_reason, goal_distance, goal_reached = apply_goal_navigation(refined_action, risk, state, args)
        action = advance_state(args, state, goal_action, goal_reason)
        frame = draw_virtual_frame(
            state,
            main_position,
            pred_world,
            true_future,
            action,
            raw_action,
            refined_action,
            goal_action,
            reason,
            controller_reason,
            goal_reason,
            min_distance,
            risk,
            goal_distance,
            goal_reached,
            args,
        )

        if args.headless and state.step >= headless_steps:
            save_path = Path(args.save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), frame)
            print(f"Saved virtual demo: {save_path}")
            print(
                {
                    "scenario": args.scenario,
                    "action": action,
                    "raw_action": raw_action,
                    "refined_action": refined_action,
                    "goal_action": goal_action,
                    "risk": risk,
                    "goal_distance": goal_distance,
                    "goal_reached": goal_reached,
                    "controller_reason": controller_reason,
                    "goal_reason": goal_reason,
                    "reason": reason,
                }
            )
            return

        if not args.headless:
            cv2.imshow("Virtual moving-robot navigation demo", frame)
            key = cv2.waitKey(args.fps_delay_ms) & 0xFF
            if key == ord("q"):
                break
            state = handle_key(key, args, state)
            if state.goal_reached and args.auto_reset_on_goal:
                state = reset_state(args)

    cv2.destroyAllWindows()


def main() -> None:
    run_demo(parse_args())


if __name__ == "__main__":
    main()
