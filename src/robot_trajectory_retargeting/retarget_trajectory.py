from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.model.kinematics import RobotKinematics

from robot_trajectory_retargeting.frames import (
    make_replay_timestamps,
    pose_error,
    smooth_joint_trajectory,
    smooth_pose_trajectory,
)
from robot_trajectory_retargeting.urdf_chain import compute_canonical_tcp, read_joint_limits


def load_input_file(path: Path) -> Any:
    if path.suffix == ".npy":
        data = np.load(path, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.dtype == object and data.shape == (1,):
            return data.item()
        return data
    if path.suffix == ".npz":
        return dict(np.load(path, allow_pickle=True))
    if path.suffix in {".pkl", ".pickle"}:
        with path.open("rb") as handle:
            return pickle.load(handle)
    raise ValueError(f"Unsupported input file format: {path.suffix}")


def resolve_data_key(data: Any, key: str | None) -> Any:
    if key is None or key == "":
        return data
    current = data
    for part in key.split("."):
        if isinstance(current, dict):
            current = current[part]
        else:
            current = getattr(current, part)
    return current


def parse_names(spec: str | None) -> list[str] | None:
    if spec is None or spec.strip() == "":
        return None
    return [part.strip() for part in spec.split(",") if part.strip()]


def parse_slice(spec: str | None, total_columns: int) -> np.ndarray:
    if spec is None or spec.strip() == "":
        return np.arange(total_columns, dtype=int)
    spec = spec.strip()
    if "," in spec:
        return np.array([int(part.strip()) for part in spec.split(",") if part.strip()], dtype=int)
    if ":" in spec:
        start_text, stop_text, *rest = spec.split(":")
        step_text = rest[0] if rest else ""
        start = None if start_text == "" else int(start_text)
        stop = None if stop_text == "" else int(stop_text)
        step = None if step_text == "" else int(step_text)
        return np.arange(total_columns)[slice(start, stop, step)]
    return np.array([int(spec)], dtype=int)


def parse_vector(spec: str | None, expected: int, units: str, default_deg: np.ndarray) -> np.ndarray:
    if spec is None or spec.strip() == "":
        return default_deg.copy()
    values = np.array([float(part.strip()) for part in spec.split(",") if part.strip()], dtype=float)
    if values.shape != (expected,):
        raise ValueError(f"Expected {expected} values, got {values.shape}")
    return np.rad2deg(values) if units == "rad" else values


def _parse_axis(spec: str | None, label: str) -> np.ndarray | None:
    if spec is None or spec.strip() == "":
        return None
    values = np.array([float(part.strip()) for part in spec.split(",") if part.strip()], dtype=float)
    if values.shape != (3,):
        raise ValueError(f"{label} must be x,y,z")
    return values


def reshape_pose_trajectory(pose_array: np.ndarray) -> np.ndarray:
    pose_array = np.asarray(pose_array, dtype=float)
    if pose_array.ndim == 3 and pose_array.shape[1:] == (4, 4):
        return pose_array
    if pose_array.ndim == 2 and pose_array.shape[1] == 16:
        return pose_array.reshape(-1, 4, 4)
    if pose_array.ndim == 2 and pose_array.shape[1] == 12:
        output = np.tile(np.eye(4, dtype=float), (pose_array.shape[0], 1, 1))
        output[:, :3, :] = pose_array.reshape(-1, 3, 4)
        return output
    raise ValueError("Pose input must have shape (T,4,4), (T,16), or (T,12).")


def normalize_gripper_openness(raw: np.ndarray, open_value: float | None, closed_value: float | None) -> np.ndarray:
    raw = np.asarray(raw, dtype=float).reshape(-1)
    if open_value is None or closed_value is None:
        min_value = float(np.min(raw))
        max_value = float(np.max(raw))
        if np.isclose(min_value, max_value):
            return np.full_like(raw, 1.0, dtype=float)
        closed_value = min_value if closed_value is None else closed_value
        open_value = max_value if open_value is None else open_value
    scale = float(open_value) - float(closed_value)
    if abs(scale) < 1e-9:
        return np.full_like(raw, 1.0, dtype=float)
    return np.clip((raw - float(closed_value)) / scale, 0.0, 1.0)


def extract_gripper_fraction(payload: Any, key: str | None, args: argparse.Namespace, frames: int) -> np.ndarray | None:
    if not key:
        return None
    raw = np.asarray(resolve_data_key(payload, key), dtype=float).reshape(-1)
    if len(raw) != frames:
        raise ValueError("Gripper trajectory length does not match source trajectory length.")
    if args.source_gripper_units == "fraction":
        fraction = np.clip(raw, 0.0, 1.0)
    else:
        if args.source_gripper_open_value is None or args.source_gripper_closed_value is None:
            print(
                "WARNING: gripper open/closed values not given; normalizing from the demo's own "
                "min/max. If the demo never fully opens or closes the gripper this stretches the range."
            )
        fraction = normalize_gripper_openness(raw, args.source_gripper_open_value, args.source_gripper_closed_value)
    if args.source_gripper_max_aperture_m and args.target_gripper_max_aperture_m:
        # Map physical jaw opening, not fraction-of-stroke: "half open" on a wide gripper
        # is a different object width than on a narrow one.
        fraction = np.clip(
            fraction * float(args.source_gripper_max_aperture_m) / float(args.target_gripper_max_aperture_m),
            0.0,
            1.0,
        )
    return fraction


def load_source(args: argparse.Namespace) -> tuple[Any, np.ndarray | None, np.ndarray | None, np.ndarray]:
    payload = load_input_file(Path(args.input).expanduser().resolve())
    source_data = resolve_data_key(payload, args.input_data_key)

    if args.source_mode == "joints":
        joints = np.asarray(source_data, dtype=float)
        if joints.ndim != 2:
            raise ValueError(f"Expected 2D joint trajectory, got {joints.shape}")
        joints = joints[:, parse_slice(args.input_joint_slice, joints.shape[1])]
        joints_deg = np.rad2deg(joints) if args.input_units == "rad" else joints
        poses = None
        frames = len(joints_deg)
    else:
        poses = reshape_pose_trajectory(np.asarray(source_data, dtype=float))
        joints_deg = None
        frames = len(poses)

    if args.input_time_key:
        timestamps = np.asarray(resolve_data_key(payload, args.input_time_key), dtype=float).reshape(-1)
        if len(timestamps) != frames:
            raise ValueError("input-time-key length does not match source trajectory length.")
    else:
        timestamps = np.arange(frames, dtype=float) * args.input_dt
    return payload, joints_deg, poses, timestamps


def make_kinematics(urdf: str, tcp_link: str, joint_names: str | None) -> RobotKinematics:
    return RobotKinematics(
        urdf_path=str(Path(urdf).expanduser().resolve()),
        target_frame_name=tcp_link,
        joint_names=parse_names(joint_names),
    )


def compute_source_poses(source_kinematics: RobotKinematics, source_joints_deg: np.ndarray) -> np.ndarray:
    return np.asarray([source_kinematics.forward_kinematics(joint_frame) for joint_frame in source_joints_deg], dtype=float)


def _solve_pose_converged(
    target_kinematics: RobotKinematics,
    desired_pose: np.ndarray,
    seed_joints_deg: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Iterate placo's single-QP-step IK until the pose error stops improving.

    One `inverse_kinematics` call is one solver step, not a converged solve; judging
    reachability from a single step conflates "unreachable" with "didn't converge yet".
    """
    current_joints = np.asarray(seed_joints_deg, dtype=float).copy()
    best: tuple[float, np.ndarray, np.ndarray, float, float] | None = None
    stall_count = 0
    for _ in range(max(1, int(args.ik_max_iters_per_frame))):
        current_joints = target_kinematics.inverse_kinematics(
            current_joint_pos=current_joints,
            desired_ee_pose=desired_pose,
            position_weight=args.position_weight,
            orientation_weight=args.orientation_weight,
        )
        achieved_pose = target_kinematics.forward_kinematics(current_joints)
        position_error, orientation_error = pose_error(achieved_pose, desired_pose)
        cost = position_error
        if float(args.orientation_weight) > 0.0:
            cost += 0.1 * orientation_error
        if best is None or cost < best[0] - 1e-8:
            best = (cost, current_joints.copy(), achieved_pose, position_error, orientation_error)
            stall_count = 0
        else:
            # The QP can plateau for several steps near joint-limit corners before
            # descending again; require a sustained stall before giving up.
            stall_count += 1
            if stall_count >= 8:
                break
    assert best is not None
    return best[1], best[2], best[3], best[4]


def _sample_seeds(
    joint_limits_deg: list[tuple[float, float]],
    count: int,
    rng: np.random.Generator,
    include: list[np.ndarray] | None = None,
) -> list[np.ndarray]:
    """Random joint configurations across the URDF limits, for global IK multi-start."""
    lows = np.array([lo if np.isfinite(lo) else -180.0 for lo, _ in joint_limits_deg], dtype=float)
    highs = np.array([hi if np.isfinite(hi) else 180.0 for _, hi in joint_limits_deg], dtype=float)
    seeds = [np.asarray(s, dtype=float) for s in (include or [])]
    for _ in range(max(0, count)):
        seeds.append(rng.uniform(lows, highs))
    return seeds


def _cost(position_error: float, orientation_error: float, orientation_enabled: bool) -> float:
    return position_error + (0.1 * orientation_error if orientation_enabled else 0.0)


def solve_with_lerobot_ik(
    target_kinematics: RobotKinematics,
    desired_poses: np.ndarray,
    initial_joints_deg: np.ndarray,
    args: argparse.Namespace,
    joint_limits_deg: list[tuple[float, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    current_joints = np.asarray(initial_joints_deg, dtype=float).copy()
    raw_joints = []
    achieved_poses = []
    position_errors = []
    orientation_errors = []

    if len(desired_poses) == 0:
        return (
            np.empty((0, len(current_joints)), dtype=float),
            np.empty((0, 4, 4), dtype=float),
            np.empty((0,), dtype=float),
            np.empty((0,), dtype=float),
        )

    rng = np.random.default_rng(0)
    orientation_enabled = float(args.orientation_weight) > 0.0
    for frame_index, desired_pose in enumerate(desired_poses):
        if frame_index == 0 and args.auto_seed and joint_limits_deg is not None:
            # Global multi-start on the FIRST frame so we don't depend on a hand-picked
            # starting pose to land the right IK branch. A pose-optimal frame-0 branch can
            # still fail to *continue*, so we keep the best few seeds and pick the one whose
            # short continuation (the next few frames) stays smoothest.
            scored = []
            for seed in _sample_seeds(joint_limits_deg, int(args.auto_seed_samples), rng, [current_joints]):
                candidate = _solve_pose_converged(target_kinematics, desired_pose, seed, args)
                scored.append((_cost(candidate[2], candidate[3], orientation_enabled), candidate))
            scored.sort(key=lambda item: item[0])
            lookahead = desired_poses[1 : 1 + min(3, len(desired_poses) - 1)]
            best, best_branch_cost = None, np.inf
            for _, candidate in scored[:5]:
                joints = candidate[0]
                branch_cost = _cost(candidate[2], candidate[3], orientation_enabled)
                for ahead_pose in lookahead:
                    joints, _, perr, oerr = _solve_pose_converged(target_kinematics, ahead_pose, joints, args)
                    branch_cost = max(branch_cost, _cost(perr, oerr, orientation_enabled))
                if branch_cost < best_branch_cost:
                    best_branch_cost, best = branch_cost, candidate
            solution, achieved_pose, position_error, orientation_error = best
            current_joints = solution
            raw_joints.append(solution.copy())
            achieved_poses.append(achieved_pose)
            position_errors.append(position_error)
            orientation_errors.append(orientation_error)
            continue
        solution, achieved_pose, position_error, orientation_error = _solve_pose_converged(
            target_kinematics, desired_pose, current_joints, args
        )
        # The converged residual is the reachability signal; before trusting a bad one,
        # rule out a local minimum / wrong elbow branch with a few alternative seeds.
        needs_restart = position_error > float(args.ik_position_tolerance_m) or (
            orientation_enabled and orientation_error > float(args.orientation_warning_threshold_rad)
        )
        if needs_restart:
            initial = np.asarray(initial_joints_deg, dtype=float)
            seeds = [initial]
            for restart in range(max(0, int(args.ik_restarts))):
                # Alternate perturbation centers: the stuck solution may sit in a local
                # minimum whose basin the user's initial pose is outside of (and vice
                # versa), so escape attempts must straddle both.
                center = initial if restart % 2 == 0 else current_joints
                seeds.append(center + rng.uniform(-90.0, 90.0, size=current_joints.shape))
            for seed in seeds:
                candidate = _solve_pose_converged(target_kinematics, desired_pose, seed, args)
                candidate_cost = candidate[2] + (0.1 * candidate[3] if orientation_enabled else 0.0)
                best_cost = position_error + (0.1 * orientation_error if orientation_enabled else 0.0)
                if candidate_cost < best_cost:
                    solution, achieved_pose, position_error, orientation_error = candidate
        current_joints = solution
        raw_joints.append(solution.copy())
        achieved_poses.append(achieved_pose)
        position_errors.append(position_error)
        orientation_errors.append(orientation_error)

    return (
        np.asarray(raw_joints, dtype=float),
        np.asarray(achieved_poses, dtype=float),
        np.asarray(position_errors, dtype=float),
        np.asarray(orientation_errors, dtype=float),
    )


def _probe_shift(
    target_kinematics: RobotKinematics,
    desired_rotations: np.ndarray,
    base_positions: np.ndarray,
    shift: np.ndarray,
    joint_limits_deg: list[tuple[float, float]],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Feasible-frame fraction + worst error for one candidate mount shift, on a subset.

    Desired EE positions only depend on the shift (orientation does not), so a candidate
    mount placement is the shift-zero positions minus the shift.
    """
    orientation_enabled = float(args.orientation_weight) > 0.0
    feasible = 0
    worst = 0.0
    current = None
    for index in range(len(base_positions)):
        pose = np.eye(4, dtype=float)
        pose[:3, :3] = desired_rotations[index]
        pose[:3, 3] = base_positions[index] - shift
        if current is None:
            best = None
            for seed in _sample_seeds(joint_limits_deg, max(8, int(args.auto_seed_samples) // 4), rng):
                candidate = _solve_pose_converged(target_kinematics, pose, seed, args)
                if best is None or _cost(candidate[2], candidate[3], orientation_enabled) < _cost(
                    best[2], best[3], orientation_enabled
                ):
                    best = candidate
            current, position_error = best[0], best[2]
        else:
            current, _, position_error, _ = _solve_pose_converged(target_kinematics, pose, current, args)
        worst = max(worst, position_error)
        if position_error <= float(args.reachability_warning_threshold_m):
            feasible += 1
    return feasible / len(base_positions), worst


def auto_mount_shift(
    target_kinematics: RobotKinematics,
    desired_rotations: np.ndarray,
    base_positions: np.ndarray,
    mount_offset: np.ndarray,
    max_reach: float,
    joint_limits_deg: list[tuple[float, float]],
    args: argparse.Namespace,
) -> np.ndarray:
    """Pick a mount placement automatically: keep the shared mount (zero shift) if the task
    is reachable from it, otherwise bolt the arm back and up (a pedestal) so the fixed task
    falls in a comfortable, reach-DOWN region of the workspace. Returns a base-frame shift.

    Candidates are scored first by reachability (feasible-frame fraction on a subset), then
    by workspace quality: the task centroid should sit at a comfortable fraction of reach
    and BELOW the shoulder so the arm reaches down with its elbow up -- a cheap proxy for
    "elbow doesn't dip under the desk", since pure EE reachability can't see that collision.
    Ties go to the smallest relocation, so a desktop arm that already reaches stays put.
    """
    n = len(base_positions)
    index = np.linspace(0, n - 1, min(n, 12)).astype(int)
    rotations = desired_rotations[index]
    positions = base_positions[index]
    centroid = base_positions.mean(axis=0) - mount_offset  # task centroid relative to shoulder
    rng = np.random.default_rng(1)

    # Keep the shared mount whenever the task is reachable from it -- the full solver's
    # restarts clean up the odd marginal frame, so a tolerant probe threshold is fine. Only
    # relocate when the shared mount genuinely can't reach (a big arm folding into its inner
    # workspace), which avoids needlessly moving a desktop arm that was already fine.
    shared_fraction, _ = _probe_shift(
        target_kinematics, rotations, positions, np.zeros(3), joint_limits_deg, args, rng
    )
    if shared_fraction >= 0.85:
        return np.zeros(3, dtype=float)

    def quality_penalty(shift: np.ndarray) -> float:
        rel = centroid - shift  # task centroid relative to the shoulder after the shift
        horizontal = float(np.hypot(rel[0], rel[1]))
        # want the task at ~55% of reach horizontally, and at/below shoulder height so the
        # arm reaches DOWN (elbow up) -- a cheap proxy for "elbow clears the desk".
        return abs(horizontal - 0.55 * max_reach) + 1.5 * max(0.0, rel[2] + 0.02)

    best_shift = np.zeros(3, dtype=float)
    best_score = None  # maximize (feasible fraction, -quality penalty, -|shift|)
    for back in (0.1, 0.2, 0.3, 0.4, 0.5):
        for up in (0.0, 0.15, 0.30):
            shift = np.array([-back, 0.0, up], dtype=float)  # task forward + down
            fraction, _ = _probe_shift(
                target_kinematics, rotations, positions, shift, joint_limits_deg, args, rng
            )
            score = (round(fraction, 2), -quality_penalty(shift), -float(np.linalg.norm(shift)))
            if best_score is None or score > best_score:
                best_score, best_shift = score, shift
    return best_shift


def build_retarget_diagnostics(
    position_errors_m: np.ndarray,
    orientation_errors_rad: np.ndarray,
    desired_poses: np.ndarray,
    achieved_poses: np.ndarray,
    args: argparse.Namespace,
    frames_beyond_reach: np.ndarray | None = None,
    target_max_reach_m: float = float("nan"),
    joint_jump_frames: np.ndarray | None = None,
    joints_at_limit_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    frame_count = int(len(position_errors_m))
    warning_threshold_m = float(args.reachability_warning_threshold_m)
    orientation_warning_threshold_rad = float(args.orientation_warning_threshold_rad)
    ik_tolerance_m = float(args.ik_position_tolerance_m)
    frames_over_ik_tolerance = np.flatnonzero(position_errors_m > ik_tolerance_m).astype(int)
    frames_over_warning_threshold = np.flatnonzero(position_errors_m > warning_threshold_m).astype(int)
    frames_over_orientation_threshold = np.flatnonzero(
        orientation_errors_rad > orientation_warning_threshold_rad
    ).astype(int)
    position_error_vectors_m = achieved_poses[:, :3, 3] - desired_poses[:, :3, 3]

    if frame_count == 0:
        max_position_error_frame = -1
        max_position_error_m = 0.0
        max_position_error_vector_m = np.zeros(3, dtype=float)
        position_error_mean_m = 0.0
        position_error_p95_m = 0.0
        orientation_error_mean_rad = 0.0
        orientation_error_p95_rad = 0.0
        orientation_error_max_rad = 0.0
    else:
        max_position_error_frame = int(np.argmax(position_errors_m))
        max_position_error_m = float(position_errors_m[max_position_error_frame])
        max_position_error_vector_m = position_error_vectors_m[max_position_error_frame]
        position_error_mean_m = float(position_errors_m.mean())
        position_error_p95_m = float(np.percentile(position_errors_m, 95))
        orientation_error_mean_rad = float(orientation_errors_rad.mean())
        orientation_error_p95_rad = float(np.percentile(orientation_errors_rad, 95))
        orientation_error_max_rad = float(orientation_errors_rad.max())

    frames_beyond_reach = (
        np.empty(0, dtype=int) if frames_beyond_reach is None else np.asarray(frames_beyond_reach, dtype=int)
    )
    joint_jump_frames = (
        np.empty(0, dtype=int) if joint_jump_frames is None else np.asarray(joint_jump_frames, dtype=int)
    )
    joints_at_limit_counts = joints_at_limit_counts or {}

    orientation_check_enabled = float(args.orientation_weight) > 0.0
    retarget_feasible = (
        len(frames_over_warning_threshold) == 0
        and len(frames_beyond_reach) == 0
        and (not orientation_check_enabled or len(frames_over_orientation_threshold) == 0)
    )
    if retarget_feasible:
        reason = f"max XYZ error {max_position_error_m:.4f}m is within {warning_threshold_m:.4f}m"
        if orientation_check_enabled:
            reason += f"; max RPY-equivalent error {orientation_error_max_rad:.4f}rad is within {orientation_warning_threshold_rad:.4f}rad"
    else:
        reasons = []
        if len(frames_beyond_reach):
            reasons.append(
                f"{len(frames_beyond_reach)}/{frame_count} frames lie beyond the target arm's "
                f"maximum reach ({target_max_reach_m:.3f}m from its mount)"
            )
        if len(frames_over_warning_threshold):
            reasons.append(
                f"{len(frames_over_warning_threshold)}/{frame_count} frames exceed "
                f"{warning_threshold_m:.4f}m XYZ threshold"
            )
        if orientation_check_enabled and len(frames_over_orientation_threshold):
            reasons.append(
                f"{len(frames_over_orientation_threshold)}/{frame_count} frames exceed "
                f"{orientation_warning_threshold_rad:.4f}rad orientation threshold"
            )
        saturated = [name for name, count in joints_at_limit_counts.items() if count > 0]
        if saturated:
            reasons.append("joints at their limits: " + ", ".join(saturated))
        reasons.append(f"worst XYZ frame {max_position_error_frame} has {max_position_error_m:.4f}m error")
        reason = "; ".join(reasons)

    return {
        "frames_beyond_reach": frames_beyond_reach,
        "target_max_reach_m": float(target_max_reach_m),
        "joint_jump_frames": joint_jump_frames,
        "joints_at_limit_counts": joints_at_limit_counts,
        "retarget_feasible": retarget_feasible,
        "retarget_feasibility_status": "ok" if retarget_feasible else "not_feasible",
        "retarget_feasibility_reason": reason,
        "ik_position_tolerance_m": ik_tolerance_m,
        "reachability_warning_threshold_m": warning_threshold_m,
        "orientation_warning_threshold_rad": orientation_warning_threshold_rad,
        "position_error_vectors_m": position_error_vectors_m,
        "position_error_mean_m": position_error_mean_m,
        "position_error_p95_m": position_error_p95_m,
        "position_error_max_m": max_position_error_m,
        "position_error_max_frame": max_position_error_frame,
        "max_position_error_vector_m": max_position_error_vector_m,
        "orientation_error_mean_rad": orientation_error_mean_rad,
        "orientation_error_p95_rad": orientation_error_p95_rad,
        "orientation_error_max_rad": orientation_error_max_rad,
        "frames_over_ik_tolerance": frames_over_ik_tolerance,
        "frames_over_reachability_threshold": frames_over_warning_threshold,
        "frames_over_orientation_threshold": frames_over_orientation_threshold,
    }


def save_result(
    path: Path,
    args: argparse.Namespace,
    source_joints_deg: np.ndarray | None,
    source_poses: np.ndarray,
    canonical_poses: np.ndarray,
    desired_poses: np.ndarray,
    raw_target_joints_deg: np.ndarray,
    smoothed_target_joints_deg: np.ndarray,
    achieved_poses: np.ndarray,
    position_errors_m: np.ndarray,
    orientation_errors_rad: np.ndarray,
    replay_timestamps_s: np.ndarray,
    target_joint_names: list[str],
    target_output_joint_names: list[str],
    gripper_fraction: np.ndarray | None,
    source_timestamps_s: np.ndarray,
    source_canonical_rotation: np.ndarray,
    target_canonical_rotation: np.ndarray,
    source_mount_offset: np.ndarray,
    target_mount_offset: np.ndarray,
    retarget_diagnostics: dict[str, Any],
    executed_poses: np.ndarray,
    executed_position_errors: np.ndarray,
    executed_orientation_errors: np.ndarray,
    source_base_rotation: np.ndarray,
    target_base_rotation: np.ndarray,
    target_mount_shift: np.ndarray,
) -> None:
    action_positions_deg = smoothed_target_joints_deg.copy()
    action_names = list(target_output_joint_names)

    if gripper_fraction is not None and args.target_gripper_name:
        if args.target_gripper_open_deg is None or args.target_gripper_closed_deg is None:
            raise ValueError("target gripper mapping requires open and closed degree values.")
        gripper_deg = float(args.target_gripper_closed_deg) + gripper_fraction * (
            float(args.target_gripper_open_deg) - float(args.target_gripper_closed_deg)
        )
        action_positions_deg = np.concatenate([action_positions_deg, gripper_deg[:, None]], axis=1)
        action_names.append(args.target_gripper_name)

    if source_joints_deg is None:
        source_joint_positions_rad = np.zeros((len(source_poses), 0), dtype=float)
    else:
        source_joint_positions_rad = np.deg2rad(source_joints_deg)

    payload = {
        "source_joint_positions_rad": source_joint_positions_rad,
        "source_end_effector_poses": source_poses,
        "canonical_ee_poses": canonical_poses,
        "desired_target_poses": desired_poses,
        "raw_target_joint_positions_rad": np.deg2rad(raw_target_joints_deg),
        "smoothed_target_joint_positions_rad": np.deg2rad(smoothed_target_joints_deg),
        "raw_target_joint_positions_deg": raw_target_joints_deg,
        "smoothed_target_joint_positions_deg": smoothed_target_joints_deg,
        "achieved_target_poses": achieved_poses,
        "ik_success": position_errors_m <= args.ik_position_tolerance_m,
        "position_errors_m": position_errors_m,
        "orientation_errors_rad": orientation_errors_rad,
        "position_error_vectors_m": retarget_diagnostics["position_error_vectors_m"],
        "retarget_feasible": np.asarray(retarget_diagnostics["retarget_feasible"], dtype=bool),
        "retarget_feasibility_status": np.asarray(
            retarget_diagnostics["retarget_feasibility_status"], dtype=object
        ),
        "retarget_feasibility_reason": np.asarray(
            retarget_diagnostics["retarget_feasibility_reason"], dtype=object
        ),
        "ik_position_tolerance_m": np.asarray(retarget_diagnostics["ik_position_tolerance_m"], dtype=float),
        "reachability_warning_threshold_m": np.asarray(
            retarget_diagnostics["reachability_warning_threshold_m"], dtype=float
        ),
        "orientation_warning_threshold_rad": np.asarray(
            retarget_diagnostics["orientation_warning_threshold_rad"], dtype=float
        ),
        "position_error_mean_m": np.asarray(retarget_diagnostics["position_error_mean_m"], dtype=float),
        "position_error_p95_m": np.asarray(retarget_diagnostics["position_error_p95_m"], dtype=float),
        "position_error_max_m": np.asarray(retarget_diagnostics["position_error_max_m"], dtype=float),
        "position_error_max_frame": np.asarray(retarget_diagnostics["position_error_max_frame"], dtype=int),
        "max_position_error_vector_m": retarget_diagnostics["max_position_error_vector_m"],
        "orientation_error_mean_rad": np.asarray(retarget_diagnostics["orientation_error_mean_rad"], dtype=float),
        "orientation_error_p95_rad": np.asarray(retarget_diagnostics["orientation_error_p95_rad"], dtype=float),
        "orientation_error_max_rad": np.asarray(retarget_diagnostics["orientation_error_max_rad"], dtype=float),
        "frames_over_ik_tolerance": retarget_diagnostics["frames_over_ik_tolerance"],
        "frames_over_reachability_threshold": retarget_diagnostics["frames_over_reachability_threshold"],
        "frames_over_orientation_threshold": retarget_diagnostics["frames_over_orientation_threshold"],
        "source_timestamps_s": source_timestamps_s,
        "replay_timestamps_s": replay_timestamps_s,
        "target_joint_names": np.asarray(target_joint_names, dtype=object),
        "target_output_joint_names": np.asarray(target_output_joint_names, dtype=object),
        "target_action_positions_deg": action_positions_deg,
        "target_action_names": np.asarray(action_names, dtype=object),
        "executed_target_poses": executed_poses,
        "executed_position_errors_m": executed_position_errors,
        "executed_orientation_errors_rad": executed_orientation_errors,
        "frames_beyond_reach": retarget_diagnostics["frames_beyond_reach"],
        "target_max_reach_m": np.asarray(retarget_diagnostics["target_max_reach_m"], dtype=float),
        "joint_jump_frames": retarget_diagnostics["joint_jump_frames"],
        "joints_at_limit_counts": np.asarray(retarget_diagnostics["joints_at_limit_counts"], dtype=object),
        "source_canonical_rotation": source_canonical_rotation,
        "target_canonical_rotation": target_canonical_rotation,
        "source_base_rotation": source_base_rotation,
        "target_base_rotation": target_base_rotation,
        "source_mount_offset": source_mount_offset,
        "target_mount_offset": target_mount_offset,
        "target_mount_shift": target_mount_shift,
        "base_position_shift": target_mount_offset - source_mount_offset,
        "source_tcp_link": np.asarray(args.source_tcp_link, dtype=object),
        "target_tcp_link": np.asarray(args.target_tcp_link, dtype=object),
        "ik_backend": np.asarray("lerobot.model.kinematics.RobotKinematics", dtype=object),
    }
    if gripper_fraction is not None:
        payload["source_gripper_open_fraction"] = gripper_fraction

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retarget a robot trajectory between arms with different DOF. Source joints are "
            "forward-kinematicked into TCP poses, re-expressed in a shared canonical TCP frame "
            "(bases coincident at the origin), then inverse-kinematicked onto the target arm."
        )
    )
    parser.add_argument("--input", required=True, help="Source trajectory file: .npz, .npy, .pkl, or .pickle.")
    parser.add_argument("--output", required=True, help="Output .npz path.")
    parser.add_argument("--source-mode", default="joints", choices=("joints", "poses"))
    parser.add_argument("--input-data-key", required=True, help="Key/path for source joints or source EE poses.")
    parser.add_argument("--input-joint-slice", default="", help="Columns from input-data-key to use as source joints.")
    parser.add_argument("--input-time-key", help="Optional key/path for timestamps.")
    parser.add_argument("--input-dt", type=float, default=0.1, help="Fallback timestep if no timestamps are stored.")
    parser.add_argument("--input-units", default="rad", choices=("rad", "deg"))

    parser.add_argument("--source-urdf", help="Source URDF. Required for --source-mode=joints; optional for poses.")
    parser.add_argument("--source-tcp-link", default="tool0", help="Source TCP/end-effector link.")
    parser.add_argument("--source-joint-names", help="Comma-separated source URDF joint names.")

    parser.add_argument("--target-urdf", required=True, help="Target URDF path.")
    parser.add_argument("--target-tcp-link", required=True, help="Target TCP/end-effector link.")
    parser.add_argument("--target-joint-names", required=True, help="Comma-separated target URDF joint names.")
    parser.add_argument("--target-output-joint-names", help="Comma-separated output/action joint names.")
    parser.add_argument("--target-initial-joints", help="Comma-separated target initial joints.")
    parser.add_argument("--target-initial-units", default="rad", choices=("rad", "deg"))

    parser.add_argument(
        "--no-tcp-align",
        dest="tcp_align",
        action="store_false",
        help="Disable automatic canonical TCP-frame alignment (transfer native TCP orientations verbatim).",
    )
    parser.add_argument(
        "--no-base-align",
        dest="base_align",
        action="store_false",
        help="Disable mount-point alignment (coincide the URDF base_link origins instead of the base platforms).",
    )
    parser.add_argument(
        "--no-base-rotation-align",
        dest="base_rotation_align",
        action="store_false",
        help="Disable canonical base-orientation alignment (assume both URDF bases face the same way).",
    )
    parser.set_defaults(tcp_align=True, base_align=True, base_rotation_align=True)
    parser.add_argument(
        "--source-base-forward",
        help="Override the source base 'forward' direction in its base frame: x,y,z.",
    )
    parser.add_argument(
        "--target-base-forward",
        help="Override the target base 'forward' direction in its base frame: x,y,z.",
    )
    parser.add_argument(
        "--target-mount-shift",
        help="Real-world offset of the target mount from the source mount, x,y,z meters in the target base frame.",
    )
    parser.add_argument(
        "--target-mount-yaw-deg",
        type=float,
        default=0.0,
        help="Real-world yaw of the target arm relative to the source arm, degrees about vertical.",
    )
    parser.add_argument(
        "--source-gripper-axis",
        help="Override the source gripper secondary/jaw-opening axis in its TCP frame: x,y,z.",
    )
    parser.add_argument(
        "--target-gripper-axis",
        help="Override the target gripper secondary/jaw-opening axis in its TCP frame: x,y,z.",
    )
    parser.add_argument(
        "--no-jaw-geometry",
        dest="jaw_geometry",
        action="store_false",
        help="Ignore moving-jaw geometry; always use the wrist-flex axis for gripper alignment.",
    )
    parser.set_defaults(jaw_geometry=True)

    parser.add_argument("--position-weight", type=float, default=1.0)
    parser.add_argument(
        "--orientation-weight",
        type=float,
        default=1.0,
        help="IK orientation weight. Set 0 to match position only when wrist orientation is not transferable.",
    )
    parser.add_argument(
        "--ik-warmup-iterations",
        type=int,
        default=10,
        help="Deprecated and unused: per-frame IK now iterates to convergence.",
    )
    parser.add_argument(
        "--ik-max-iters-per-frame",
        type=int,
        default=40,
        help="Max IK solver steps per frame; iteration stops early once the error converges.",
    )
    parser.add_argument(
        "--ik-restarts",
        type=int,
        default=2,
        help="Random-restart seeds tried when a converged frame still exceeds tolerance.",
    )
    parser.add_argument(
        "--no-auto-seed",
        dest="auto_seed",
        action="store_false",
        help="Disable global multi-start IK on the first frame (use --target-initial-joints only).",
    )
    parser.add_argument(
        "--auto-seed-samples",
        type=int,
        default=60,
        help="Random seeds tried across the joint limits when auto-seeding the first frame.",
    )
    parser.add_argument(
        "--no-auto-mount",
        dest="auto_mount",
        action="store_false",
        help="Disable automatic mount placement; use --target-mount-shift / --target-mount-yaw-deg verbatim.",
    )
    parser.set_defaults(auto_seed=True, auto_mount=True)
    parser.add_argument("--ik-position-tolerance-m", type=float, default=5e-3)
    parser.add_argument(
        "--reachability-warning-threshold-m",
        type=float,
        default=2e-2,
        help=(
            "Position-error threshold used to label the retarget as physically not feasible. "
            "This is intentionally looser than the IK success tolerance."
        ),
    )
    parser.add_argument(
        "--orientation-warning-threshold-rad",
        type=float,
        default=0.2,
        help=(
            "Orientation-error threshold used to label the retarget as not fully feasible "
            "when --orientation-weight is non-zero."
        ),
    )
    parser.add_argument(
        "--fail-on-unreachable",
        action="store_true",
        help="Exit non-zero after saving if the retarget exceeds the configured feasibility thresholds.",
    )
    parser.add_argument(
        "--task-smoothing-window",
        type=int,
        default=5,
        help="Window for smoothing the desired EE path in task space before IK (0/1 disables).",
    )
    parser.add_argument("--task-smoothing-passes", type=int, default=1)
    parser.add_argument("--smoothing-window", type=int, default=7)
    parser.add_argument("--smoothing-passes", type=int, default=2)
    parser.add_argument(
        "--max-joint-jump-deg",
        type=float,
        default=20.0,
        help="Per-frame joint delta above which a configuration jump (branch flip) is reported.",
    )
    parser.add_argument("--playback-speed-scale", type=float, default=0.2)
    parser.add_argument("--max-joint-velocity-deg-s", type=float)

    parser.add_argument("--source-gripper-key", help="Optional source gripper key/path.")
    parser.add_argument("--source-gripper-units", default="auto", choices=("auto", "fraction"))
    parser.add_argument("--source-gripper-open-value", type=float)
    parser.add_argument("--source-gripper-closed-value", type=float)
    parser.add_argument("--target-gripper-name", help="Optional target gripper output name.")
    parser.add_argument("--target-gripper-open-deg", type=float)
    parser.add_argument("--target-gripper-closed-deg", type=float)
    parser.add_argument(
        "--source-gripper-max-aperture-m",
        type=float,
        help="Source gripper full-open jaw width in meters; with the target value, maps physical opening instead of stroke fraction.",
    )
    parser.add_argument(
        "--target-gripper-max-aperture-m",
        type=float,
        help="Target gripper full-open jaw width in meters (see --source-gripper-max-aperture-m).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.ik_position_tolerance_m <= 0.0:
        raise ValueError("--ik-position-tolerance-m must be > 0.")
    if args.reachability_warning_threshold_m <= 0.0:
        raise ValueError("--reachability-warning-threshold-m must be > 0.")
    if args.orientation_warning_threshold_rad <= 0.0:
        raise ValueError("--orientation-warning-threshold-rad must be > 0.")

    payload, source_joints_deg, source_poses, timestamps = load_source(args)

    if source_poses is None:
        if not args.source_urdf:
            raise ValueError("--source-urdf is required when --source-mode=joints.")
        source_kinematics = make_kinematics(args.source_urdf, args.source_tcp_link, args.source_joint_names)
        source_poses = compute_source_poses(source_kinematics, source_joints_deg)

    target_joint_names = parse_names(args.target_joint_names) or []
    target_output_joint_names = parse_names(args.target_output_joint_names) or list(target_joint_names)
    if len(target_output_joint_names) != len(target_joint_names):
        raise ValueError("target-output-joint-names must match target-joint-names length.")

    target_kinematics = make_kinematics(args.target_urdf, args.target_tcp_link, args.target_joint_names)
    target_initial_deg = parse_vector(
        args.target_initial_joints,
        len(target_joint_names),
        args.target_initial_units,
        np.zeros(len(target_joint_names), dtype=float),
    )

    # Analyze each arm's URDF for the shared-frame ingredients:
    #   - canonical TCP rotation: re-expresses TCP orientation in a shared convention so a
    #     roll-pitch-yaw means the same thing on both wrists (see --no-tcp-align).
    #   - mount offset: the first-actuated-joint origin = top-center of the base platform
    #     where the arm is clamped. The two arms are aligned at this mount point, not at
    #     whatever spot each URDF places its base_link origin (see --no-base-align).
    #   - canonical base rotation: forward/left/up convention of the base frame, so two
    #     URDFs that disagree on which way the base faces still transfer into the same
    #     physical space (see --no-base-rotation-align).
    source_gripper_axis = _parse_axis(args.source_gripper_axis, "source-gripper-axis")
    target_gripper_axis = _parse_axis(args.target_gripper_axis, "target-gripper-axis")
    source_base_forward = _parse_axis(args.source_base_forward, "source-base-forward")
    target_base_forward = _parse_axis(args.target_base_forward, "target-base-forward")
    source_canonical_rotation = np.eye(3, dtype=float)
    target_canonical_rotation = np.eye(3, dtype=float)
    source_base_rotation = np.eye(3, dtype=float)
    target_base_rotation = np.eye(3, dtype=float)
    source_mount_offset = np.zeros(3, dtype=float)
    target_mount_offset = np.zeros(3, dtype=float)
    if args.source_urdf:
        source_arm = compute_canonical_tcp(
            args.source_urdf, args.source_tcp_link, source_gripper_axis, args.jaw_geometry, source_base_forward
        )
        if args.tcp_align:
            source_canonical_rotation = source_arm.rotation
        if args.base_align:
            source_mount_offset = source_arm.mount_offset_base
        if args.base_rotation_align:
            source_base_rotation = source_arm.base_rotation
    target_arm = compute_canonical_tcp(
        args.target_urdf, args.target_tcp_link, target_gripper_axis, args.jaw_geometry, target_base_forward
    )
    if args.tcp_align:
        target_canonical_rotation = target_arm.rotation
    if args.base_align:
        target_mount_offset = target_arm.mount_offset_base
    if args.base_rotation_align:
        target_base_rotation = target_arm.base_rotation

    target_mount_shift = (
        np.zeros(3, dtype=float)
        if not args.target_mount_shift
        else np.array([float(v) for v in args.target_mount_shift.split(",")], dtype=float)
    )
    if target_mount_shift.shape != (3,):
        raise ValueError("--target-mount-shift must be x,y,z in meters.")
    # The flags describe where the target arm physically sits relative to where the source
    # arm was; the task is fixed in the world, so it moves the *opposite* way in the
    # target arm's own frame.
    yaw = -np.deg2rad(float(args.target_mount_yaw_deg))
    mount_yaw_rotation = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    # Express each source TCP pose in the shared frame (mount point at the origin,
    # canonical base orientation, canonical TCP orientation), then re-express it on
    # the target arm; --target-mount-shift/--target-mount-yaw-deg describe a target
    # arm that is physically mounted offset from where the source arm was.
    canonical_poses = np.tile(np.eye(4, dtype=float), (len(source_poses), 1, 1))
    canonical_poses[:, :3, :3] = np.einsum(
        "ij,tjk,kl->til", source_base_rotation.T, source_poses[:, :3, :3], source_canonical_rotation
    )
    canonical_poses[:, :3, 3] = (source_poses[:, :3, 3] - source_mount_offset) @ source_base_rotation

    placed_rotation = target_base_rotation @ mount_yaw_rotation
    desired_poses = np.tile(np.eye(4, dtype=float), (len(source_poses), 1, 1))
    desired_poses[:, :3, :3] = np.einsum(
        "ij,tjk,kl->til", placed_rotation, canonical_poses[:, :3, :3], target_canonical_rotation.T
    )
    # Positions at zero mount shift first; the shift (manual or auto) is applied below.
    desired_poses[:, :3, 3] = canonical_poses[:, :3, 3] @ placed_rotation.T + target_mount_offset

    # Smooth the *desired EE path* in task space before IK, so the trajectory the
    # diagnostics validate is the trajectory that actually gets executed.
    if int(args.task_smoothing_window) > 1 and int(args.task_smoothing_passes) > 0:
        desired_poses = smooth_pose_trajectory(
            desired_poses, int(args.task_smoothing_window), int(args.task_smoothing_passes)
        )

    target_limits = read_joint_limits(args.target_urdf, target_joint_names)
    joint_limits_deg = [
        (float(np.rad2deg(target_limits[name][0])), float(np.rad2deg(target_limits[name][1])))
        for name in target_joint_names
    ]

    # Choose the mount placement: an explicit --target-mount-shift wins; otherwise auto-mount
    # keeps the shared mount when the task is reachable and bolts the arm back/up only when
    # it isn't (so the per-arm pedestal tuning is derived, not hand-set).
    if not args.target_mount_shift and args.auto_mount:
        target_mount_shift = auto_mount_shift(
            target_kinematics,
            desired_poses[:, :3, :3],
            desired_poses[:, :3, 3],
            target_mount_offset,
            target_arm.max_reach_m,
            joint_limits_deg,
            args,
        )
        print(f"Auto-mount shift: {np.round(target_mount_shift, 3).tolist()}m (base frame)")
    desired_poses[:, :3, 3] = desired_poses[:, :3, 3] - target_mount_shift

    # Cheap analytic pre-check: a desired point farther from the target mount than the sum
    # of its link lengths is unreachable no matter what IK does.
    reach_distances = np.linalg.norm(desired_poses[:, :3, 3] - target_mount_offset, axis=1)
    frames_beyond_reach = np.flatnonzero(reach_distances > target_arm.max_reach_m).astype(int)

    raw_target_deg, achieved_poses, position_errors, orientation_errors = solve_with_lerobot_ik(
        target_kinematics=target_kinematics,
        desired_poses=desired_poses,
        initial_joints_deg=target_initial_deg,
        args=args,
        joint_limits_deg=joint_limits_deg,
    )

    # Configuration jumps (e.g. elbow-branch flips) between consecutive IK solutions:
    # smoothing would average straight through them, sweeping the arm across space.
    if len(raw_target_deg) > 1:
        joint_deltas = np.abs(np.diff(raw_target_deg, axis=0)).max(axis=1)
        joint_jump_frames = (np.flatnonzero(joint_deltas > float(args.max_joint_jump_deg)) + 1).astype(int)
    else:
        joint_jump_frames = np.empty(0, dtype=int)

    # Joints sitting at their URDF limits explain *why* a pose could not be reached.
    limit_tolerance_rad = np.deg2rad(0.5)
    raw_target_rad = np.deg2rad(raw_target_deg)
    joints_at_limit_counts: dict[str, int] = {}
    for joint_index, joint_name in enumerate(target_joint_names):
        lower, upper = target_limits[joint_name]
        saturated = (raw_target_rad[:, joint_index] <= lower + limit_tolerance_rad) | (
            raw_target_rad[:, joint_index] >= upper - limit_tolerance_rad
        )
        joints_at_limit_counts[joint_name] = int(np.count_nonzero(saturated))

    smoothed_target_deg = smooth_joint_trajectory(raw_target_deg, args.smoothing_window, args.smoothing_passes)

    # Validate what will actually be replayed: FK of the smoothed joints, not the raw IK
    # solutions, is what the robot executes.
    executed_poses = np.asarray(
        [target_kinematics.forward_kinematics(joint_frame) for joint_frame in smoothed_target_deg], dtype=float
    )
    executed_errors = [pose_error(executed, desired) for executed, desired in zip(executed_poses, desired_poses)]
    executed_position_errors = np.asarray([err[0] for err in executed_errors], dtype=float)
    executed_orientation_errors = np.asarray([err[1] for err in executed_errors], dtype=float)

    retarget_diagnostics = build_retarget_diagnostics(
        position_errors_m=executed_position_errors,
        orientation_errors_rad=executed_orientation_errors,
        desired_poses=desired_poses,
        achieved_poses=executed_poses,
        args=args,
        frames_beyond_reach=frames_beyond_reach,
        target_max_reach_m=target_arm.max_reach_m,
        joint_jump_frames=joint_jump_frames,
        joints_at_limit_counts=joints_at_limit_counts,
    )
    replay_timestamps = make_replay_timestamps(
        joint_positions=smoothed_target_deg,
        source_timestamps_s=timestamps,
        playback_speed_scale=args.playback_speed_scale,
        max_joint_velocity_deg_s=args.max_joint_velocity_deg_s,
    )
    gripper_fraction = extract_gripper_fraction(payload, args.source_gripper_key, args, len(source_poses))

    save_result(
        path=Path(args.output).expanduser().resolve(),
        args=args,
        source_joints_deg=source_joints_deg,
        source_poses=source_poses,
        canonical_poses=canonical_poses,
        desired_poses=desired_poses,
        raw_target_joints_deg=raw_target_deg,
        smoothed_target_joints_deg=smoothed_target_deg,
        achieved_poses=achieved_poses,
        position_errors_m=position_errors,
        orientation_errors_rad=orientation_errors,
        replay_timestamps_s=replay_timestamps,
        target_joint_names=target_joint_names,
        target_output_joint_names=target_output_joint_names,
        gripper_fraction=gripper_fraction,
        source_timestamps_s=timestamps,
        source_canonical_rotation=source_canonical_rotation,
        target_canonical_rotation=target_canonical_rotation,
        source_mount_offset=source_mount_offset,
        target_mount_offset=target_mount_offset,
        retarget_diagnostics=retarget_diagnostics,
        executed_poses=executed_poses,
        executed_position_errors=executed_position_errors,
        executed_orientation_errors=executed_orientation_errors,
        source_base_rotation=source_base_rotation,
        target_base_rotation=target_base_rotation,
        target_mount_shift=target_mount_shift,
    )

    print(f"Saved retargeted trajectory to {args.output}")
    print(f"Frames: {len(source_poses)}")
    print(f"TCP alignment: {'canonical' if args.tcp_align else 'off (verbatim orientation)'}")
    base_shift = target_mount_offset - source_mount_offset
    print(
        f"Base alignment: {'mount-point' if args.base_align else 'off (base_link origins)'}"
        f" shift={np.round(base_shift, 4).tolist()}m"
    )
    print(
        "Base-rotation alignment: "
        + (
            f"canonical (source forward={np.round(source_base_rotation[:, 0], 3).tolist()},"
            f" target forward={np.round(target_base_rotation[:, 0], 3).tolist()})"
            if args.base_rotation_align
            else "off (URDF base axes assumed to agree)"
        )
    )
    print("IK backend: lerobot.model.kinematics.RobotKinematics (iterated to convergence)")
    print(f"Target joints: {target_joint_names}")
    print(f"Target max reach: {target_arm.max_reach_m:.3f}m; frames beyond reach: {len(frames_beyond_reach)}")
    print(f"IK success rate (raw): {float(np.mean(position_errors <= args.ik_position_tolerance_m)):.1%}")
    if len(joint_jump_frames):
        print(
            f"WARNING: {len(joint_jump_frames)} configuration jumps >{args.max_joint_jump_deg:.0f}deg/frame "
            f"at frames {joint_jump_frames[:10].tolist()}{'...' if len(joint_jump_frames) > 10 else ''}"
        )
    saturated_joints = {name: count for name, count in joints_at_limit_counts.items() if count > 0}
    if saturated_joints:
        print(f"Joints at URDF limits (frame counts): {saturated_joints}")
    feasible_label = "OK" if retarget_diagnostics["retarget_feasible"] else "NOT FEASIBLE"
    max_error_vector_cm = np.round(retarget_diagnostics["max_position_error_vector_m"] * 100.0, 3).tolist()
    print(f"Retarget feasibility: {feasible_label} - {retarget_diagnostics['retarget_feasibility_reason']}")
    print(
        "Position diagnostics:"
        f" threshold={float(args.reachability_warning_threshold_m):.4f}m"
        f" frames_over={len(retarget_diagnostics['frames_over_reachability_threshold'])}/{len(source_poses)}"
        f" worst_frame={int(retarget_diagnostics['position_error_max_frame'])}"
        f" worst_error_vector_cm={max_error_vector_cm}"
    )
    print(
        "Orientation diagnostics:"
        f" threshold={float(args.orientation_warning_threshold_rad):.4f}rad"
        f" frames_over={len(retarget_diagnostics['frames_over_orientation_threshold'])}/{len(source_poses)}"
        f" check={'on' if float(args.orientation_weight) > 0.0 else 'off (orientation-weight=0)'}"
    )
    print(
        "Executed pose error (FK of smoothed joints vs desired):"
        f" mean_pos={float(retarget_diagnostics['position_error_mean_m']):.4f}m"
        f" p95_pos={float(retarget_diagnostics['position_error_p95_m']):.4f}m"
        f" max_pos={float(retarget_diagnostics['position_error_max_m']):.4f}m"
        f" mean_rot={float(retarget_diagnostics['orientation_error_mean_rad']):.4f}rad"
        f" p95_rot={float(retarget_diagnostics['orientation_error_p95_rad']):.4f}rad"
        f" max_rot={float(retarget_diagnostics['orientation_error_max_rad']):.4f}rad"
    )
    print(f"Replay duration: {float(replay_timestamps[-1]):.2f}s")
    if args.fail_on_unreachable and not retarget_diagnostics["retarget_feasible"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
