#!/usr/bin/env python3
"""Synthesize source demo trajectories on the SO101 arm for cross-arm validation.

Two tasks are generated as task-space EE paths and solved onto SO101 joints with
converged IK, so the saved "demos" are genuine SO101 joint trajectories:
  - pick_place: hover -> descend -> grasp -> lift -> transfer -> place -> retreat
  - wipe_circle: approach, then two horizontal circles (a wiping motion)

Outputs .npz files with joint_positions_rad (T,5), gripper_fraction (T,),
timestamps_s (T,), plus the desired EE poses for reference.
"""

from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from lerobot.model.kinematics import RobotKinematics

from robot_trajectory_retargeting.frames import pose_error, smooth_pose_trajectory
from robot_trajectory_retargeting.urdf_chain import compute_canonical_tcp

SO101_URDF = "/home/manav/rebot_lerobot/third_party/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
SO101_TCP = "gripper_frame_link"
SO101_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
RATE_HZ = 30.0
OUT_DIR = Path(__file__).resolve().parent / "outputs"


def canonical_orientation(position: np.ndarray, mount: np.ndarray, pitch_down_deg: float) -> np.ndarray:
    """Canonical TCP orientation: approach tilted down in the vertical plane through the
    shoulder (radial), jaw axis horizontal-tangential. This keeps the orientation inside
    what a 5-DOF arm without wrist yaw can actually reach."""
    radial = position - mount
    radial[2] = 0.0
    radial = radial / np.linalg.norm(radial)
    pitch = np.deg2rad(pitch_down_deg)
    z_axis = np.cos(pitch) * radial + np.sin(pitch) * np.array([0.0, 0.0, -1.0])
    y_axis = np.cross([0.0, 0.0, 1.0], radial)
    y_axis = y_axis / np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def interp_path(waypoints: list[tuple[float, np.ndarray, float]], rate_hz: float):
    """Piecewise-linear position + pitch interp at rate_hz. waypoints: (t, xyz, pitch_deg)."""
    times = np.array([w[0] for w in waypoints])
    points = np.array([w[1] for w in waypoints], dtype=float)
    pitches = np.array([w[2] for w in waypoints], dtype=float)
    t_grid = np.arange(times[0], times[-1] + 1e-9, 1.0 / rate_hz)
    xyz = np.column_stack([np.interp(t_grid, times, points[:, k]) for k in range(3)])
    pitch = np.interp(t_grid, times, pitches)
    return t_grid, xyz, pitch


def solve_so101(desired_poses: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kin = RobotKinematics(urdf_path=SO101_URDF, target_frame_name=SO101_TCP, joint_names=SO101_JOINTS)
    joints = np.zeros(len(SO101_JOINTS), dtype=float)
    out, pos_errors, rot_errors = [], [], []
    for desired in desired_poses:
        best = None
        previous = None
        for _ in range(60):
            joints = kin.inverse_kinematics(
                current_joint_pos=joints, desired_ee_pose=desired,
                position_weight=1.0, orientation_weight=0.5,
            )
            achieved = kin.forward_kinematics(joints)
            p_err, r_err = pose_error(achieved, desired)
            cost = p_err + 0.05 * r_err
            if best is None or cost < best[0]:
                best = (cost, joints.copy(), p_err, r_err)
            if previous is not None and abs(previous - cost) < 1e-8:
                break
            previous = cost
        joints = best[1]
        out.append(best[1])
        pos_errors.append(best[2])
        rot_errors.append(best[3])
    return np.asarray(out), np.asarray(pos_errors), np.asarray(rot_errors)


def build_task(name: str, waypoints, gripper_events) -> None:
    arm = compute_canonical_tcp(SO101_URDF, SO101_TCP)
    mount = arm.mount_offset_base
    t_grid, xyz, pitch = interp_path(waypoints, RATE_HZ)

    poses = np.tile(np.eye(4), (len(t_grid), 1, 1))
    for i in range(len(t_grid)):
        canonical = canonical_orientation(xyz[i], mount, pitch[i])
        poses[i, :3, :3] = canonical @ arm.rotation.T  # native TCP orientation
        poses[i, :3, 3] = xyz[i]
    poses = smooth_pose_trajectory(poses, window_size=9, passes=1)

    joints, pos_errors, rot_errors = solve_so101(poses)

    gripper = np.ones(len(t_grid), dtype=float)
    for t_start, t_end, value_start, value_end in gripper_events:
        mask = (t_grid >= t_start) & (t_grid <= t_end)
        span = max(t_end - t_start, 1e-9)
        gripper[mask] = value_start + (t_grid[mask] - t_start) / span * (value_end - value_start)
        gripper[t_grid > t_end] = value_end

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_DIR / f"task_{name}_so101.npz",
        joint_positions_rad=np.deg2rad(joints),
        gripper_fraction=gripper,
        timestamps_s=t_grid,
        desired_ee_poses=poses,
    )
    print(
        f"{name}: {len(t_grid)} frames @ {RATE_HZ:.0f}Hz | SO101 demo fidelity: "
        f"pos mean={pos_errors.mean()*1000:.2f}mm max={pos_errors.max()*1000:.2f}mm | "
        f"rot mean={np.rad2deg(rot_errors.mean()):.2f}deg max={np.rad2deg(rot_errors.max()):.2f}deg"
    )


def main() -> None:
    # --- Task A: pick and place (positions in SO101 base frame, meters) ---
    pick_place = [
        (0.0, np.array([0.20, 0.00, 0.22]), 50.0),
        (2.0, np.array([0.30, 0.10, 0.16]), 55.0),
        (3.5, np.array([0.30, 0.10, 0.065]), 65.0),
        (4.3, np.array([0.30, 0.10, 0.065]), 65.0),   # dwell: close gripper
        (5.5, np.array([0.30, 0.10, 0.20]), 50.0),
        (7.0, np.array([0.26, -0.05, 0.22]), 45.0),
        (8.5, np.array([0.30, -0.12, 0.16]), 55.0),
        (10.0, np.array([0.30, -0.12, 0.065]), 65.0),
        (10.8, np.array([0.30, -0.12, 0.065]), 65.0),  # dwell: open gripper
        (12.5, np.array([0.24, -0.06, 0.22]), 50.0),
    ]
    gripper_a = [(3.6, 4.2, 1.0, 0.15), (10.1, 10.7, 0.15, 1.0)]
    build_task("pick_place", pick_place, gripper_a)

    # --- Task B: wipe two horizontal circles ---
    center = np.array([0.27, 0.0, 0.10])
    radius = 0.06
    waypoints = [(0.0, np.array([0.20, 0.00, 0.20]), 50.0), (2.0, center + [radius, 0, 0], 50.0)]
    n_seg = 48
    for k in range(1, n_seg + 1):
        angle = 2.0 * np.pi * 2.0 * k / n_seg  # two revolutions
        waypoints.append(
            (2.0 + 8.0 * k / n_seg, center + radius * np.array([np.cos(angle), np.sin(angle), 0.0]), 50.0)
        )
    waypoints.append((11.5, np.array([0.20, 0.00, 0.20]), 50.0))
    build_task("wipe_circle", waypoints, [])


if __name__ == "__main__":
    main()
