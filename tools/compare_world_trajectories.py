#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(v) for v in rpy]
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    rot_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    rot_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return rot_z @ rot_y @ rot_x


def pose_from_xyz_rpy(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = _rpy_matrix(np.asarray(rpy, dtype=float))
    pose[:3, 3] = np.asarray(xyz, dtype=float)
    return pose


def rotation_matrix_to_rotvec(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    cos_angle = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    angle = math.acos(cos_angle)
    if angle < 1e-9 or abs(math.sin(angle)) < 1e-9:
        return np.zeros(3, dtype=float)
    skew = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=float,
    )
    return angle * skew / (2.0 * math.sin(angle))


def parse_xyz_rpy(text: str) -> np.ndarray:
    values = np.array([float(part.strip()) for part in text.split(",") if part.strip()], dtype=float)
    if values.shape != (6,):
        raise ValueError("Expected 6 comma-separated values: x,y,z,roll,pitch,yaw")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare source and target trajectories in a shared world frame.")
    parser.add_argument("--retargeted-npz", required=True, help="Retargeted trajectory .npz file.")
    parser.add_argument(
        "--source-base-world-xyz-rpy",
        default="0,0,0,0,0,0",
        help="World pose of the source base frame: x,y,z,roll,pitch,yaw",
    )
    parser.add_argument(
        "--target-base-world-xyz-rpy",
        default="0,0,0,0,0,0",
        help="World pose of the target base frame: x,y,z,roll,pitch,yaw",
    )
    parser.add_argument("--output-json", help="Optional path to save the computed metrics as JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = np.load(args.retargeted_npz, allow_pickle=True)
    source_poses = np.asarray(payload["source_end_effector_poses"], dtype=float)
    achieved_poses = np.asarray(payload["achieved_target_poses"], dtype=float)
    desired_poses = np.asarray(payload["desired_target_poses"], dtype=float)

    source_base_world = pose_from_xyz_rpy(
        parse_xyz_rpy(args.source_base_world_xyz_rpy)[:3],
        parse_xyz_rpy(args.source_base_world_xyz_rpy)[3:],
    )
    target_base_world = pose_from_xyz_rpy(
        parse_xyz_rpy(args.target_base_world_xyz_rpy)[:3],
        parse_xyz_rpy(args.target_base_world_xyz_rpy)[3:],
    )

    source_world = np.asarray([source_base_world @ pose for pose in source_poses], dtype=float)
    desired_world = np.asarray([target_base_world @ pose for pose in desired_poses], dtype=float)
    achieved_world = np.asarray([target_base_world @ pose for pose in achieved_poses], dtype=float)

    source_vs_desired_pos = np.linalg.norm(source_world[:, :3, 3] - desired_world[:, :3, 3], axis=1)
    source_vs_achieved_pos = np.linalg.norm(source_world[:, :3, 3] - achieved_world[:, :3, 3], axis=1)
    desired_vs_achieved_pos = np.linalg.norm(desired_world[:, :3, 3] - achieved_world[:, :3, 3], axis=1)

    source_vs_desired_rot = np.array(
        [np.linalg.norm(rotation_matrix_to_rotvec(desired_world[i, :3, :3] @ source_world[i, :3, :3].T)) for i in range(len(source_world))],
        dtype=float,
    )
    source_vs_achieved_rot = np.array(
        [np.linalg.norm(rotation_matrix_to_rotvec(achieved_world[i, :3, :3] @ source_world[i, :3, :3].T)) for i in range(len(source_world))],
        dtype=float,
    )
    desired_vs_achieved_rot = np.array(
        [np.linalg.norm(rotation_matrix_to_rotvec(achieved_world[i, :3, :3] @ desired_world[i, :3, :3].T)) for i in range(len(source_world))],
        dtype=float,
    )

    result = {
        "retargeted_npz": str(Path(args.retargeted_npz).resolve()),
        "source_base_world_xyz_rpy": [float(v) for v in parse_xyz_rpy(args.source_base_world_xyz_rpy)],
        "target_base_world_xyz_rpy": [float(v) for v in parse_xyz_rpy(args.target_base_world_xyz_rpy)],
        "source_vs_desired": {
            "mean_pos_m": float(source_vs_desired_pos.mean()),
            "max_pos_m": float(source_vs_desired_pos.max()),
            "mean_rot_rad": float(source_vs_desired_rot.mean()),
            "max_rot_rad": float(source_vs_desired_rot.max()),
        },
        "source_vs_achieved": {
            "mean_pos_m": float(source_vs_achieved_pos.mean()),
            "max_pos_m": float(source_vs_achieved_pos.max()),
            "mean_rot_rad": float(source_vs_achieved_rot.mean()),
            "max_rot_rad": float(source_vs_achieved_rot.max()),
        },
        "desired_vs_achieved": {
            "mean_pos_m": float(desired_vs_achieved_pos.mean()),
            "max_pos_m": float(desired_vs_achieved_pos.max()),
            "mean_rot_rad": float(desired_vs_achieved_rot.mean()),
            "max_rot_rad": float(desired_vs_achieved_rot.max()),
        },
        "exact_world_xyz_equal_source_vs_achieved": bool(
            np.allclose(source_world[:, :3, 3], achieved_world[:, :3, 3], atol=0.0, rtol=0.0)
        ),
        "exact_world_pose_equal_source_vs_achieved": bool(
            np.allclose(source_world, achieved_world, atol=0.0, rtol=0.0)
        ),
    }

    print(json.dumps(result, indent=2))
    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
