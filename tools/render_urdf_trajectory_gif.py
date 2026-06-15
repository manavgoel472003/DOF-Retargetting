#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import imageio.v2 as imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lerobot.model.kinematics import RobotKinematics


def _parse_joint_names(spec: str | None) -> list[str] | None:
    if spec is None or spec.strip() == "":
        return None
    return [part.strip() for part in spec.split(",") if part.strip()]


def _parse_joint_slice(spec: str | None, total_columns: int) -> np.ndarray:
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


def _set_equal_3d_axes(ax, points: np.ndarray):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = max(0.15, 0.5 * np.max(maxs - mins))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _draw_box(ax, center_xyz: np.ndarray, size_xyz: np.ndarray, color: str = "#f59f00"):
    center_xyz = np.asarray(center_xyz, dtype=float)
    size_xyz = np.asarray(size_xyz, dtype=float)
    dx, dy, dz = size_xyz / 2.0
    xs = [center_xyz[0] - dx, center_xyz[0] + dx]
    ys = [center_xyz[1] - dy, center_xyz[1] + dy]
    zs = [center_xyz[2] - dz, center_xyz[2] + dz]
    vertices = np.array(
        [
            [xs[0], ys[0], zs[0]],
            [xs[1], ys[0], zs[0]],
            [xs[1], ys[1], zs[0]],
            [xs[0], ys[1], zs[0]],
            [xs[0], ys[0], zs[1]],
            [xs[1], ys[0], zs[1]],
            [xs[1], ys[1], zs[1]],
            [xs[0], ys[1], zs[1]],
        ],
        dtype=float,
    )
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for i0, i1 in edges:
        ax.plot(
            [vertices[i0, 0], vertices[i1, 0]],
            [vertices[i0, 1], vertices[i1, 1]],
            [vertices[i0, 2], vertices[i1, 2]],
            color=color,
            linewidth=1.5,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a URDF trajectory to a GIF.")
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--target-link", required=True)
    parser.add_argument("--joint-data-key", required=True)
    parser.add_argument("--joint-names", required=True, help="Comma-separated URDF joint names to use for FK.")
    parser.add_argument("--joint-slice", default=None, help="Columns in joint-data-key that correspond to the arm chain.")
    parser.add_argument("--joint-units", default="deg", choices=("deg", "rad"))
    parser.add_argument("--time-key", default="timestamps_s")
    parser.add_argument("--desired-poses-key", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="Trajectory replay")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--stride", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trajectory = np.load(args.trajectory, allow_pickle=True)

    q_data_full = np.asarray(trajectory[args.joint_data_key], dtype=float)
    if q_data_full.ndim != 2:
        raise ValueError(f"Expected 2D joint trajectory, got {q_data_full.shape}")
    q_cols = _parse_joint_slice(args.joint_slice, q_data_full.shape[1])
    q_data = q_data_full[:, q_cols]
    if args.joint_units == "rad":
        q_data = np.rad2deg(q_data)

    timestamps = np.asarray(trajectory[args.time_key], dtype=float)
    if len(timestamps) != len(q_data):
        raise ValueError("timestamps and joint trajectory must align")

    desired_poses = np.asarray(trajectory[args.desired_poses_key], dtype=float) if args.desired_poses_key else None
    object_box_xyz = np.asarray(trajectory["object_box_xyz"], dtype=float) if "object_box_xyz" in trajectory else None
    object_box_size = np.asarray(trajectory["object_box_size"], dtype=float) if "object_box_size" in trajectory else None
    object_attach_start_index = (
        int(np.asarray(trajectory["object_attach_start_index"]).reshape(()))
        if "object_attach_start_index" in trajectory
        else None
    )

    kinematics = RobotKinematics(
        urdf_path=str(Path(args.urdf).expanduser().resolve()),
        target_frame_name=args.target_link,
        joint_names=_parse_joint_names(args.joint_names),
    )

    frame_indices = np.arange(0, len(q_data), max(1, args.stride), dtype=int)
    actual_path = np.asarray([kinematics.forward_kinematics(q_data[idx])[:3, 3] for idx in frame_indices], dtype=float)
    desired_path = desired_poses[frame_indices, :3, 3] if desired_poses is not None else None

    all_points = [actual_path]
    if desired_path is not None:
        all_points.append(desired_path)
    if object_box_xyz is not None:
        all_points.append(np.asarray(object_box_xyz, dtype=float).reshape(1, 3))
    all_points = np.concatenate(all_points, axis=0)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = output_path.parent / f".{output_path.stem}_frames"
    temp_dir.mkdir(parents=True, exist_ok=True)

    frame_paths: list[Path] = []
    for render_idx, frame_idx in enumerate(frame_indices):
        fig = plt.figure(figsize=(8.5, 6.0), constrained_layout=True)
        ax = fig.add_subplot(1, 1, 1, projection="3d")

        current_point = actual_path[render_idx]
        ax.scatter(
            [current_point[0]],
            [current_point[1]],
            [current_point[2]],
            color="#0b7285",
            s=45,
            label="current tcp",
        )
        ax.plot(
            actual_path[: render_idx + 1, 0],
            actual_path[: render_idx + 1, 1],
            actual_path[: render_idx + 1, 2],
            color="#2b8a3e",
            linewidth=2,
            label="actual ee",
        )
        if desired_path is not None:
            ax.plot(
                desired_path[: render_idx + 1, 0],
                desired_path[: render_idx + 1, 1],
                desired_path[: render_idx + 1, 2],
                "--",
                color="#d6336c",
                linewidth=2,
                label="desired ee",
            )
        if object_box_xyz is not None and object_box_size is not None:
            object_xyz = object_box_xyz
            if object_attach_start_index is not None and frame_idx >= object_attach_start_index:
                object_xyz = actual_path[render_idx]
            _draw_box(ax, object_xyz, object_box_size)

        _set_equal_3d_axes(ax, all_points)
        ax.set_title(args.title)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.view_init(elev=25, azim=-55)
        ax.legend(loc="upper left")

        frame_path = temp_dir / f"frame_{render_idx:04d}.png"
        fig.savefig(frame_path, dpi=120)
        plt.close(fig)
        frame_paths.append(frame_path)

    images = [imageio.imread(frame_path) for frame_path in frame_paths]
    imageio.mimsave(output_path, images, duration=1.0 / max(1, args.fps), loop=0)
    preview_path = output_path.with_suffix(".png")
    imageio.imwrite(preview_path, images[min(len(images) - 1, max(0, len(images) // 2))])

    print(f"Saved animation to {output_path}")
    print(f"Saved preview to {preview_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
