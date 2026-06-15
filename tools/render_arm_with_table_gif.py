#!/usr/bin/env python3
"""Render an arm trajectory as a stick-figure skeleton descending onto a table plane.

The table plane is drawn at the lowest TCP height reached, so the motion reads the
way it does in the real world (gripper coming down to the table) instead of the
arm appearing to dive below the base-frame z=0 plane.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio.v2 as imageio

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from robot_trajectory_retargeting.urdf_chain import _read_joints, _chain_to_link, _origin_transform  # noqa: E402

_REVOLUTE = {"revolute", "continuous"}


def _rot_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    a = axis / (np.linalg.norm(axis) + 1e-12)
    c, s = math.cos(angle), math.sin(angle)
    x, y, z = a
    R = np.array([
        [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    return T


def skeleton_points(chain, q_by_name: dict[str, float]) -> np.ndarray:
    """Return Nx3 link-origin positions from base to TCP for one configuration."""
    T = np.eye(4)
    pts = [T[:3, 3].copy()]
    for j in chain:
        T = T @ _origin_transform(j)
        pts.append(T[:3, 3].copy())
        if j.joint_type in _REVOLUTE:
            T = T @ _rot_axis(j.axis, float(q_by_name.get(j.name, 0.0)))
    return np.asarray(pts, dtype=float)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--trajectory", required=True)
    p.add_argument("--urdf", required=True)
    p.add_argument("--tcp-link", required=True)
    p.add_argument("--joint-data-key", required=True, help="Nx J array of joint angles (radians).")
    p.add_argument("--joint-names", required=True, help="Comma-separated URDF joint names matching the columns.")
    p.add_argument("--output", required=True)
    p.add_argument("--title", default="")
    p.add_argument("--stride", type=int, default=6)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--highlight-range", default="", help="Optional a,b frame range to mark (e.g. unreachable window).")
    p.add_argument("--table-z", type=float, default=None,
                   help="Table height in base frame (m). Default: lowest TCP reached. Set this to the true "
                        "physical table when the arm cannot reach it (e.g. the desired-pose minimum).")
    args = p.parse_args()

    d = np.load(args.trajectory, allow_pickle=True)
    q = np.asarray(d[args.joint_data_key], dtype=float)
    names = [s.strip() for s in args.joint_names.split(",")]
    if q.shape[1] != len(names):
        raise ValueError(f"{q.shape[1]} columns vs {len(names)} joint names")

    chain = _chain_to_link(_read_joints(Path(args.urdf)), args.tcp_link)
    all_sk = [skeleton_points(chain, dict(zip(names, q[i]))) for i in range(len(q))]
    all_sk = np.asarray(all_sk)            # (T, N, 3)
    tcp = all_sk[:, -1, :]                 # (T, 3)
    table_z = args.table_z if args.table_z is not None else float(tcp[:, 2].min())

    hl = None
    if args.highlight_range.strip():
        a, b = (int(x) for x in args.highlight_range.split(","))
        hl = (a, b)

    pts = all_sk.reshape(-1, 3)
    mins, maxs = pts.min(0), pts.max(0)
    ctr = 0.5 * (mins + maxs)
    rad = max(0.18, 0.5 * float((maxs - mins).max()))
    # table quad spanning the xy workspace
    mx = 1.3 * rad
    quad = np.array([
        [ctr[0] - mx, ctr[1] - mx, table_z],
        [ctr[0] + mx, ctr[1] - mx, table_z],
        [ctr[0] + mx, ctr[1] + mx, table_z],
        [ctr[0] - mx, ctr[1] + mx, table_z],
    ])

    frames = list(range(0, len(q), max(1, args.stride)))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.parent / f".{out.stem}_frames"
    tmp.mkdir(exist_ok=True)
    paths = []
    for ridx, fi in enumerate(frames):
        fig = plt.figure(figsize=(7.5, 6.5), constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")
        ax.add_collection3d(Poly3DCollection([quad], alpha=0.25, facecolor="#b08968", edgecolor="#7f5539"))
        ax.text(ctr[0], ctr[1] + mx, table_z, "  table", color="#7f5539", fontsize=9)

        sk = all_sk[fi]
        ax.plot(sk[:, 0], sk[:, 1], sk[:, 2], "-o", color="#1f4e79", lw=3, ms=4, label="arm")
        ax.plot(tcp[: fi + 1, 0], tcp[: fi + 1, 1], tcp[: fi + 1, 2], color="#2b8a3e", lw=1.6, label="tcp path")
        in_hl = hl is not None and hl[0] <= fi <= hl[1]
        ax.scatter([tcp[fi, 0]], [tcp[fi, 1]], [tcp[fi, 2]],
                   color="#e8590c" if in_hl else "#0b7285", s=55,
                   label="wrist-limited" if in_hl else "tcp")

        ax.set_xlim(ctr[0] - rad, ctr[0] + rad)
        ax.set_ylim(ctr[1] - rad, ctr[1] + rad)
        ax.set_zlim(min(table_z, ctr[2] - rad), ctr[2] + rad)
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
        gap = (tcp[fi, 2] - table_z) * 100.0
        ax.set_title(f"{args.title}\nframe {fi}  tcp z={tcp[fi,2]*100:.1f}cm  height above table={gap:.1f}cm",
                     fontsize=10)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m] (base frame)")
        ax.view_init(elev=18, azim=-60)
        ax.legend(loc="upper left", fontsize=8)
        fp = tmp / f"f{ridx:04d}.png"
        fig.savefig(fp, dpi=110); plt.close(fig); paths.append(fp)

    imageio.mimsave(out, [imageio.imread(fp) for fp in paths], duration=1.0 / max(1, args.fps), loop=0)
    print(f"table_z (base frame): {table_z*100:.1f} cm")
    print(f"saved {out}  ({len(paths)} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
