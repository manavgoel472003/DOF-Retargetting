#!/usr/bin/env python3
"""Render the MuJoCo validation: 2x2 GIF per task (SO101 source + 3 retargeted arms
replaying in sync, with the shared source EE path drawn as a dotted trace in every
panel) and matplotlib summary plots of world-space EE paths and errors."""

from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco

from mujoco_validate import ARMS, TARGET_ARMS, TASKS, MujocoArm

OUT = REPO / "validation" / "outputs"
PANEL_W, PANEL_H = 400, 300
FRAME_STEP = 3
GIF_FPS = 10


def render_arm_frames(arm: MujocoArm, joints_rad: np.ndarray, gripper: np.ndarray | None,
                      trace_world: np.ndarray) -> list[np.ndarray]:
    renderer = mujoco.Renderer(arm.model, PANEL_H, PANEL_W)
    camera = mujoco.MjvCamera()
    # Scenes are in each arm's base frame; look at the task workspace (canonical coords
    # ~= base coords for these arms, shifted by the mount offset).
    lookat_world = np.array([0.24, 0.0, 0.10])
    camera.lookat[:] = arm.world_rotation.T @ lookat_world + arm.mount_offset
    camera.distance = 1.05
    camera.azimuth = 160
    camera.elevation = -25

    # Source EE path expressed in this arm's base frame.
    trace_base = (arm.world_rotation.T @ trace_world.T).T + arm.mount_offset
    trace_idx = np.linspace(0, len(trace_base) - 1, 60).astype(int)

    frames = []
    for k in range(0, len(joints_rad), FRAME_STEP):
        arm.data.qpos[:] = 0.0
        for adr, value in zip(arm.joint_qpos, joints_rad[k]):
            arm.data.qpos[adr] = value
        if gripper is not None:
            arm.set_gripper(float(gripper[k]))
        mujoco.mj_forward(arm.model, arm.data)
        renderer.update_scene(arm.data, camera)
        scene = renderer._scene
        for point in trace_base[trace_idx]:
            if scene.ngeom >= scene.maxgeom:
                break
            g = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.004, 0, 0]),
                point.astype(np.float64), np.eye(3).flatten(),
                np.array([0.1, 0.9, 0.9, 0.8], dtype=np.float32),
            )
            scene.ngeom += 1
        frames.append(renderer.render().copy())
    renderer.close()
    return frames


def label(img: np.ndarray, name: str, color: tuple) -> np.ndarray:
    img = img.copy()
    img[:6, :, :] = color  # colored top border identifies the arm
    return img


def main() -> None:
    colors = {
        "so101": (255, 255, 255),
        "b601": (255, 80, 80),
        "piper": (80, 255, 80),
        "yam": (80, 160, 255),
        "panda": (255, 200, 60),
        "kuka": (220, 100, 255),
    }
    arm_order = ["so101"] + TARGET_ARMS
    for task in TASKS:
        arms = {name: MujocoArm(name, spec) for name, spec in ARMS.items()}
        source = np.load(OUT / f"task_{task}_so101.npz")
        source_world = np.load(OUT / f"mujoco_world_{task}_so101.npz")["world_positions"]
        gripper = source["gripper_fraction"]

        panels = {}
        panels["so101"] = render_arm_frames(arms["so101"], source["joint_positions_rad"], gripper, source_world)
        for target in TARGET_ARMS:
            data = np.load(OUT / f"retarget_{task}_{target}.npz", allow_pickle=True)
            joints = data["smoothed_target_joint_positions_rad"]
            arms[target].set_mount_shift(np.asarray(data["target_mount_shift"], dtype=float))  # auto-derived
            if not bool(data["retarget_feasible"]):
                # Infeasible -> hold the home pose instead of replaying a flailing path.
                home = np.asarray(ARMS[target].get("initial_joints", joints[0]), dtype=float)
                joints = np.repeat(home[None, :], len(joints), axis=0)
            panels[target] = render_arm_frames(arms[target], joints, gripper, source_world)

        n = min(len(f) for f in panels.values())
        columns = 3
        tiles = []
        for i in range(n):
            rows = []
            for start in range(0, len(arm_order), columns):
                rows.append(np.hstack([
                    label(panels[a][i], a, colors[a]) for a in arm_order[start:start + columns]
                ]))
            tiles.append(np.vstack(rows))
        gif_path = OUT / f"validation_{task}.gif"
        imageio.mimsave(gif_path, tiles, fps=GIF_FPS, loop=0)
        print(f"wrote {gif_path} ({n} frames)")

        # --- plots ---
        fig = plt.figure(figsize=(13, 5))
        ax3d = fig.add_subplot(1, 2, 1, projection="3d")
        ax3d.plot(*source_world.T, color="black", lw=2.5, label="SO101 source")
        plot_colors = {"b601": "tab:red", "piper": "tab:green", "yam": "tab:blue",
                       "panda": "tab:orange", "kuka": "tab:purple"}
        ax_err = fig.add_subplot(1, 2, 2)
        t = source["timestamps_s"]
        for target, c in plot_colors.items():
            w = np.load(OUT / f"mujoco_world_{task}_{target}.npz")
            ax3d.plot(*w["world_positions"].T, color=c, lw=1.2, ls="--", label=target)
            ax_err.plot(t, w["pos_err_m"] * 1000, color=c, lw=1.2, label=target)
        ax3d.set_title(f"{task}: EE world paths (canonical mounting)")
        ax3d.set_xlabel("x [m]"); ax3d.set_ylabel("y [m]"); ax3d.set_zlabel("z [m]")
        ax3d.legend()
        try:
            ax3d.set_aspect("equal")
        except NotImplementedError:
            pass
        ax_err.set_title("EE position error vs SO101 source")
        ax_err.set_xlabel("time [s]"); ax_err.set_ylabel("error [mm]")
        ax_err.grid(alpha=0.3); ax_err.legend()
        fig.tight_layout()
        png_path = OUT / f"validation_{task}.png"
        fig.savefig(png_path, dpi=110)
        plt.close(fig)
        print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
