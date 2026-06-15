#!/usr/bin/env python3
"""Build a task only a 6-DOF wrist can execute: keep the gripper orientation constant in
the world while sweeping sideways. A 5-DOF arm (SO101, no wrist yaw) must rotate its
approach with the pan joint, so it cannot track this; the retargeter must call it
NOT FEASIBLE on orientation."""

from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from lerobot.model.kinematics import RobotKinematics

from robot_trajectory_retargeting.frames import pose_error
from robot_trajectory_retargeting.urdf_chain import compute_canonical_tcp

B601_URDF = str(REPO / "examples/assets/rebot_b601/reBot-DevArm_fixend_description/urdf/reBot-DevArm_fixend.urdf")
B601_JOINTS = ["joint1", "joint2", "join3", "joint4", "joint5", "joint6"]
OUT = REPO / "validation" / "outputs"


def main() -> None:
    arm = compute_canonical_tcp(B601_URDF, "gripper_tcp")
    mount = arm.mount_offset_base

    # Constant canonical orientation: approach 45 deg down, facing +x, jaw axis +y.
    pitch = np.deg2rad(45.0)
    z_axis = np.array([np.cos(pitch), 0.0, -np.sin(pitch)])
    y_axis = np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(y_axis, z_axis)
    canonical = np.column_stack([x_axis, y_axis, z_axis])
    native = canonical @ arm.rotation.T

    t_grid = np.arange(0.0, 8.0, 1.0 / 30.0)
    sweep = 0.16 * np.sin(2.0 * np.pi * t_grid / 8.0)  # y: 0 -> +16cm -> -16cm -> 0
    poses = np.tile(np.eye(4), (len(t_grid), 1, 1))
    poses[:, :3, :3] = native
    poses[:, :3, 3] = mount + np.column_stack(
        [np.full_like(sweep, 0.28), sweep, np.full_like(sweep, 0.10)]
    )

    kin = RobotKinematics(urdf_path=B601_URDF, target_frame_name="gripper_tcp", joint_names=B601_JOINTS)
    joints, errors = [], []
    q = np.zeros(6)
    for desired in poses:
        best = None
        prev = None
        for _ in range(80):
            q = kin.inverse_kinematics(current_joint_pos=q, desired_ee_pose=desired,
                                       position_weight=1.0, orientation_weight=1.0)
            achieved = kin.forward_kinematics(q)
            p_err, r_err = pose_error(achieved, desired)
            cost = p_err + 0.1 * r_err
            if best is None or cost < best[0]:
                best = (cost, q.copy(), p_err, r_err)
            if prev is not None and abs(prev - cost) < 1e-8:
                break
            prev = cost
        q = best[1]
        joints.append(best[1])
        errors.append((best[2], best[3]))
    joints = np.asarray(joints)
    errors = np.asarray(errors)
    np.savez(
        OUT / "task_fixed_orientation_sweep_b601.npz",
        joint_positions_rad=np.deg2rad(joints),
        timestamps_s=t_grid,
    )
    print(
        f"fixed_orientation_sweep on B601: {len(t_grid)} frames | "
        f"pos max={errors[:,0].max()*1000:.2f}mm rot max={np.rad2deg(errors[:,1].max()):.2f}deg"
    )


if __name__ == "__main__":
    main()
