#!/usr/bin/env python3
"""MuJoCo cross-arm validation of the retargeter.

Each arm's URDF is loaded into MuJoCo (an FK implementation independent of placo),
its base is placed so that the *canonical mount frame* coincides with the world frame
(same physical mounting for every arm: mount point at the origin, canonical
forward/left/up axes = world x/y/z), and the retargeted joint trajectory is replayed
kinematically. If the retargeter is right, every arm's TCP traces the same world-space
path with the same canonical orientation.

Outputs per task: per-arm position/orientation error stats vs the source path,
a 3D path overlay plot, error-over-time plot, and a 2x2 replay GIF.
"""

from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import mujoco

from robot_trajectory_retargeting.frames import invert_pose, rotation_matrix_to_rotvec
from robot_trajectory_retargeting.urdf_chain import (
    _chain_to_link,
    _link_base_transform,
    _read_joints,
    compute_canonical_tcp,
)

OUT = REPO / "validation" / "outputs"
SO101_URDF = "/home/manav/rebot_lerobot/third_party/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"

ARMS = {
    "so101": {
        "urdf": SO101_URDF,
        "tcp": "gripper_frame_link",
        "joints": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
    },
    "b601": {
        "urdf": str(REPO / "examples/assets/rebot_b601/reBot-DevArm_fixend_description/urdf/reBot-DevArm_gripper.urdf"),
        "tcp": "gripper_tcp",
        "joints": ["joint1", "joint2", "join3", "joint4", "joint5", "joint6"],
    },
    "piper": {
        "urdf": str(REPO / "examples/assets/piper/piper_retarget.urdf"),
        "tcp": "gripper_tcp",
        "joints": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
    },
    "yam": {
        "urdf": str(REPO / "examples/assets/yam/yam_retarget.urdf"),
        "tcp": "gripper_tcp",
        "joints": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
    },
    # DEFAULT MOUNTING: each arm is mounted where it can realistically do the task -- the
    # can stays fixed in the world, the *arm* is bolted back (mount_shift) so the desktop
    # task isn't in its cramped inner workspace. Desktop arms need no shift; the big 7-DOF
    # arms are set back. An arm whose retarget is still infeasible is frozen, not flailed.
    # Mounted back AND up (on a pedestal) so they reach DOWN to the low can, keeping their
    # elbows above the desk -- how a big arm is really deployed over a low workspace.
    "panda": {
        "urdf": str(REPO / "examples/assets/panda/panda_retarget.urdf"),
        "tcp": "panda_grasptarget",
        "joints": [f"panda_joint{i}" for i in range(1, 8)],
        "mount_shift": [-0.30, 0.0, 0.25],
        "initial_joints": [-1.80, 1.29, 1.66, -2.71, -2.04, 2.16, 1.94],
    },
    "kuka": {
        "urdf": str(REPO / "examples/assets/kuka_iiwa/iiwa_retarget.urdf"),
        "tcp": "gripper_tcp",
        "joints": [f"lbr_iiwa_joint_{i}" for i in range(1, 8)],
        "mount_shift": [-0.40, 0.0, 0.30],
        "initial_joints": [0.0, 0.6, 0.0, -1.0, 0.0, 0.9, 0.0],
    },
}
TARGET_ARMS = ["b601", "piper", "yam", "panda", "kuka"]
TASKS = ["pick_place", "wipe_circle"]


class MujocoArm:
    def __init__(self, name: str, spec: dict):
        self.name = name
        self.spec = spec
        self.model = mujoco.MjModel.from_xml_path(spec["urdf"])
        self.data = mujoco.MjData(self.model)
        arm = compute_canonical_tcp(spec["urdf"], spec["tcp"])
        self.canonical_rotation = arm.rotation
        # Express MuJoCo's base-frame FK in the arm's canonical mount frame (mount point at
        # origin, forward/left/up = world x/y/z). A per-arm mount_shift bolts the arm back
        # from the shared spot so the fixed-in-the-world task is reachable.
        self.world_rotation = arm.base_rotation.T
        self.mount_base = arm.mount_offset_base
        self.set_mount_shift(np.asarray(spec.get("mount_shift", [0.0, 0.0, 0.0])))
        self.joint_qpos = [self.model.joint(j).qposadr[0] for j in spec["joints"]]

        # MuJoCo's URDF importer fuses fixed-joint links into their parents, so the TCP
        # link may not exist as a body. Walk up the (all-fixed) tail of the chain to the
        # deepest surviving ancestor body and keep the constant ancestor->TCP transform.
        body_names = {self.model.body(i).name for i in range(self.model.nbody)}
        joints = _read_joints(spec["urdf"])
        chain = _chain_to_link(joints, spec["tcp"])
        link = spec["tcp"]
        by_child = {j.child: j for j in chain}
        while link not in body_names:
            if link not in by_child:
                raise ValueError(f"No surviving MuJoCo body found above '{spec['tcp']}'")
            link = by_child[link].parent
        self.tcp_anchor_body = self.model.body(link).id
        anchor_to_base = invert_pose(_link_base_transform(joints, link))
        self.anchor_to_tcp = anchor_to_base @ _link_base_transform(joints, spec["tcp"])

    def set_mount_shift(self, mount_shift: np.ndarray) -> None:
        """The arm is bolted back/up from the shared mount by this base-frame shift."""
        self.mount_offset = self.mount_base - np.asarray(mount_shift, dtype=float)

    def fk(self, joints_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.data.qpos[:] = 0.0
        for adr, value in zip(self.joint_qpos, joints_rad):
            self.data.qpos[adr] = value
        mujoco.mj_kinematics(self.model, self.data)
        anchor = np.eye(4)
        anchor[:3, 3] = self.data.xpos[self.tcp_anchor_body]
        anchor[:3, :3] = self.data.xmat[self.tcp_anchor_body].reshape(3, 3)
        tcp = anchor @ self.anchor_to_tcp
        position = self.world_rotation @ (tcp[:3, 3] - self.mount_offset)
        rotation = self.world_rotation @ tcp[:3, :3]
        return position, rotation

    def set_gripper(self, fraction: float) -> None:
        """Drive each arm's modeled jaw from the shared open-fraction signal."""
        if self.name == "so101":
            jid = self.model.joint("gripper")
            lo, hi = jid.range
            self.data.qpos[jid.qposadr[0]] = lo + fraction * (hi - lo)
        elif self.name == "piper":
            j7, j8 = self.model.joint("joint7"), self.model.joint("joint8")
            self.data.qpos[j7.qposadr[0]] = fraction * j7.range[1]
            self.data.qpos[j8.qposadr[0]] = fraction * j8.range[0]
        elif self.name == "b601":
            left, right = self.model.joint("jaw_left"), self.model.joint("jaw_right")
            self.data.qpos[left.qposadr[0]] = fraction * left.range[1]
            self.data.qpos[right.qposadr[0]] = fraction * right.range[1]
        elif self.name == "yam":
            # URDF range -0.048..0: 0 = tips fully apart (open), -0.048 = closed.
            for joint_name in ("tip1", "tip2"):
                jid = self.model.joint(joint_name)
                self.data.qpos[jid.qposadr[0]] = jid.range[0] * (1.0 - fraction)
        elif self.name == "panda":
            for joint_name in ("panda_finger_joint1", "panda_finger_joint2"):
                jid = self.model.joint(joint_name)
                self.data.qpos[jid.qposadr[0]] = fraction * jid.range[1]
        elif self.name == "kuka":
            for joint_name in ("jaw_left", "jaw_right"):
                jid = self.model.joint(joint_name)
                self.data.qpos[jid.qposadr[0]] = fraction * jid.range[1]


def replay(arm: MujocoArm, joints_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positions, rotations = [], []
    for q in joints_rad:
        p, r = arm.fk(q)
        positions.append(p)
        rotations.append(r @ arm.canonical_rotation)  # canonical TCP orientation in world
    return np.asarray(positions), np.asarray(rotations)


def main() -> None:
    arms = {name: MujocoArm(name, spec) for name, spec in ARMS.items()}
    report: dict[str, dict[str, dict[str, float]]] = {}

    for task in TASKS:
        source = np.load(OUT / f"task_{task}_so101.npz")
        source_positions, source_rotations = replay(arms["so101"], source["joint_positions_rad"])
        report[task] = {}

        for target in TARGET_ARMS:
            data = np.load(OUT / f"retarget_{task}_{target}.npz", allow_pickle=True)
            joints = data["smoothed_target_joint_positions_rad"]
            arms[target].set_mount_shift(np.asarray(data["target_mount_shift"], dtype=float))  # auto-derived
            positions, rotations = replay(arms[target], joints)
            pos_err = np.linalg.norm(positions - source_positions, axis=1)
            rot_err = np.array(
                [
                    np.linalg.norm(rotation_matrix_to_rotvec(rt @ rs.T))
                    for rt, rs in zip(rotations, source_rotations)
                ]
            )
            report[task][target] = {
                "pos_mean_mm": float(pos_err.mean() * 1000),
                "pos_p95_mm": float(np.percentile(pos_err, 95) * 1000),
                "pos_max_mm": float(pos_err.max() * 1000),
                "rot_mean_deg": float(np.rad2deg(rot_err.mean())),
                "rot_p95_deg": float(np.rad2deg(np.percentile(rot_err, 95))),
                "rot_max_deg": float(np.rad2deg(rot_err.max())),
            }
            np.savez(
                OUT / f"mujoco_world_{task}_{target}.npz",
                world_positions=positions,
                source_world_positions=source_positions,
                pos_err_m=pos_err,
                rot_err_rad=rot_err,
            )

        np.savez(OUT / f"mujoco_world_{task}_so101.npz", world_positions=source_positions)

    print("\n=== MuJoCo cross-arm EE path agreement (vs SO101 source, same canonical mounting) ===")
    header = f"{'task':<12} {'target':<7} {'pos mean':>9} {'pos p95':>9} {'pos max':>9} {'rot mean':>9} {'rot p95':>9} {'rot max':>9}"
    print(header)
    for task in TASKS:
        for target, stats in report[task].items():
            print(
                f"{task:<12} {target:<7} {stats['pos_mean_mm']:>7.2f}mm {stats['pos_p95_mm']:>7.2f}mm "
                f"{stats['pos_max_mm']:>7.2f}mm {stats['rot_mean_deg']:>6.2f}deg {stats['rot_p95_deg']:>6.2f}deg "
                f"{stats['rot_max_deg']:>6.2f}deg"
            )


if __name__ == "__main__":
    main()
