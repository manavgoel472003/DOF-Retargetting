#!/usr/bin/env python3
"""Physics grasp test: every arm must actually PICK the object, not just mime it.

Each arm gets its own MuJoCo dynamics scene (built with mjSpec from the same URDFs the
retargeter used): a 22mm box rests on a stand at the task's pick point, a second stand
waits at the place point, position servos drive the retargeted joint trajectory, and the
jaws are commanded from the demo's gripper fraction. Success requires that BOTH fingers
are in contact with the object while it is carried, that the object is lifted, and that
it ends up on the place stand.

Robot self/arm collisions are disabled so the test isolates gripper-object physics; only
the jaw bodies, the object, the stands, and the floor collide.
"""

from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "validation"))

import mujoco

from mujoco_validate import ARMS
from robot_trajectory_retargeting.urdf_chain import compute_canonical_tcp

OUT = REPO / "validation" / "outputs"

OBJECT_HALF = np.array([0.008, 0.008, 0.030])  # 16mm wide, 60mm tall box
STEPS_PER_FRAME = 20
SETTLE_SECONDS = 1.0

# (joint, closed_value, open_value) and the two finger bodies whose contacts prove the grasp.
GRIPPERS = {
    # jaw_kp/jaw_force: hinge jaws get Nm/rad-scale torque servos, prismatic jaws get
    # N/m-scale force servos; both saturate at a realistic grip force.
    "so101": {
        "joints": [("gripper", -0.12, 1.5)],
        "jaw_bodies": ("gripper_link", "moving_jaw_so101_v1_link"),
        "jaw_kp": 20.0,
        "jaw_force": 2.5,
    },
    "b601": {
        "joints": [("jaw_left", 0.0, 0.035), ("jaw_right", 0.0, 0.035)],
        "jaw_bodies": ("finger_left", "finger_right"),
        "jaw_kp": 500.0,
        "jaw_force": 15.0,
    },
    "piper": {
        "joints": [("joint7", 0.0, 0.035), ("joint8", 0.0, -0.035)],
        "jaw_bodies": ("link7", "link8"),
        "jaw_kp": 500.0,
        "jaw_force": 15.0,
    },
    "yam": {
        "joints": [("tip1", -0.048, 0.0), ("tip2", -0.048, 0.0)],
        "jaw_bodies": ("linear_module", "linear_module_2"),
        "jaw_kp": 500.0,
        "jaw_force": 15.0,
    },
    "panda": {
        "joints": [("panda_finger_joint1", 0.0, 0.04), ("panda_finger_joint2", 0.0, 0.04)],
        "jaw_bodies": ("panda_leftfinger", "panda_rightfinger"),
        "jaw_kp": 500.0,
        "jaw_force": 15.0,
    },
    "kuka": {
        "joints": [("jaw_left", 0.0, 0.04), ("jaw_right", 0.0, 0.04)],
        "jaw_bodies": ("finger_left", "finger_right"),
        "jaw_kp": 500.0,
        "jaw_force": 15.0,
    },
}

HOLD_TIME_S = 6.0  # mid-transfer: object must be airborne and held
PICK_TIME_S = 4.0  # middle of the grasp dwell
PLACE_TIME_S = 10.4  # middle of the release dwell


def task_geometry_from_demo() -> tuple[np.ndarray, np.ndarray, float]:
    """Object pick/place points and grasp yaw, taken from where the source demo
    *actually* goes (canonical-frame FK), not from the synthetic task spec — exactly like
    an object placed under a real teleop demo. The yaw aligns the box faces with the
    demo's achieved jaw-separation direction."""
    reference = np.load(OUT / "retarget_pick_place_b601.npz", allow_pickle=True)
    canonical = reference["canonical_ee_poses"]
    t = reference["source_timestamps_s"]
    pick_pose = canonical[int(np.argmin(np.abs(t - PICK_TIME_S)))]
    place_pose = canonical[int(np.argmin(np.abs(t - PLACE_TIME_S)))]
    separation = pick_pose[:3, 1]
    yaw = float(np.arctan2(separation[1], separation[0]))
    return pick_pose[:3, 3].copy(), place_pose[:3, 3].copy(), yaw


def grip_command(fraction: float) -> float:
    """Demo fraction -> jaw servo setpoint fraction. Anything below 0.2 commands a full
    close so the position servo keeps squeezing the object instead of hovering at the
    exact object width (which differs per gripper)."""
    return float(np.clip((fraction - 0.2) / 0.8, 0.0, 1.0))


class GraspScene:
    def __init__(self, name: str, spec_info: dict, pick_canonical: np.ndarray,
                 place_canonical: np.ndarray, grasp_yaw: float,
                 mount_shift: np.ndarray | None = None):
        self.name = name
        urdf = spec_info["urdf"]
        self.urdf_path = urdf
        self.tcp_link = spec_info["tcp"]
        arm = compute_canonical_tcp(urdf, spec_info["tcp"])
        self.world_rotation = arm.base_rotation.T
        # The can/desk live at fixed positions in the world; the mount_shift (auto-derived
        # by the retargeter, read from its output) bolts the arm back/up from the shared
        # spot so it can reach the fixed task.
        if mount_shift is None:
            mount_shift = np.asarray(spec_info.get("mount_shift", [0.0, 0.0, 0.0]))
        self.mount = arm.mount_offset_base - np.asarray(mount_shift, dtype=float)
        to_base = lambda p: arm.base_rotation @ p + self.mount  # noqa: E731

        spec = mujoco.MjSpec.from_file(str(Path(urdf).resolve()))
        spec.option.timestep = 1.0 / (30 * STEPS_PER_FRAME)
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        # Robot geoms are made non-colliding below; URDF compilation discards
        # non-colliding geoms unless told otherwise, leaving an invisible arm.
        spec.compiler.discardvisual = False

        # Contact bitmask (contype/conaffinity), one bit per role so we get exactly the
        # contacts a real pick needs and none of the ones that explode the solver:
        #   desk=1  object=2  pad=4  armlink=8
        # desk<->{object,armlink}, object<->{desk,pad}, pad<->object, armlink<->desk.
        # Crucially armlinks collide with the DESK ONLY -- never the object (mesh hulls
        # would swallow it), never each other (self-collision blows up), and never the
        # fingers' approach to a can resting on the desk.
        DESK_T, OBJ_T, PAD_T, LINK_T = 1, 2, 4, 8
        # Mesh convex hulls grabbing the object are useless, so the gripper itself stays
        # non-colliding (primitive fingertip pads, added below, do the grasping). Arm links
        # get a one-sided floor BELOW the desk (an invisible safety plane a margin under the
        # work surface) so a flailing arm can't plunge through the desk, while a small arm
        # legitimately working at desk level to grab a low can is not propped up / blocked.
        desk_collide_bodies = self._arm_link_bodies(name)
        FLOOR_T = 16
        # Gravity-compensate every robot body (what real position-controlled arms do):
        # the heavy industrial arms would otherwise sag centimeters under the servos.
        for body in spec.bodies:
            body.gravcomp = 1.0
            collide = body.name in desk_collide_bodies
            for geom in body.geoms:
                geom.contype = LINK_T if collide else 0
                geom.conaffinity = FLOOR_T if collide else 0

        jaw_joint_names = {j for j, _, _ in GRIPPERS[name]["joints"]}
        for joint in spec.joints:
            if joint.type == mujoco.mjtJoint.mjJNT_FREE:
                continue
            if joint.name in jaw_joint_names:
                # Jaws must close within the demo's grasp dwell (~0.6s): keep them light.
                joint.damping = np.full(3, 0.05)
                joint.armature = 0.005
                continue
            if np.all(np.asarray(joint.damping) == 0.0):
                joint.damping = np.full(3, 0.5)
            # Reflected motor inertia: keeps stiff position servos numerically stable on
            # light hobby-arm links (natural frequency sqrt(kp/armature) << 1/dt).
            joint.armature = 0.05

        arm_joints = spec_info["joints"]
        for joint_name in arm_joints:
            self._add_servo(spec, joint_name, kp=400.0, kv=40.0, force=100.0)
        for joint_name, _, _ in GRIPPERS[name]["joints"]:
            self._add_servo(
                spec, joint_name,
                kp=GRIPPERS[name]["jaw_kp"], kv=GRIPPERS[name]["jaw_kp"] / 15.0,
                force=GRIPPERS[name]["jaw_force"],
            )

        self._add_finger_pads(spec, name, arm)

        # Environment: ONE shared desk, the can resting on it, and a place marker -- all
        # at fixed canonical positions, so the world is identical for every arm. The desk
        # surface sits at the can's resting height (can bottom flush on the desk).
        spec.worldbody.add_light(pos=[0.6, 0.3, 1.5], dir=[-0.4, -0.2, -1.0])
        spec.worldbody.add_light(pos=[-0.4, -0.6, 1.2], dir=[0.3, 0.5, -1.0])
        pick = to_base(pick_canonical)
        place = to_base(place_canonical)
        desk_z = to_base(pick_canonical - np.array([0.0, 0.0, OBJECT_HALF[2]]))[2]
        spec.worldbody.add_geom(
            name="desk", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.1],
            pos=[float(self.mount[0]), float(self.mount[1]), float(desk_z)],
            contype=DESK_T, conaffinity=OBJ_T,  # the visual desk holds the can up
            rgba=[0.55, 0.43, 0.30, 1.0],
        )
        # Invisible safety floor a margin below the desk: catches an arm plunging through
        # the desk without propping up a small arm working at desk level to grab a low can.
        spec.worldbody.add_geom(
            name="arm_floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.1],
            pos=[float(self.mount[0]), float(self.mount[1]), float(desk_z - 0.05)],
            contype=FLOOR_T, conaffinity=LINK_T, rgba=[0, 0, 0, 0],
        )
        spec.worldbody.add_geom(
            name="place_marker", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[0.02, 0.001, 0.0], pos=[place[0], place[1], desk_z + 0.001],
            contype=0, conaffinity=0, rgba=[0.2, 0.7, 0.3, 1.0],
        )
        yaw = grasp_yaw  # box x-faces presented to the jaws (canonical == base axes here)
        body = spec.worldbody.add_body(
            name="pick_object", pos=pick.tolist(),
            quat=[np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)],
        )
        body.add_freejoint()
        body.add_geom(
            name="pick_object_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=OBJECT_HALF.tolist(), mass=0.03, friction=[1.5, 0.02, 0.0001],
            contype=OBJ_T, conaffinity=DESK_T | PAD_T,  # rests on desk, gripped by pads
            rgba=[0.85, 0.15, 0.15, 1.0],
        )

        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.object_body = self.model.body("pick_object").id
        self.jaw_body_ids = tuple(self.model.body(n).id for n in GRIPPERS[name]["jaw_bodies"])
        self.arm_actuators = [self.model.actuator(f"servo_{j}").id for j in arm_joints]
        self.jaw_actuators = [
            (self.model.actuator(f"servo_{j}").id, closed, opened)
            for j, closed, opened in GRIPPERS[name]["joints"]
        ]
        self.pick_base, self.place_base = pick, place

    def _arm_link_bodies(self, name: str) -> set[str]:
        """Names of the moving arm-link bodies that should collide with the desk.

        Includes every body driven by an actuated joint (so nothing dips under the desk),
        but excludes (a) the base-platform bodies, which are bolted *through* the desk, and
        (b) the gripper/finger bodies, which must reach down to a can resting on the desk
        without being blocked or jittering against it.
        """
        probe = mujoco.MjSpec.from_file(str(Path(self.urdf_path).resolve())).compile()
        movable: set[str] = set()
        for i in range(1, probe.nbody):
            j = i
            while j != 0:
                if probe.body_dofnum[j] > 0:
                    movable.add(probe.body(i).name)
                    break
                j = probe.body_parentid[j]
        exclude = set(GRIPPERS[name]["jaw_bodies"])
        for jaw in GRIPPERS[name]["jaw_bodies"]:
            exclude.add(probe.body(probe.body(jaw).parentid).name)  # the hand / gripper base
        return movable - exclude

    def _add_finger_pads(self, spec: mujoco.MjSpec, name: str, arm) -> None:
        """Attach primitive fingertip pads where the closed jaws meet.

        Pads are placed +-12mm from the TCP along the canonical jaw axis at the
        closed-jaw zero configuration; which finger owns which side is read from how each
        finger body moves between closed and open (a static finger, like SO101's fixed
        jaw, takes the side opposite the moving one). Pads only collide with the object.
        """
        from robot_trajectory_retargeting.urdf_chain import _link_base_transform, _read_joints

        # Measure jaw kinematics on a throwaway copy: MjSpec.compile() mutates the spec
        # while merging fixed links, so the real spec must be compiled exactly once.
        probe_spec = mujoco.MjSpec.from_file(self.urdf_path)
        model = probe_spec.compile()
        data = mujoco.MjData(model)

        def jaw_kinematics(values_key: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
            data.qpos[:] = 0.0
            for joint_name, closed, opened in GRIPPERS[name]["joints"]:
                joint = model.joint(joint_name)
                data.qpos[joint.qposadr[0]] = (closed, opened)[values_key]
            mujoco.mj_kinematics(model, data)
            return {
                body: (data.xpos[model.body(body).id].copy(),
                       data.xmat[model.body(body).id].reshape(3, 3).copy())
                for body in GRIPPERS[name]["jaw_bodies"]
            }

        closed_kin = jaw_kinematics(0)
        open_kin = jaw_kinematics(1)

        urdf_joints = _read_joints(self.urdf_path)
        tcp_rotation = _link_base_transform(urdf_joints, self.tcp_link)[:3, :3]
        canonical_in_base = tcp_rotation @ arm.rotation
        jaw_axis = canonical_in_base[:, 1]
        tcp_position = arm.tcp_position_base

        signs = {}
        for body, (p_closed, rotation_closed) in closed_kin.items():
            # Track where a TCP-attached point on this finger travels as the jaw opens
            # (a revolute jaw's body origin sits on the joint axis and never moves).
            tcp_local = rotation_closed.T @ (tcp_position - p_closed)
            p_open, rotation_open = open_kin[body]
            attached_open = rotation_open @ tcp_local + p_open
            travel = float(np.dot(attached_open - tcp_position, jaw_axis))
            signs[body] = 0.0 if abs(travel) < 1e-3 else float(np.sign(travel))
        moving = [s for s in signs.values() if s != 0.0]
        for body, sign in signs.items():
            if sign == 0.0:  # static finger: opposite side of the (single) moving one
                signs[body] = -moving[0]

        moving_bodies = {
            body for body, (p_closed, rotation_closed) in closed_kin.items()
            if np.linalg.norm(
                (open_kin[body][1] @ (rotation_closed.T @ (tcp_position - p_closed)) + open_kin[body][0])
                - tcp_position
            ) > 1e-3
        }
        has_static_finger = len(moving_bodies) < len(closed_kin)
        for body, sign in signs.items():
            p_body, rotation_body = closed_kin[body]
            # Moving fingers overshoot into the object at full close (squeeze force);
            # a static finger sits just clear so the approach does not clip it. With a
            # static finger the moving one must overshoot deeper, since the object gets
            # pushed sideways against the static face before any squeeze builds up.
            if body in moving_bodies:
                offset = 0.006 if has_static_finger else 0.008
            else:
                offset = 0.0125
            pad_world = tcp_position + sign * offset * jaw_axis
            pad_local = rotation_body.T @ (pad_world - p_body)
            pad_rotation_local = rotation_body.T @ canonical_in_base
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, pad_rotation_local.flatten())
            spec_body = next(b for b in spec.bodies if b.name == body)
            spec_body.add_geom(
                name=f"pad_{body}", type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.008, 0.002, 0.016], pos=pad_local.tolist(), quat=quat.tolist(),
                contype=4, conaffinity=2,  # PAD_T, touches OBJ_T only
                friction=[2.0, 0.02, 0.0001], mass=0.002,
                rgba=[0.05, 0.05, 0.05, 1.0],
            )

    @staticmethod
    def _add_servo(spec: mujoco.MjSpec, joint_name: str, kp: float, kv: float, force: float) -> None:
        act = spec.add_actuator(name=f"servo_{joint_name}", target=joint_name,
                                trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.gainprm[0] = kp
        act.biasprm[1] = -kp
        act.biasprm[2] = -kv
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        act.forcerange[:] = [-force, force]

    def set_targets(self, joints_rad: np.ndarray, fraction: float) -> None:
        for actuator, value in zip(self.arm_actuators, joints_rad):
            self.data.ctrl[actuator] = value
        command = grip_command(fraction)
        for actuator, closed, opened in self.jaw_actuators:
            self.data.ctrl[actuator] = closed + command * (opened - closed)

    def finger_contacts(self) -> tuple[int, int]:
        counts = [0, 0]
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            bodies = {
                self.model.geom_bodyid[contact.geom1],
                self.model.geom_bodyid[contact.geom2],
            }
            if self.object_body not in bodies:
                continue
            for k, jaw in enumerate(self.jaw_body_ids):
                if jaw in bodies:
                    counts[k] += 1
        return counts[0], counts[1]


def run(scene: GraspScene, joints_rad: np.ndarray, fractions: np.ndarray,
        render: bool = False, camera=None):
    object_track = []
    contact_track = []
    frames = []
    renderer = None
    view_option = None
    if render:
        renderer = mujoco.Renderer(scene.model, 300, 400)
        view_option = mujoco.MjvOption()
        view_option.geomgroup[:] = 1  # URDF-imported geoms land in group 3 (hidden by default)

    scene.set_targets(joints_rad[0], fractions[0])
    for _ in range(int(SETTLE_SECONDS * 30 * STEPS_PER_FRAME)):
        mujoco.mj_step(scene.model, scene.data)

    for k in range(len(joints_rad)):
        scene.set_targets(joints_rad[k], fractions[k])
        for _ in range(STEPS_PER_FRAME):
            mujoco.mj_step(scene.model, scene.data)
        object_track.append(scene.data.xpos[scene.object_body].copy())
        contact_track.append(scene.finger_contacts())
        if render and k % 3 == 0:
            renderer.update_scene(scene.data, camera, view_option)
            frames.append(renderer.render().copy())
    if renderer is not None:
        renderer.close()
    return np.asarray(object_track), np.asarray(contact_track), frames


def main() -> None:
    render = "--render" in sys.argv
    task = np.load(OUT / "task_pick_place_so101.npz")
    fractions = task["gripper_fraction"]
    t = task["timestamps_s"]
    hold_index = int(np.argmin(np.abs(t - HOLD_TIME_S)))

    pick_canonical, place_canonical, grasp_yaw = task_geometry_from_demo()
    print(f"object at canonical {np.round(pick_canonical, 4).tolist()} -> {np.round(place_canonical, 4).tolist()}, grasp yaw {np.degrees(grasp_yaw):.1f}deg")
    arm_order = ["so101", "b601", "piper", "yam", "panda", "kuka"]
    all_frames = {}
    print(f"{'arm':<7} {'retarget':>12} {'both @hold':>11} {'lift [cm]':>10} {'place err [cm]':>15} {'verdict':>14}")
    for name in arm_order:
        feasible = True
        mount_shift = np.zeros(3, dtype=float)
        if name == "so101":
            joints = task["joint_positions_rad"]
        else:
            data = np.load(OUT / f"retarget_pick_place_{name}.npz", allow_pickle=True)
            joints = data["smoothed_target_joint_positions_rad"]
            feasible = bool(data["retarget_feasible"])
            mount_shift = np.asarray(data["target_mount_shift"], dtype=float)  # auto-derived
        scene = GraspScene(name, ARMS[name], pick_canonical, place_canonical, grasp_yaw, mount_shift)
        if not feasible:
            # The arm can't do this cleanly -> it stays put at its home pose rather than
            # flailing through unreachable targets.
            home = np.asarray(ARMS[name].get("initial_joints", joints[0]), dtype=float)
            joints = np.repeat(home[None, :], len(joints), axis=0)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = (scene.pick_base + scene.place_base) / 2.0 + [0, 0, 0.03]
        camera.distance = 1.1 if name in ("panda", "kuka") else 0.75
        camera.azimuth = 150
        camera.elevation = -20
        track, contacts, frames = run(scene, joints, fractions, render=render, camera=camera)
        all_frames[name] = frames

        start_z = track[0, 2]
        lift_cm = (track[:, 2].max() - start_z) * 100.0
        left, right = contacts[hold_index]
        both_holding = bool(left > 0 and right > 0)
        final_err_cm = float(np.linalg.norm(track[-1, :2] - scene.place_base[:2]) * 100.0)
        landed = abs(track[-1, 2] - scene.pick_base[2]) < 0.03
        grasp_ok = both_holding and lift_cm > 5.0 and final_err_cm < 6.0 and landed
        if not feasible:
            verdict = "NOT FEASIBLE"  # honest: the arm can't reach the can from the shared mount
        else:
            verdict = "PASS" if grasp_ok else "FAIL"
        print(
            f"{name:<7} {('feasible' if feasible else 'infeasible'):>12} {f'{left}L/{right}R':>11}"
            f" {lift_cm:>10.1f} {final_err_cm:>15.1f} {verdict:>14}"
        )
        np.savez(OUT / f"grasp_track_{name}.npz", object_positions=track, contacts=contacts,
                 timestamps_s=t, place_target=scene.place_base, pick_point=scene.pick_base)

    if render:
        import imageio.v2 as imageio

        n = min(len(f) for f in all_frames.values())
        columns = 3
        tiles = []
        for i in range(n):
            rows = []
            for start in range(0, len(arm_order), columns):
                rows.append(np.hstack([all_frames[a][i] for a in arm_order[start:start + columns]]))
            tiles.append(np.vstack(rows))
        path = OUT / "grasp_physics.gif"
        imageio.mimsave(path, tiles, fps=10, loop=0)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
