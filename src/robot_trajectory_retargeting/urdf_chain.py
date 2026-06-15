"""Minimal URDF chain reader used to derive a shared canonical TCP convention.

The retargeter transfers end-effector poses between arms by re-expressing every
TCP pose in a *shared canonical TCP frame* before transfer, then converting back
into the target arm's native TCP frame. The rotation that maps an arm's native
TCP frame onto that shared convention is a fixed property of the URDF, so it is
computed once, here, from joint geometry at the zero configuration.

Canonical TCP convention (right-handed, rigidly attached to the TCP link):
  - z = gripper approach axis (points out of the gripper, away from the wrist)
  - y = jaw separation direction: the direction the fingertips move apart, so every
        gripper closes on the same faces of an object. Prismatic jaws separate along
        their slide axis; hinged jaws separate along hinge x approach. When no jaw is
        modeled, a hinged gripper with its hinge on the wrist-flex axis is assumed
        (see _find_jaw_axis_base and _select_pitch_axis).
  - x = y x z

This needs no meshes: the approach direction is taken as the TCP point relative
to the last revolute joint (the wrist center), and the in-plane reference is the
wrist-pitch axis. Both are read straight from the kinematic tree.

FK is only ever evaluated at the zero configuration, where every revolute angle
is 0, so a joint's local rotation is identity and the link transform is just the
product of the joints' origin transforms along the chain.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from robot_trajectory_retargeting.frames import rpy_matrix

_REVOLUTE_TYPES = {"revolute", "continuous"}
_ACTUATED_TYPES = _REVOLUTE_TYPES | {"prismatic"}


@dataclass(frozen=True)
class _Joint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray


def _parse_triplet(text: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if text is None or text.strip() == "":
        return np.array(default, dtype=float)
    parts = [float(value) for value in text.replace(",", " ").split()]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 values, got {text!r}")
    return np.array(parts, dtype=float)


def _read_joints(urdf_path: Path) -> list[_Joint]:
    root = ET.parse(urdf_path).getroot()
    joints: list[_Joint] = []
    for joint_elem in root.findall("joint"):
        parent_elem = joint_elem.find("parent")
        child_elem = joint_elem.find("child")
        if parent_elem is None or child_elem is None:
            continue
        origin_elem = joint_elem.find("origin")
        axis_elem = joint_elem.find("axis")
        joints.append(
            _Joint(
                name=joint_elem.attrib.get("name", ""),
                joint_type=joint_elem.attrib.get("type", "fixed"),
                parent=parent_elem.attrib["link"],
                child=child_elem.attrib["link"],
                origin_xyz=_parse_triplet(None if origin_elem is None else origin_elem.attrib.get("xyz"), (0.0, 0.0, 0.0)),
                origin_rpy=_parse_triplet(None if origin_elem is None else origin_elem.attrib.get("rpy"), (0.0, 0.0, 0.0)),
                axis=_parse_triplet(None if axis_elem is None else axis_elem.attrib.get("xyz"), (0.0, 0.0, 1.0)),
            )
        )
    if not joints:
        raise ValueError(f"No joints found in URDF '{urdf_path}'.")
    return joints


def _origin_transform(joint: _Joint) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rpy_matrix(joint.origin_rpy)
    transform[:3, 3] = joint.origin_xyz
    return transform


def _chain_to_link(joints: list[_Joint], tcp_link: str) -> list[_Joint]:
    """Ordered base -> tcp_link list of joints by walking child links upward."""
    by_child = {joint.child: joint for joint in joints}
    if tcp_link not in by_child:
        raise ValueError(f"Link '{tcp_link}' is not a child of any joint in the URDF.")
    chain: list[_Joint] = []
    link = tcp_link
    seen: set[str] = set()
    while link in by_child:
        if link in seen:
            raise ValueError("Cycle detected while walking URDF chain.")
        seen.add(link)
        joint = by_child[link]
        chain.append(joint)
        link = joint.parent
    chain.reverse()
    return chain


def _link_base_transform(joints: list[_Joint], link: str) -> np.ndarray:
    """Transform of a link's frame in the base at the zero configuration."""
    by_child = {joint.child: joint for joint in joints}
    stack: list[_Joint] = []
    cur = link
    while cur in by_child:
        joint = by_child[cur]
        stack.append(joint)
        cur = joint.parent
    transform = np.eye(4, dtype=float)
    for joint in reversed(stack):
        transform = transform @ _origin_transform(joint)
    return transform


def _find_jaw_axis_base(
    joints: list[_Joint], tcp_link: str, tcp_position: np.ndarray | None = None
) -> np.ndarray | None:
    """Jaw *separation* direction in the base frame, if the URDF models a moving jaw.

    The gripper's moving finger is an actuated joint that branches off the arm chain: its
    parent link is on the path to the TCP, its child link is not. Among such joints we take
    the one attached deepest (closest to the gripper). The separation direction — the
    direction the fingertips move apart, which is what must transfer between arms so every
    gripper closes on the same faces of an object — depends on the joint type:
      - prismatic jaw: the slide axis itself;
      - revolute (hinged) jaw: the fingertip swing tangent, hinge_axis x (tcp - hinge),
        which is perpendicular to the hinge. Using the hinge axis here would leave
        hinged and parallel grippers 90 degrees apart.
    Returns None when the URDF has no moving jaw (e.g. a fixed-end-effector URDF).
    """
    chain = _chain_to_link(joints, tcp_link)
    chain_order = {chain[0].parent: -1}
    for index, joint in enumerate(chain):
        chain_order[joint.child] = index
    # A gripper body is often hung off the chain via a fixed mount link
    # (e.g. flange -> gripper_base -> moving jaws); fold such fixed descendants
    # into the chain so their actuated children are still recognized as jaws.
    expanded = True
    while expanded:
        expanded = False
        for joint in joints:
            if joint.joint_type == "fixed" and joint.parent in chain_order and joint.child not in chain_order:
                chain_order[joint.child] = chain_order[joint.parent]
                expanded = True
    actuated = _REVOLUTE_TYPES | {"prismatic"}
    candidates = [
        joint
        for joint in joints
        if joint.joint_type in actuated and joint.parent in chain_order and joint.child not in chain_order
    ]
    if not candidates:
        return None
    jaw = max(candidates, key=lambda joint: chain_order[joint.parent])
    joint_frame = _link_base_transform(joints, jaw.parent) @ _origin_transform(jaw)
    axis_base = joint_frame[:3, :3] @ jaw.axis
    if jaw.joint_type == "prismatic":
        return axis_base
    if tcp_position is None:
        return axis_base
    swing = np.cross(axis_base, tcp_position - joint_frame[:3, 3])
    norm = float(np.linalg.norm(swing))
    return axis_base if norm < 1e-9 else swing / norm


@dataclass(frozen=True)
class CanonicalTcp:
    """Result of analyzing one arm's TCP frame at the zero configuration."""

    rotation: np.ndarray  # 3x3, maps native TCP frame -> shared canonical frame (C)
    approach_axis_base: np.ndarray  # canonical z in base frame at zero config
    pitch_axis_base: np.ndarray  # wrist-pitch axis in base frame at zero config
    tcp_position_base: np.ndarray  # TCP origin in base frame at zero config
    mount_offset_base: np.ndarray  # first *actuated* joint origin in base frame: the
    #                                top-center of the base platform where the arm is
    #                                clamped down. Fixed mounting joints before the first
    #                                motor are folded in, so two arms are aligned at their
    #                                mount point, not at whatever spot each URDF happens
    #                                to place its base_link origin.
    pitch_axis_source: str  # where the secondary axis came from: "jaw" | "flex" | "override"
    base_rotation: np.ndarray  # 3x3 canonical base frame in base coords (columns =
    #                            forward, left, up). Cancels differing base-frame
    #                            conventions between URDFs (e.g. an arm whose zero-config
    #                            reach direction is +y instead of +x).
    up_axis_base: np.ndarray  # canonical base z: the pan-joint axis (vertical)
    forward_axis_base: np.ndarray  # canonical base x: zero-config reach direction
    forward_source: str  # where forward came from: "tcp" | "lift" | "override"
    max_reach_m: float  # sum of link segment lengths mount -> TCP: hard upper bound
    #                     on how far the TCP can ever be from the mount point.


def _select_pitch_axis(revolute_axes: list[np.ndarray], z_axis: np.ndarray) -> np.ndarray:
    """Pick the gripper's secondary (pitch) reference axis: the wrist *flex* axis.

    The flex axis tilts the gripper in the arm's bending plane, so it is parallel to the
    shoulder-lift axis. We scan the revolute joints from the wrist inward and take the last
    one that is (a) not the final roll (i.e. not parallel to the approach axis) and (b)
    parallel to the shoulder-lift axis. This is robust to wrists with a different number of
    DOF: naively taking the "second-to-last revolute" grabs the *yaw* joint on a 6-DOF wrist
    (SO101 has flex+roll; B601 has flex+yaw+roll), which rotates the gripper ~90 degrees.
    """
    if len(revolute_axes) < 2:
        return revolute_axes[-1]
    lift_axis = revolute_axes[1] / (np.linalg.norm(revolute_axes[1]) + 1e-12)  # shoulder-lift
    for axis in reversed(revolute_axes[:-1]):  # skip the final roll
        unit = axis / (np.linalg.norm(axis) + 1e-12)
        if abs(float(np.dot(unit, z_axis))) > 0.9:
            continue  # parallel to approach -> roll-like, not a pitch
        if abs(float(np.dot(unit, lift_axis))) > 0.5:
            return axis  # parallel to the bending plane -> a flex/pitch joint
    return revolute_axes[-2]  # fallback: previous "second-to-last revolute" behaviour


def compute_canonical_tcp(
    urdf_path: str | Path,
    tcp_link: str,
    gripper_axis_tcp: np.ndarray | None = None,
    use_jaw_geometry: bool = True,
    base_forward: np.ndarray | None = None,
) -> CanonicalTcp:
    """Derive the constant native-TCP -> canonical-TCP rotation for one arm.

    The secondary (pitch / jaw-opening) reference axis is chosen, in priority order:
      1. ``gripper_axis_tcp`` if given (explicit axis in the native TCP frame).
      2. the gripper's moving-jaw axis from the URDF, if one is modelled and
         ``use_jaw_geometry`` is True. This anchors the frame to how the gripper actually
         opens. Its sign is reconciled to the wrist-flex axis so arms that fall back to the
         flex heuristic stay consistent with arms that have jaw geometry.
      3. the wrist-flex axis heuristic (see _select_pitch_axis).

    Also derives a *canonical base frame* (forward/left/up in base coords): up is the
    pan-joint axis and forward is the zero-config reach direction projected onto the
    horizontal plane (``base_forward`` overrides). Two arms whose URDFs disagree on which
    way the base x-axis points still transfer trajectories into the same physical space.
    """
    urdf_path = Path(urdf_path).expanduser().resolve()
    joints = _read_joints(urdf_path)
    chain = _chain_to_link(joints, tcp_link)

    transform = np.eye(4, dtype=float)
    revolute_frames: list[np.ndarray] = []
    revolute_axes: list[np.ndarray] = []
    actuated_origins: list[np.ndarray] = []
    mount_offset = None
    for joint in chain:
        transform = transform @ _origin_transform(joint)
        if joint.joint_type in _ACTUATED_TYPES:
            if mount_offset is None:
                # First *actuated* joint off the base: its origin is the arm's mounting
                # interface (top-center of the base platform). Fixed mounting joints
                # (e.g. base_link -> mount_plate) are folded into the running transform
                # instead of being mistaken for the mount themselves.
                mount_offset = transform[:3, 3].copy()
            actuated_origins.append(transform[:3, 3].copy())
        if joint.joint_type in _REVOLUTE_TYPES:
            revolute_frames.append(transform.copy())
            revolute_axes.append(transform[:3, :3] @ joint.axis)

    tcp_rotation = transform[:3, :3]
    tcp_position = transform[:3, 3]

    if not revolute_frames:
        raise ValueError(f"No revolute joints found on the chain to '{tcp_link}'.")

    # Hard reach bound: the TCP can never be farther from the mount than the sum of the
    # rigid segment lengths between consecutive actuated-joint origins (plus wrist->TCP).
    waypoints = actuated_origins + [tcp_position]
    max_reach = float(
        sum(np.linalg.norm(waypoints[i + 1] - waypoints[i]) for i in range(len(waypoints) - 1))
    )

    wrist_center = revolute_frames[-1][:3, 3]
    approach = tcp_position - wrist_center
    if np.linalg.norm(approach) < 1e-6:
        # Degenerate: TCP coincides with the last joint; fall back to its axis.
        approach = revolute_axes[-1]
    z_axis = _normalize(approach)

    # --- Canonical base frame: up = pan axis, forward = zero-config reach direction ---
    up_axis = _normalize(revolute_axes[0])
    if abs(up_axis[2]) > 0.5:
        if up_axis[2] < 0.0:
            up_axis = -up_axis
    elif float(np.dot(up_axis, tcp_position - mount_offset)) < 0.0:
        up_axis = -up_axis

    forward_source = "tcp"
    if base_forward is not None:
        forward_raw = np.asarray(base_forward, dtype=float)
        forward_source = "override"
    else:
        forward_raw = tcp_position - mount_offset
    forward_axis = forward_raw - np.dot(forward_raw, up_axis) * up_axis
    if np.linalg.norm(forward_axis) < 0.02:
        # Zero-config TCP sits (nearly) on the pan axis: use the bending plane instead.
        lift_axis = revolute_axes[1] if len(revolute_axes) > 1 else revolute_axes[0]
        forward_axis = np.cross(lift_axis, up_axis)
        forward_source = "lift"
    forward_axis = _normalize(forward_axis)
    left_axis = np.cross(up_axis, forward_axis)
    base_rotation = np.column_stack([forward_axis, left_axis, up_axis])

    flex_axis = _select_pitch_axis(revolute_axes, z_axis)
    flex_unit = flex_axis / (np.linalg.norm(flex_axis) + 1e-12)
    left_alignment = float(np.dot(flex_unit, left_axis))
    if abs(left_alignment) > 0.2 and left_alignment < 0.0:
        flex_axis = -flex_axis
    jaw_axis = _find_jaw_axis_base(joints, tcp_link, tcp_position) if use_jaw_geometry else None
    if gripper_axis_tcp is not None:
        pitch_axis = tcp_rotation @ np.asarray(gripper_axis_tcp, dtype=float)
        pitch_axis_source = "override"
    elif jaw_axis is not None:
        pitch_axis = jaw_axis
        pitch_axis_source = "jaw"
    else:
        # No jaw modeled: assume a hinged hobby gripper whose hinge lies on the flex
        # axis, so the fingers separate along flex x approach.
        pitch_axis = np.cross(flex_axis, z_axis)
        if np.linalg.norm(pitch_axis) < 1e-6:
            pitch_axis = flex_axis
        pitch_axis_source = "flex"
    y_axis = pitch_axis - np.dot(pitch_axis, z_axis) * z_axis
    if np.linalg.norm(y_axis) < 1e-6:
        # Separation axis is parallel to approach; pick any in-plane reference.
        fallback = np.array([1.0, 0.0, 0.0]) if abs(z_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        y_axis = fallback - np.dot(fallback, z_axis) * z_axis
    y_axis = _normalize(y_axis)
    # Deterministic sign so two arms agree on +y regardless of how each URDF happened to
    # orient its jaw joint: prefer y along canonical-base "left"; when y is perpendicular
    # to "left" (vertical separation), require x = y x z to point along "left" instead.
    y_left = float(np.dot(y_axis, left_axis))
    if abs(y_left) > 0.2:
        if y_left < 0.0:
            y_axis = -y_axis
    elif float(np.dot(np.cross(y_axis, z_axis), left_axis)) < 0.0:
        y_axis = -y_axis
    x_axis = _normalize(np.cross(y_axis, z_axis))
    y_axis = np.cross(z_axis, x_axis)

    canonical_in_base = np.column_stack([x_axis, y_axis, z_axis])
    rotation = tcp_rotation.T @ canonical_in_base
    return CanonicalTcp(
        rotation=rotation,
        approach_axis_base=z_axis,
        pitch_axis_base=_normalize(pitch_axis),
        tcp_position_base=tcp_position,
        mount_offset_base=mount_offset,
        pitch_axis_source=pitch_axis_source,
        base_rotation=base_rotation,
        up_axis_base=up_axis,
        forward_axis_base=forward_axis,
        forward_source=forward_source,
        max_reach_m=max_reach,
    )


def read_joint_limits(urdf_path: str | Path, joint_names: list[str]) -> dict[str, tuple[float, float]]:
    """Per-joint (lower, upper) position limits in radians, read from the URDF.

    Continuous joints and joints without a <limit> element map to (-inf, +inf).
    """
    root = ET.parse(Path(urdf_path).expanduser().resolve()).getroot()
    limits: dict[str, tuple[float, float]] = {}
    for joint_elem in root.findall("joint"):
        name = joint_elem.attrib.get("name", "")
        if name not in joint_names:
            continue
        limit_elem = joint_elem.find("limit")
        if joint_elem.attrib.get("type") == "continuous" or limit_elem is None:
            limits[name] = (-np.inf, np.inf)
        else:
            limits[name] = (
                float(limit_elem.attrib.get("lower", "-inf")),
                float(limit_elem.attrib.get("upper", "inf")),
            )
    return {name: limits.get(name, (-np.inf, np.inf)) for name in joint_names}


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError("Cannot normalize a zero-length vector.")
    return vector / norm
