#!/usr/bin/env python3
"""Give every validation arm a modeled, animatable gripper so end-effector handling is
consistent across arms.

- YAM: attaches the real i2rt flexible_4310 gripper URDF at link_6 (identity mount, per
  the upstream MJCF combine convention) and moves gripper_tcp to the grasp center 0.10m
  out (the upstream grasp_site). Regenerates yam_retarget.urdf from the pristine yam.urdf.
- B601: the real gripper is a damiao 4310 motor with no published URDF, so a
  representative parallel jaw (two prismatic fingers, 70mm max aperture) is added around
  the existing gripper_tcp, closing along the arm's canonical jaw axis. The TCP itself is
  untouched, so retargeting results are unchanged. Writes reBot-DevArm_gripper.urdf.
- SO101 and Piper already model their jaws in the URDF.
"""

from pathlib import Path
import re
import sys

import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from robot_trajectory_retargeting.urdf_chain import (
    _link_base_transform,
    _read_joints,
    compute_canonical_tcp,
)

ASSETS = REPO / "examples" / "assets"
YAM_TCP_FROM_LINK6 = 0.10  # upstream flexible_4310 MJCF grasp_site distance
B601_APERTURE = 0.070


def build_yam() -> None:
    yam = (ASSETS / "yam" / "yam.urdf").read_text()
    yam = yam.replace("package://assets/", "assets/")

    # link_6 visual/collision meshes do not exist upstream; the gripper provides the
    # visual instead.
    def strip_geometry(match: re.Match) -> str:
        block = re.sub(r"<visual>.*?</visual>", "", match.group(0), flags=re.S)
        return re.sub(r"<collision>.*?</collision>", "", block, flags=re.S)

    yam = re.sub(r'<link\s+name="link_6">.*?</link>', strip_geometry, yam, flags=re.S)

    gripper = (Path("/tmp/flexible_4310.urdf")).read_text()
    gripper = gripper.replace('robot name="output"', 'robot name="flexible_4310"')
    gripper = gripper.replace('"base"', '"gripper_base_link"')
    gripper = gripper.replace('filename="assets/', 'filename="gripper_assets/')
    body = re.search(r"<robot[^>]*>(.*)</robot>", gripper, flags=re.S).group(1)

    attach = f"""  <joint name="gripper_mount_joint" type="fixed">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="link_6"/>
    <child link="gripper_base_link"/>
  </joint>
{body}
  <link name="gripper_tcp"/>
  <joint name="gripper_tcp_joint" type="fixed">
    <origin xyz="0 0 {YAM_TCP_FROM_LINK6}" rpy="0 0 0"/>
    <parent link="link_6"/>
    <child link="gripper_tcp"/>
  </joint>
</robot>"""
    yam = yam.replace("</robot>", attach)
    (ASSETS / "yam" / "yam_retarget.urdf").write_text(yam)
    print("wrote yam_retarget.urdf (flexible_4310 attached, tcp at link_6 + %.2fm)" % YAM_TCP_FROM_LINK6)


def synthetic_jaw_blocks(urdf_path: Path, parent_link: str, tcp_link: str, aperture: float) -> str:
    """URDF for a representative parallel jaw on ``parent_link``, centered at
    ``tcp_link`` and separating along the arm's canonical jaw axis."""
    arm = compute_canonical_tcp(urdf_path, tcp_link)
    joints = _read_joints(urdf_path)
    parent = _link_base_transform(joints, parent_link)
    tcp = _link_base_transform(joints, tcp_link)
    tcp_local = (np.linalg.inv(parent) @ tcp)[:3, 3]
    z_local = parent[:3, :3].T @ arm.approach_axis_base
    y_local = parent[:3, :3].T @ arm.pitch_axis_base
    y_local -= np.dot(y_local, z_local) * z_local
    y_local /= np.linalg.norm(y_local)
    x_local = np.cross(y_local, z_local)
    rpy = Rotation.from_matrix(np.column_stack([x_local, y_local, z_local])).as_euler("xyz")

    half_stroke = aperture / 2.0
    finger = """  <link name="finger_{side}">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.02"/>
      <inertia ixx="1e-6" ixy="0" ixz="0" iyy="1e-6" iyz="0" izz="1e-6"/>
    </inertial>
    <visual>
      <origin xyz="0 {y0} -0.01" rpy="0 0 0"/>
      <geometry><box size="0.012 0.008 0.055"/></geometry>
      <material name="finger_material"><color rgba="0.2 0.2 0.25 1"/></material>
    </visual>
    <collision>
      <origin xyz="0 {y0} -0.01" rpy="0 0 0"/>
      <geometry><box size="0.012 0.008 0.055"/></geometry>
    </collision>
  </link>
  <joint name="jaw_{side}" type="prismatic">
    <origin xyz="{x} {y} {z}" rpy="{r} {p} {yaw}"/>
    <parent link="{parent}"/>
    <child link="finger_{side}"/>
    <axis xyz="0 {sign}1 0"/>
    <limit effort="10" velocity="1" lower="0" upper="{stroke}"/>
  </joint>
"""
    blocks = ""
    for side, sign in (("left", "+"), ("right", "-")):
        blocks += finger.format(
            side=side, sign=sign, stroke=f"{half_stroke:.4f}", parent=parent_link,
            x=f"{tcp_local[0]:.6f}", y=f"{tcp_local[1]:.6f}", z=f"{tcp_local[2]:.6f}",
            r=f"{rpy[0]:.6f}", p=f"{rpy[1]:.6f}", yaw=f"{rpy[2]:.6f}",
            y0=f"{'+' if sign == '+' else '-'}0.006",
        )
    return blocks


def build_b601() -> None:
    urdf_dir = ASSETS / "rebot_b601" / "reBot-DevArm_fixend_description" / "urdf"
    source_path = urdf_dir / "reBot-DevArm_fixend.urdf"
    text = source_path.read_text()
    blocks = synthetic_jaw_blocks(source_path, "end_link", "gripper_tcp", B601_APERTURE)
    out = urdf_dir / "reBot-DevArm_gripper.urdf"
    out.write_text(text.replace("</robot>", blocks + "</robot>"))
    print(f"wrote {out.name} (parallel jaw, {B601_APERTURE*1000:.0f}mm aperture at gripper_tcp)")


def build_panda() -> None:
    text = (ASSETS / "panda" / "panda.urdf").read_text()
    text = text.replace("package://meshes/", "meshes/")
    out = ASSETS / "panda" / "panda_retarget.urdf"
    out.write_text(text)
    print(f"wrote {out.name} (real prismatic fingers, tcp = panda_grasptarget)")


def build_iiwa() -> None:
    """KUKA iiwa14 ships with a bare flange: add a TCP 0.12m out along the flange z and
    a representative 80mm parallel jaw around it."""
    src = ASSETS / "kuka_iiwa" / "model.urdf"
    text = src.read_text()
    tcp = """  <link name="gripper_tcp"/>
  <joint name="gripper_tcp_joint" type="fixed">
    <origin xyz="0 0 0.12" rpy="0 0 0"/>
    <parent link="lbr_iiwa_link_7"/>
    <child link="gripper_tcp"/>
  </joint>
</robot>"""
    out = ASSETS / "kuka_iiwa" / "iiwa_retarget.urdf"
    out.write_text(text.replace("</robot>", tcp))
    blocks = synthetic_jaw_blocks(out, "lbr_iiwa_link_7", "gripper_tcp", 0.08)
    out.write_text(out.read_text().replace("</robot>", blocks + "</robot>"))
    print(f"wrote {out.name} (tcp at flange+0.12m, 80mm parallel jaw)")


if __name__ == "__main__":
    build_yam()
    build_b601()
    build_panda()
    build_iiwa()
