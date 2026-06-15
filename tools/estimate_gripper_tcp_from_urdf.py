#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import numpy as np


def _parse_xyz(text: str | None, default: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    if not text:
        return np.array(default, dtype=float)
    return np.array([float(value) for value in text.split()], dtype=float)


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    rot_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    rot_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return rot_z @ rot_y @ rot_x


def _matrix_to_rpy(rotation: np.ndarray) -> np.ndarray:
    pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
    roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
    yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    return np.array([roll, pitch, yaw], dtype=float)


@dataclass(frozen=True)
class VisualMesh:
    filename: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray


def _iter_link_visual_meshes(root: ET.Element, link_name: str) -> list[VisualMesh]:
    for link_elem in root.findall("link"):
        if link_elem.attrib.get("name") != link_name:
            continue
        visuals: list[VisualMesh] = []
        for visual_elem in link_elem.findall("visual"):
            geometry_elem = visual_elem.find("geometry")
            if geometry_elem is None:
                continue
            mesh_elem = geometry_elem.find("mesh")
            if mesh_elem is None:
                continue
            origin_elem = visual_elem.find("origin")
            visuals.append(
                VisualMesh(
                    filename=mesh_elem.attrib["filename"],
                    origin_xyz=_parse_xyz(None if origin_elem is None else origin_elem.attrib.get("xyz")),
                    origin_rpy=_parse_xyz(None if origin_elem is None else origin_elem.attrib.get("rpy")),
                )
            )
        return visuals
    raise ValueError(f"Link '{link_name}' not found in URDF.")


def _resolve_mesh_path(urdf_path: Path, mesh_filename: str) -> Path:
    mesh_path = mesh_filename.strip()
    if mesh_path.startswith("package://"):
        package_spec = mesh_path.removeprefix("package://")
        package_name, relative_path = package_spec.split("/", 1)
        candidates = [
            urdf_path.parent.parent / package_name / relative_path,
            urdf_path.parent.parent / relative_path,
            urdf_path.parent / relative_path,
            urdf_path.parent.parent.parent / package_name / relative_path,
        ]
    else:
        candidates = [
            urdf_path.parent / mesh_path,
            urdf_path.parent.parent / mesh_path,
        ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve mesh '{mesh_filename}' from URDF '{urdf_path}'.")


def _read_stl_vertices(mesh_path: Path) -> np.ndarray:
    with mesh_path.open("rb") as handle:
        header = handle.read(80)
        triangle_count_bytes = handle.read(4)
        if len(triangle_count_bytes) != 4:
            raise ValueError(f"Invalid STL header in {mesh_path}")
        triangle_count = struct.unpack("<I", triangle_count_bytes)[0]
        payload = handle.read()

    expected_size = triangle_count * 50
    if len(payload) == expected_size:
        vertices = np.empty((triangle_count * 3, 3), dtype=np.float32)
        for idx in range(triangle_count):
            offset = idx * 50 + 12
            vertices[idx * 3 : idx * 3 + 3] = np.asarray(
                struct.unpack("<9f", payload[offset : offset + 36]),
                dtype=np.float32,
            ).reshape(3, 3)
        return vertices.astype(float)

    text = (header + triangle_count_bytes + payload).decode("utf-8", errors="ignore")
    vertices = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) == 4 and parts[0].lower() == "vertex":
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not vertices:
        raise ValueError(f"Unsupported mesh format for {mesh_path}; only STL is supported.")
    return np.asarray(vertices, dtype=float)


def _collect_link_vertices(urdf_path: Path, link_name: str) -> np.ndarray:
    root = ET.parse(urdf_path).getroot()
    visuals = _iter_link_visual_meshes(root, link_name)
    all_vertices: list[np.ndarray] = []
    for visual in visuals:
        mesh_path = _resolve_mesh_path(urdf_path, visual.filename)
        vertices = _read_stl_vertices(mesh_path)
        rotation = _rpy_matrix(visual.origin_rpy)
        transformed = (rotation @ vertices.T).T + visual.origin_xyz
        all_vertices.append(transformed)
    if not all_vertices:
        raise ValueError(f"No visual meshes found for link '{link_name}'.")
    return np.concatenate(all_vertices, axis=0)


def _infer_terminal_link(root: ET.Element) -> str:
    parents = set()
    children = set()
    for joint_elem in root.findall("joint"):
        parent_elem = joint_elem.find("parent")
        child_elem = joint_elem.find("child")
        if parent_elem is None or child_elem is None:
            continue
        parents.add(parent_elem.attrib["link"])
        children.add(child_elem.attrib["link"])
    leaf_links = sorted(children - parents)
    if not leaf_links:
        raise ValueError("Failed to infer a terminal link from the URDF.")
    if "end_link" in leaf_links:
        return "end_link"
    return leaf_links[-1]


def _estimate_tcp_from_vertices(vertices: np.ndarray, depth_ratio: float) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    spans = maxs - mins
    centers = 0.5 * (mins + maxs)
    symmetry = np.abs(centers) / np.maximum(spans, 1e-9)

    approach_axis = int(np.argmax(symmetry))
    remaining_axes = [axis for axis in range(3) if axis != approach_axis]
    opening_axis = remaining_axes[int(np.argmax(spans[remaining_axes]))]
    thickness_axis = [axis for axis in remaining_axes if axis != opening_axis][0]

    approach_sign = -1.0 if abs(mins[approach_axis]) > abs(maxs[approach_axis]) else 1.0
    tip_coord = mins[approach_axis] if approach_sign < 0.0 else maxs[approach_axis]
    backoff = max(0.008, depth_ratio * spans[approach_axis])

    tcp_xyz = centers.copy()
    tcp_xyz[approach_axis] = tip_coord - approach_sign * backoff
    tcp_xyz[opening_axis] = centers[opening_axis]
    tcp_xyz[thickness_axis] = centers[thickness_axis]

    axis_vectors = np.eye(3, dtype=float)
    tcp_z = approach_sign * axis_vectors[:, approach_axis]
    tcp_x = axis_vectors[:, opening_axis]
    tcp_y = np.cross(tcp_z, tcp_x)
    tcp_y = tcp_y / np.linalg.norm(tcp_y)
    tcp_x = np.cross(tcp_y, tcp_z)
    tcp_x = tcp_x / np.linalg.norm(tcp_x)
    rotation = np.column_stack([tcp_x, tcp_y, tcp_z])
    tcp_rpy = _matrix_to_rpy(rotation)

    debug = {
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
        "spans": spans.tolist(),
        "centers": centers.tolist(),
        "symmetry_scores": symmetry.tolist(),
        "approach_axis": approach_axis,
        "opening_axis": opening_axis,
        "thickness_axis": thickness_axis,
        "approach_sign": approach_sign,
        "backoff_m": backoff,
    }
    return tcp_xyz, tcp_rpy, debug


def _append_tcp_to_urdf(urdf_path: Path, parent_link: str, tcp_link_name: str, tcp_xyz: np.ndarray, tcp_rpy: np.ndarray, output_path: Path) -> None:
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    for element in root.findall("link"):
        if element.attrib.get("name") == tcp_link_name:
            root.remove(element)
    for element in root.findall("joint"):
        if element.attrib.get("name") == f"{tcp_link_name}_joint":
            root.remove(element)

    link_elem = ET.Element("link", {"name": tcp_link_name})
    inertial_elem = ET.SubElement(link_elem, "inertial")
    ET.SubElement(inertial_elem, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(inertial_elem, "mass", {"value": "1e-09"})
    ET.SubElement(
        inertial_elem,
        "inertia",
        {"ixx": "0", "ixy": "0", "ixz": "0", "iyy": "0", "iyz": "0", "izz": "0"},
    )

    joint_elem = ET.Element("joint", {"name": f"{tcp_link_name}_joint", "type": "fixed"})
    ET.SubElement(
        joint_elem,
        "origin",
        {
            "xyz": " ".join(f"{value:.6f}" for value in tcp_xyz),
            "rpy": " ".join(f"{value:.6f}" for value in tcp_rpy),
        },
    )
    ET.SubElement(joint_elem, "parent", {"link": parent_link})
    ET.SubElement(joint_elem, "child", {"link": tcp_link_name})
    ET.SubElement(joint_elem, "axis", {"xyz": "0 0 0"})

    root.append(link_elem)
    root.append(joint_elem)
    ET.indent(tree, space="  ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate a gripper TCP from a URDF terminal link mesh.")
    parser.add_argument("--urdf", required=True, help="Path to the URDF file.")
    parser.add_argument("--link", help="Terminal link to analyze. Defaults to inferred leaf link.")
    parser.add_argument("--tcp-link-name", default="gripper_tcp", help="Name for the generated TCP link.")
    parser.add_argument(
        "--depth-ratio",
        type=float,
        default=0.12,
        help="Backoff ratio from the mesh tip toward the gripper center along the approach axis.",
    )
    parser.add_argument("--output-urdf", help="Optional path to write a patched URDF with the fixed TCP link added.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    urdf_path = Path(args.urdf).expanduser().resolve()
    root = ET.parse(urdf_path).getroot()
    link_name = args.link or _infer_terminal_link(root)
    vertices = _collect_link_vertices(urdf_path, link_name)
    tcp_xyz, tcp_rpy, debug = _estimate_tcp_from_vertices(vertices, args.depth_ratio)

    payload = {
        "urdf": str(urdf_path),
        "link": link_name,
        "tcp_link_name": args.tcp_link_name,
        "tcp_xyz": [float(value) for value in tcp_xyz],
        "tcp_rpy": [float(value) for value in tcp_rpy],
        "debug": debug,
    }
    print(json.dumps(payload, indent=2))

    if args.output_urdf:
        output_path = Path(args.output_urdf).expanduser().resolve()
        _append_tcp_to_urdf(urdf_path, link_name, args.tcp_link_name, tcp_xyz, tcp_rpy, output_path)
        print(f"Wrote patched URDF to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
