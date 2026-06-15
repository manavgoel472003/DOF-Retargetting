from __future__ import annotations

import math

import numpy as np


def rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw to a 3x3 rotation: Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    roll, pitch, yaw = [float(value) for value in rpy]
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    rot_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    rot_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return rot_z @ rot_y @ rot_x


def pose_from_xyz_rpy(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = rpy_matrix(np.asarray(rpy, dtype=float))
    pose[:3, 3] = np.asarray(xyz, dtype=float)
    return pose


def invert_pose(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=float)
    inverse = np.eye(4, dtype=float)
    inverse[:3, :3] = pose[:3, :3].T
    inverse[:3, 3] = -pose[:3, :3].T @ pose[:3, 3]
    return inverse


def rotation_matrix_to_rotvec(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    cos_angle = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    angle = math.acos(cos_angle)
    if angle < 1e-9:
        return np.zeros(3, dtype=float)
    sin_angle = math.sin(angle)
    if sin_angle > 1e-6:
        skew = np.array(
            [
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ],
            dtype=float,
        )
        return angle * skew / (2.0 * sin_angle)
    # angle ~ pi: the skew part vanishes, so recover the axis from the symmetric
    # part instead (R ~ 2*aa^T - I). A half-turn error must NOT report as zero.
    axis = np.sqrt(np.maximum((np.diagonal(rotation) + 1.0) * 0.5, 0.0))
    major = int(np.argmax(axis))
    if axis[major] < 1e-9:
        return np.zeros(3, dtype=float)
    for other in range(3):
        if other != major:
            axis[other] = rotation[major, other] / (2.0 * axis[major])
    return angle * axis / np.linalg.norm(axis)


def pose_error(current_pose: np.ndarray, target_pose: np.ndarray) -> tuple[float, float]:
    position_error = np.linalg.norm(target_pose[:3, 3] - current_pose[:3, 3])
    rotation_error = rotation_matrix_to_rotvec(target_pose[:3, :3] @ current_pose[:3, :3].T)
    return float(position_error), float(np.linalg.norm(rotation_error))


def apply_rotation_offset(poses: np.ndarray, rotation_offset: np.ndarray) -> np.ndarray:
    """Right-multiply each pose's orientation by a constant 3x3 rotation; keep positions.

    Used to re-express a TCP-pose trajectory in a different, rigidly-attached frame
    convention (e.g. the shared canonical TCP frame) without moving the TCP point.
    """
    poses = np.asarray(poses, dtype=float)
    rotation_offset = np.asarray(rotation_offset, dtype=float)
    mapped = poses.copy()
    mapped[:, :3, :3] = poses[:, :3, :3] @ rotation_offset
    return mapped


def smooth_joint_trajectory(joint_positions: np.ndarray, window_size: int = 7, passes: int = 2) -> np.ndarray:
    if window_size <= 1 or passes <= 0:
        return np.asarray(joint_positions, dtype=float).copy()
    window_size = int(window_size)
    if window_size % 2 == 0:
        window_size += 1
    kernel = np.ones(window_size, dtype=float)
    kernel /= kernel.sum()
    smoothed = np.asarray(joint_positions, dtype=float).copy()
    half_window = window_size // 2
    for _ in range(passes):
        padded = np.pad(smoothed, ((half_window, half_window), (0, 0)), mode="edge")
        next_smoothed = np.empty_like(smoothed)
        for joint_idx in range(smoothed.shape[1]):
            next_smoothed[:, joint_idx] = np.convolve(padded[:, joint_idx], kernel, mode="valid")
        smoothed = next_smoothed
    return smoothed


def smooth_pose_trajectory(poses: np.ndarray, window_size: int = 5, passes: int = 1) -> np.ndarray:
    """Smooth an SE(3) pose trajectory directly in task space.

    Positions get a moving average; orientations get a windowed chordal mean (average the
    rotation matrices, project back onto SO(3) via SVD). Smoothing the desired EE path
    *before* IK keeps the executed path on top of what the diagnostics measure, instead of
    letting a later joint-space filter silently bend the EE path.
    """
    poses = np.asarray(poses, dtype=float)
    if window_size <= 1 or passes <= 0 or len(poses) < 3:
        return poses.copy()
    window_size = int(window_size)
    if window_size % 2 == 0:
        window_size += 1
    half_window = window_size // 2
    smoothed = poses.copy()
    for _ in range(passes):
        padded = np.concatenate(
            [np.repeat(smoothed[:1], half_window, axis=0), smoothed, np.repeat(smoothed[-1:], half_window, axis=0)]
        )
        next_smoothed = smoothed.copy()
        for index in range(len(smoothed)):
            window = padded[index : index + window_size]
            next_smoothed[index, :3, 3] = window[:, :3, 3].mean(axis=0)
            mean_rotation = window[:, :3, :3].mean(axis=0)
            u, _, vt = np.linalg.svd(mean_rotation)
            projection = u @ vt
            if np.linalg.det(projection) < 0.0:
                u[:, -1] = -u[:, -1]
                projection = u @ vt
            next_smoothed[index, :3, :3] = projection
        smoothed = next_smoothed
    return smoothed


def make_replay_timestamps(
    joint_positions: np.ndarray,
    source_timestamps_s: np.ndarray,
    playback_speed_scale: float = 0.2,
    max_joint_velocity_deg_s: float | None = None,
) -> np.ndarray:
    if playback_speed_scale <= 0.0:
        raise ValueError("playback_speed_scale must be positive.")
    timestamps = np.asarray(source_timestamps_s, dtype=float)
    if len(timestamps) <= 1:
        return timestamps.copy()
    dt = np.diff(timestamps, prepend=timestamps[0])
    dt[0] = dt[1]
    dt = np.maximum(dt / playback_speed_scale, 1e-3)
    if max_joint_velocity_deg_s is not None and max_joint_velocity_deg_s > 0.0:
        dq = np.abs(np.diff(joint_positions, axis=0, prepend=joint_positions[:1]))
        dt = np.maximum(dt, np.max(dq / max_joint_velocity_deg_s, axis=1))
    replay_timestamps = np.cumsum(dt)
    replay_timestamps -= replay_timestamps[0]
    return replay_timestamps
