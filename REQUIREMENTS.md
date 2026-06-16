# Requirements

This project retargets robot end-effector trajectories across URDF-described
arms. It needs Python plus LeRobot's kinematics stack for FK/IK.

## System Requirements

- Python 3.10 or newer
- Git
- A compiler/toolchain suitable for Python packages with native extensions
- Linux or macOS recommended

## Python Dependencies

Core dependencies are declared in `pyproject.toml`:

- `numpy>=1.23`
- `lerobot[kinematics]` from the pinned Seeed LeRobot commit

Rendering dependencies are optional:

- `imageio>=2.30`
- `matplotlib>=3.7`

The LeRobot kinematics backend uses `placo`. Installing
`lerobot[kinematics]` should pull the required kinematics dependencies.

## Input Data Requirements

For retargeting from joint trajectories:

- Source trajectory file: `.npz`, `.npy`, `.pkl`, or `.pickle`
- Source URDF
- Source TCP link name
- Source joint names
- Target URDF
- Target TCP link name
- Target joint names

For retargeting from existing TCP pose trajectories:

- Source pose trajectory file
- Pose array key in the file
- Target URDF
- Target TCP link name
- Target joint names

## Included Assets

The repository includes example URDF/mesh assets for validation and rendering:

- B601-style reBot arm
- Piper
- Panda
- Kuka iiwa
- Yam

The checked-in validation GIFs are the final generated visual results. Generated
`.npz` files, logs, caches, and intermediate images are intentionally ignored.

## Optional Validation Tools

The validation scripts may also require:

- `mujoco`
- `Pillow`

These are only needed for regenerating the physics/render validation outputs,
not for normal trajectory retargeting.
