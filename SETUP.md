# Setup

Follow these steps from the repository root.

## 1. Create An Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Conda is also fine:

```bash
conda create -n dof-retargetting python=3.10 -y
conda activate dof-retargetting
python -m pip install --upgrade pip
```

## 2. Install The Package

For retargeting only:

```bash
python -m pip install -e .
```

For retargeting plus GIF rendering:

```bash
python -m pip install -e ".[render]"
```

## 3. Verify The Install

```bash
python tools/retarget_trajectory.py --help
python tools/inspect_tcp_convention.py --help
python tools/render_urdf_trajectory_gif.py --help
```

If these commands print help text, the package entry points and imports are
available.

## 4. Inspect TCP Conventions

Before retargeting a new robot pair, inspect each robot's TCP convention:

```bash
python tools/inspect_tcp_convention.py \
  --urdf /path/to/source_robot.urdf \
  --tcp-link /source/tcp/link

python tools/inspect_tcp_convention.py \
  --urdf /path/to/target_robot.urdf \
  --tcp-link /target/tcp/link
```

The reported approach axis should point out of the gripper.

## 5. Run A Retarget

```bash
python tools/retarget_trajectory.py \
  --input /path/to/source_episode.npz \
  --output outputs/source_to_target.npz \
  --source-mode joints \
  --source-urdf /path/to/source_robot.urdf \
  --source-tcp-link gripper_frame_link \
  --source-joint-names shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll \
  --input-data-key joint_positions_rad \
  --input-joint-slice 0:5 \
  --input-time-key timestamps_s \
  --target-urdf /path/to/target_robot.urdf \
  --target-tcp-link gripper_tcp \
  --target-joint-names joint1,joint2,joint3,joint4,joint5,joint6 \
  --target-output-joint-names shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_yaw,wrist_roll
```

The output `.npz` contains the desired target poses, solved target joint
trajectory, achieved target poses, and tracking errors.

## 6. Render A GIF

```bash
python tools/render_urdf_trajectory_gif.py \
  --trajectory outputs/source_to_target.npz \
  --urdf /path/to/target_robot.urdf \
  --target-link gripper_tcp \
  --joint-data-key smoothed_target_joint_positions_deg \
  --joint-units deg \
  --joint-names joint1,joint2,joint3,joint4,joint5,joint6 \
  --desired-poses-key desired_target_poses \
  --time-key replay_timestamps_s \
  --output outputs/source_to_target.gif
```

## Notes

- Keep generated outputs under `outputs/` or `validation/outputs/`; most
  generated files are ignored by Git.
- The checked-in validation GIFs under `validation/outputs/` are the curated
  result artifacts for GitHub.
- If IK struggles with a target wrist, reduce `--orientation-weight` toward
  `0.0` for position-only matching.
