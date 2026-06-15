#!/usr/bin/env python3

"""Print the canonical TCP frame an arm's URDF resolves to.

Use this to sanity-check cross-arm orientation transfer before retargeting:
the canonical z should point out of the gripper and the canonical y should lie
along the wrist-pitch axis. If two arms disagree on what "out of the gripper"
means, their printed approach axes will reveal it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from robot_trajectory_retargeting.urdf_chain import compute_canonical_tcp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect the canonical TCP frame derived from a URDF.")
    parser.add_argument("--urdf", required=True, help="Path to the URDF file.")
    parser.add_argument("--tcp-link", required=True, help="TCP/end-effector link to analyze.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = compute_canonical_tcp(args.urdf, args.tcp_link)
    payload = {
        "urdf": str(Path(args.urdf).expanduser().resolve()),
        "tcp_link": args.tcp_link,
        "native_tcp_to_canonical_rotation": [[float(v) for v in row] for row in result.rotation],
        "approach_axis_base": [float(v) for v in result.approach_axis_base],
        "pitch_axis_base": [float(v) for v in result.pitch_axis_base],
        "pitch_axis_source": result.pitch_axis_source,
        "tcp_position_base": [float(v) for v in result.tcp_position_base],
        "mount_offset_base": [float(v) for v in result.mount_offset_base],
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
