"""MuJoCo model import test utility.

Usage examples:
  python mujoco_test.py --model path/to/model.xml
  python mujoco_test.py --model path/to/model.urdf --steps 100
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import time
import mujoco
import mujoco.viewer

def print_model_stats(model: mujoco.MjModel) -> None:
    print("Model loaded successfully")
    print(f"  nbody:   {model.nbody}")
    print(f"  njnt:    {model.njnt}")
    print(f"  ngeom:   {model.ngeom}")
    print(f"  nsite:   {model.nsite}")
    print(f"  nq:      {model.nq}")
    print(f"  nv:      {model.nv}")
    print(f"  nu:      {model.nu}")
    print(f"  nact:    {model.na}")


def main() -> int:
    # args = parse_args()
    model_path = Path(__file__).resolve().parent / "go_b.xml"
    os.chdir(model_path.parent)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    model.opt.timestep = float(0.02)
    print_model_stats(model)
    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            # time.sleep(model.opt.timestep)
            time.sleep(10000)
    print("Simulation steps completed")
    return 0


if __name__ == "__main__":
	raise SystemExit(main())
