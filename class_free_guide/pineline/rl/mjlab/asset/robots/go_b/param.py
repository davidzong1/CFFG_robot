"""Unitree GO_B constants."""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator, reflected_inertia
from mjlab.utils.spec_config import CollisionCfg
from class_free_guide import ROOT_PATH

##
# MJCF and assets.
##

GO_B_XML: Path = ROOT_PATH / "robots" / "go_b" / "urdf" / "go_b.xml"
assert GO_B_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    assets_dir = GO_B_XML.parent / "assets"
    if assets_dir.exists():
        for filename in assets_dir.rglob("*"):
            if filename.is_file():
                rel_path = str(filename.relative_to(assets_dir))
                with open(filename, "rb") as f:
                    assets[rel_path] = f.read()
    return assets


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(GO_B_XML))
    spec.assets = get_assets(spec.meshdir)
    return spec


##
# Actuator config.
##

GO_B_ACTUATOR_HIP = BuiltinPositionActuatorCfg(
    target_names_expr=(".*hip_.*",),
    stiffness=20.0,  # KP
    damping=1.0,  # KD
    effort_limit=23.5,
    armature=0.01,
)
GO_B_ACTUATOR_THIGH = BuiltinPositionActuatorCfg(
    target_names_expr=(".*thigh_.*",),
    stiffness=20.0,  # KP
    damping=1.0,  # KD
    effort_limit=23.5,
    armature=0.01,
)
GO_B_ACTUATOR_CALF = BuiltinPositionActuatorCfg(
    target_names_expr=(".*calf_.*",),
    stiffness=40.0,  # KP
    damping=2.0,  # KD
    effort_limit=45,
    armature=0.02,
)

GO_B_ACTUATOR_ARM = BuiltinPositionActuatorCfg(
    target_names_expr=("arm_.*",),
    stiffness=10.0,  # KP
    damping=0.5,  # KD
    effort_limit=15.0,
    armature=0.005,
)

##
# Keyframes.
##

INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.32),
    joint_pos={
        ".*thigh_joint": 0.9,
        ".*calf_joint": -1.8,
        ".*R_hip_joint": 0.1,
        ".*L_hip_joint": -0.1,
        "arm_.*": 0.0,
    },
    joint_vel={".*": 0.0},
)


##
# Collision config.
##

_foot_regex = "^[FR][LR]_foot_collision$"

# This disables all collisions except the feet.
# Furthermore, feet self collisions are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
    geom_names_expr=(_foot_regex,),
    contype=0,
    conaffinity=1,
    condim=3,
    priority=1,
    friction=(0.6,),
    solimp=(0.9, 0.95, 0.023),
)

# This enables all collisions, excluding self collisions.
# Foot collisions are given custom condim, friction and solimp.
FULL_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",),
    condim={_foot_regex: 3, ".*_collision": 1},
    priority={_foot_regex: 1},
    friction={_foot_regex: (0.6,)},
    solimp={_foot_regex: (0.9, 0.95, 0.023)},
    contype=1,
    conaffinity=0,
)

##
# Final config.
##

GO_B_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        GO_B_ACTUATOR_HIP,
        GO_B_ACTUATOR_THIGH,
        GO_B_ACTUATOR_CALF,
        GO_B_ACTUATOR_ARM,
    ),
    soft_joint_pos_limit_factor=0.9,  # soft limit
)


def get_go_b_robot_cfg() -> EntityCfg:
    """Get a fresh GO_B robot configuration instance.

    Returns a new EntityCfg instance each time to avoid mutation issues when
    the config is shared across multiple places.
    """
    return EntityCfg(
        init_state=INIT_STATE,
        collisions=(FULL_COLLISION,),
        spec_fn=get_spec,
        articulation=GO_B_ARTICULATION,
    )


if __name__ == "__main__":
    import mujoco.viewer as viewer

    from mjlab.entity.entity import Entity

    robot = Entity(get_go_b_robot_cfg())

    viewer.launch(robot.spec.compile())
