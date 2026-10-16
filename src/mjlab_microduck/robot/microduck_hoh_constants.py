"""Head-on-hips Microduck with Feetech HD-1910 servos: robot cfgs (side variant, no task registered).

The model (robot/microduck_hoh/robot_walk_hoh.xml) is generated from
robot/microduck/robot_walk.xml by bipedal-leg-bot/tools/microduck_head_on_hips.py:
the neck chain and its 4 servos are removed and the head is mounted rigidly on
trunk_base, leaving 10 leg servos (left leg 0-4, right leg 5-9). The backlash
twin is that file run through microduck/add_backlash.py --joint-class hd1910_actuator.

Same provisional HD-1910 actuator as microduck_hd1910_constants. The 10-joint
obs/command layout would need its own task variant (no head_pose command, 45-D
actor obs); only the full 15-servo Microduck is registered for now.
"""

import os
from pathlib import Path

import mujoco
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from mjlab_microduck.robot.microduck_constants import FULL_COLLISION, HOME_FRAME
from mjlab_microduck.robot.microduck_hd1910_constants import (
    hd1910_actuators,
    hd1910_backlash_actuators,
)

_ROBOT_DIR: Path = Path(os.path.dirname(__file__)) / "microduck_hoh"
HOH_WALK_XML: Path = _ROBOT_DIR / "robot_walk_hoh.xml"
HOH_WALK_BACKLASH_XML: Path = _ROBOT_DIR / "robot_walk_hoh_backlash.xml"
HOH_SCENE_XML: Path = _ROBOT_DIR / "scene_walk_hoh.xml"  # viewer / CPU eval (STAND keyframe)

for _p in (HOH_WALK_XML, HOH_WALK_BACKLASH_XML, HOH_SCENE_XML):
    assert _p.exists(), f"not found: {_p}"

HOH_JOINT_NAMES = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
)


def get_hoh_walk_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(HOH_WALK_XML))


def get_hoh_walk_backlash_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(HOH_WALK_BACKLASH_XML))


# Microduck's STAND2 leg pose (HOME_FRAME) without the neck/head entries.
HOH_HOME_FRAME = EntityCfg.InitialStateCfg(
    joint_pos={k: v for k, v in HOME_FRAME.joint_pos.items() if "neck" not in k and "head" not in k},
    joint_vel={".*": 0.0},
)
HOH_BACKLASH_HOME_FRAME = EntityCfg.InitialStateCfg(
    joint_pos={r".*_backlash$": 0.0, **HOH_HOME_FRAME.joint_pos},
    joint_vel={".*": 0.0},
)

HOH_WALK_ROBOT_CFG = EntityCfg(
    spec_fn=get_hoh_walk_spec,
    init_state=HOH_HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(hd1910_actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

HOH_WALK_BACKLASH_ROBOT_CFG = EntityCfg(
    spec_fn=get_hoh_walk_backlash_spec,
    init_state=HOH_BACKLASH_HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(hd1910_backlash_actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)
