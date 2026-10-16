"""Microduck with Feetech HD-1910-C001 servos instead of Dynamixel XL330: robot cfgs.

The HD-1910 has the XL330 footprint, so the body is the stock Microduck. The
model (robot/microduck_hd1910/robot_walk_hd1910.xml) is generated from
robot/microduck/robot_walk.xml by bipedal-leg-bot/tools/hd1910.py: +3 g per
servo (21 g vs 18 g, all 15 incl. the mouth servo) and an `hd1910_actuator`
XML-PD class. The backlash twin is that file run through
microduck/add_backlash.py --backlash-deg 2.0 --joint-class hd1910_actuator.

The actuator is the provisional HD-1910 BAM model (actuator/params/
hd1910_provisional_m6.json): kt/R derived from the datasheet, XL330 friction and
armature as placeholders, XL330 voltage control law with HD-1910 firmware limits.
It is NOT a fit: replace it with bam.fit on a real servo before trusting sim2real.
"""

import os
from pathlib import Path

import mujoco
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from mjlab_microduck.actuator import (
    HD1910BacklashEncoderBamActuatorCfg,
    HD1910FrictionDRBamActuatorCfg,
)
from mjlab_microduck.robot.microduck_constants import (
    _BAM_ACTUATOR_KWARGS,
    BACKLASH_HOME_FRAME,
    FULL_COLLISION,
    HOME_FRAME,
)

_ROBOT_DIR: Path = Path(os.path.dirname(__file__)) / "microduck_hd1910"
HD1910_WALK_XML: Path = _ROBOT_DIR / "robot_walk_hd1910.xml"
HD1910_WALK_BACKLASH_XML: Path = _ROBOT_DIR / "robot_walk_hd1910_backlash.xml"
HD1910_SCENE_XML: Path = _ROBOT_DIR / "scene_walk_hd1910.xml"  # viewer / CPU eval (keyframes)
HD1910_PARAMS_JSON: Path = (
    Path(os.path.dirname(__file__)).parent / "actuator" / "params" / "hd1910_provisional_m6.json"
)

for _p in (HD1910_WALK_XML, HD1910_WALK_BACKLASH_XML, HD1910_SCENE_XML, HD1910_PARAMS_JSON):
    assert _p.exists(), f"not found: {_p}"


def get_hd1910_walk_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(HD1910_WALK_XML))


def get_hd1910_walk_backlash_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(HD1910_WALK_BACKLASH_XML))


# Firmware gain giving Microduck's proven 0.55 N.m/rad stiffness at 7.4 V:
# kp_fw * error_gain * vin * kt / R = 0.55 with the XL330 error_gain (0.002877)
# and the HD-1910's kt/R = 0.62/3.75. Mapping this onto the HD-1910's own
# position-P register (factory 32, Kp scale 1/8) needs a measurement on hardware.
HD1910_KP_FW = 156.0

# Same DR as Microduck (battery 6.5-8.2 V, voltage sag, 3-6 step delay); only the
# params JSON and the firmware gain change. The HD-1910 is rated to 8.4 V, so a
# fully charged 2S pack sits right at its limit.
HD1910_BAM_ACTUATOR_KWARGS = {
    **{k: v for k, v in _BAM_ACTUATOR_KWARGS.items() if k not in ("motor_name", "model")},
    "json_path": str(HD1910_PARAMS_JSON),
    "kp_fw": HD1910_KP_FW,
}
hd1910_actuators = HD1910FrictionDRBamActuatorCfg(**HD1910_BAM_ACTUATOR_KWARGS)
hd1910_backlash_actuators = HD1910BacklashEncoderBamActuatorCfg(**HD1910_BAM_ACTUATOR_KWARGS)

HD1910_WALK_ROBOT_CFG = EntityCfg(
    spec_fn=get_hd1910_walk_spec,
    init_state=HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(hd1910_actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

HD1910_WALK_BACKLASH_ROBOT_CFG = EntityCfg(
    spec_fn=get_hd1910_walk_backlash_spec,
    init_state=BACKLASH_HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(hd1910_backlash_actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)
