"""HD-1910 task variants — the same tasks on Feetech HD-1910-C001 servos.

``make_hd1910_variant(cfg, robot_cfg)`` swaps the robot of any microduck env cfg
for its HD-1910 twin (robot/microduck_hd1910_constants.py): the stock body with
HD-1910 servo masses and the provisional HD-1910 BAM actuator. Task ids append
``-HD1910``. Everything else (obs 61D, commands, rewards, DR, curricula) is the
base task's, so policies keep the runtime's obs/action contract.

The HD-1910 model is NOT a BAM fit (datasheet kt/R, XL330 friction): these tasks
answer "can this torque/speed budget walk", not "will this policy transfer".
The backlash twin needs no HD-1910 step: ``make_backlash_variant`` already swaps
the robot, so pass it HD1910_WALK_BACKLASH_ROBOT_CFG (see tasks/__init__.py).
"""

from dataclasses import replace

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg

from mjlab_microduck.robot.microduck_hd1910_constants import HD1910_WALK_ROBOT_CFG
from mjlab_microduck.tasks.microduck_velocity_env_cfg import MicroduckRlCfg


def make_hd1910_variant(
    cfg: ManagerBasedRlEnvCfg,
    robot_cfg: EntityCfg = HD1910_WALK_ROBOT_CFG,
) -> ManagerBasedRlEnvCfg:
    """Swap a microduck env cfg's robot for the HD-1910 one."""
    cfg.scene.entities = {**cfg.scene.entities, "robot": robot_cfg}
    return cfg


# Own log directory so HD-1910 runs never mix with XL330 ones.
MicroduckHD1910RlCfg = replace(MicroduckRlCfg, experiment_name="velocity_hd1910")
