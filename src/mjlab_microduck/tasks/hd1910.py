"""HD-1910 task variants — the same tasks on Feetech HD-1910-C001 servos.

``make_hd1910_variant(cfg, robot_cfg)`` swaps the robot of any microduck env cfg
for its HD-1910 twin (robot/microduck_hd1910_constants.py): the stock body with
HD-1910 servo masses, the provisional HD-1910 BAM actuator and the HD-1910 stand
pose (robot/hd1910_pose.py), then applies ``apply_hd1910_recipe``: everything
HD-1910-specific that is not the robot entity. Task ids append ``-HD1910``. The
obs/action contract (61D / 14) is the base task's, so policies keep the runtime's
contract.

Keep HD-1910 changes HERE, not in microduck_velocity_env_cfg.py: that file is
upstream's, and the overlay is what keeps merges cheap. Base templates share cfg
objects across make() calls (see tasks/backlash.py), so copy a term before
changing it.

The HD-1910 model is NOT a BAM fit (datasheet kt/R, XL330 friction): these tasks
answer "can this torque/speed budget walk", not "will this policy transfer".
The backlash twin goes through ``make_backlash_variant`` for the robot and obs
swap and then through ``make_hd1910_variant`` as well, so it gets the recipe
(see tasks/__init__.py).
"""

from copy import deepcopy
from dataclasses import replace

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import RewardTermCfg

from mjlab_microduck.robot.hd1910_pose import HD1910_SPAWN_Z
from mjlab_microduck.robot.microduck_hd1910_constants import HD1910_WALK_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import MicroduckRlCfg

# -- Slow-command fix (2026-10-16). hd1910_v1 ignored commands below ~0.15 m/s:
# the stock track_linear_velocity (std² 0.1) pays 90 % of its reward for standing
# still at 0.10 m/s, and slow straight commands were ~0.1 % of the training data.
# A fine xy term next to the stock one prices the small error; a slow-command
# bucket supplies the data. Everything else of the recipe is untouched so the
# comparison with v1 stays readable. See docs/hd1910.md.
FINE_LIN_VEL_WEIGHT = 1.0   # a task term, fixed from step 0 (no new curriculum boundary)
FINE_LIN_VEL_STD = 0.10     # the error we still care about: standing at 0.10 m/s scores exp(-1)
FINE_LIN_VEL_TAU_S = 0.5    # EMA ≈ two strides: prices the DC error, not the stride oscillation
SLOW_ENVS = 0.15            # share of envs that creep (0.05-0.15 m/s, one axis) each resample


def apply_hd1910_recipe(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
    """The HD-1910 overlay on a velocity-recipe cfg (idempotent)."""
    # The HD-1910 stand pose is ~9 mm taller than HOME: spawn so the soles land on
    # the floor, as the base's (0.12, 0.13) does for HOME. Copy the term first.
    reset = deepcopy(cfg.events["reset_base"])
    reset.params["pose_range"] = {**reset.params["pose_range"], "z": HD1910_SPAWN_Z}
    cfg.events["reset_base"] = reset
    # Slow-command fix: fine tracking term + slow-command bucket (the twist cfg is
    # fresh per factory call, but copy anyway: cheap, and leak-proof).
    cfg.rewards["track_linear_velocity_fine"] = RewardTermCfg(
        func=microduck_mdp.track_linear_velocity_fine,
        weight=FINE_LIN_VEL_WEIGHT,
        params={"std": FINE_LIN_VEL_STD, "command_name": "twist", "tau_s": FINE_LIN_VEL_TAU_S},
    )
    twist = deepcopy(cfg.commands["twist"])
    twist.rel_slow_envs = SLOW_ENVS
    cfg.commands["twist"] = twist
    return cfg


def make_hd1910_variant(
    cfg: ManagerBasedRlEnvCfg,
    robot_cfg: EntityCfg = HD1910_WALK_ROBOT_CFG,
) -> ManagerBasedRlEnvCfg:
    """Swap a microduck env cfg's robot for the HD-1910 one and apply the HD-1910 recipe."""
    cfg.scene.entities = {**cfg.scene.entities, "robot": robot_cfg}
    return apply_hd1910_recipe(cfg)


# Own log directory so HD-1910 runs never mix with XL330 ones.
MicroduckHD1910RlCfg = replace(MicroduckRlCfg, experiment_name="velocity_hd1910")
