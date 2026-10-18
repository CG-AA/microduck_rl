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
from mjlab.managers import CurriculumTermCfg, EventTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

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
#
# v2 (weight 1.0) walked at 0.10 m/s at iteration 1000 and had folded it back
# into standing by 2000, when the standing share reached 25 % and the
# action_rate tax -1.0; v2b doubles the term's margin instead of touching the
# smoothness tax (a sim2real property we cannot measure yet).
FINE_LIN_VEL_WEIGHT = 2.0   # a task term, fixed from step 0 (no new curriculum boundary)
FINE_LIN_VEL_STD = 0.10     # the error we still care about: standing at 0.10 m/s scores exp(-1)
FINE_LIN_VEL_TAU_S = 0.5    # EMA ≈ two strides: prices the DC error, not the stride oscillation
SLOW_ENVS = 0.15            # share of envs that creep (0.05-0.15 m/s, one axis) each resample
# -- Standing torque (v2b). With nothing pricing torque, v2 braced while standing:
# hip rolls commanded 0.22 rad inward, knees 0.14 rad, feet pinned by friction,
# 0.12 / 0.08 N.m per servo for no motion. Bracing's sum tau² over the legs is
# ~0.05, so this weight makes it cost ~0.1/step against ~4/step of standing
# reward: enough to drop a habit that buys nothing, not enough to forbid torque
# where a push needs it. Gated on zero command, so the gait is untaxed.
STAND_TORQUE_WEIGHT = -2.0
# v2c raises it tenfold (staged, see below): at -2.0 it logged -0.03 to -0.05, ~1 %
# of the standing reward, and bracing survived in two of v2b's three checkpoints.
STAND_TORQUE_WEIGHT_FINAL = -20.0
# -- v2c: the inherited mjlab rel_forward_envs = 0.2 clamps a fifth of the commands
# to vx >= 0.3 m/s. It shifted the vx normalizer mean to +0.04, which put +0.10 m/s
# nearest the standing cluster and made forward the last slow axis to come back.
# Uniform sampling still gives 0.3-0.4 m/s to 12.5 % of envs.
FORWARD_ENVS = 0.2  # mjlab's value, kept while the gait is discovered
FORWARD_ENVS_FINAL = 0.0
# -- v2c: servo DR for a servo that has not been measured. Friction and rotor
# inertia are borrowed from the XL330, and the P-register -> duty/rad mapping is
# unknown (and coarse: P ~ 3 if it scales like the STS3215). The sensitivity
# matrix (scripts/sensitivity.py, docs/hd1910.md) found the gain the dominant
# axis (v2b falls at x0.7, cannot stand at x4) and friction second (x2 fails).
# Ranges by plausibility, narrowed after the bench fit. Training's stock ranges
# are (0.9, 1.1) for both and no gain randomization.
FRICTION_SCALE_RANGE = (0.5, 1.5)
ARMATURE_SCALE_RANGE = (0.5, 1.5)
KP_SCALE_RANGE = (0.7, 1.4)
# -- v2c staging (2026-10-17). With all three changes active from step 0, seed 42
# had not found a gait by iteration 1000 (0.20 m/s -> 0.06, no turning; v2b walked
# and turned by then). Per AGENTS.md (introduce taxes after the skill exists;
# phase-align stages), the run now trains exactly as v2b until the gait exists,
# then widens the servo DR in two steps and drops the forward-only bucket, and
# prices standing torque at full weight only after the standing curriculum
# (25 % at 2000) has settled. Stages sit between the base recipe's boundaries
# (500/750/1000/1250/1500/2000). Iterations (x NUM_STEPS_PER_ENV in the cfg).
SERVO_DR_STAGES = [
    (0, dict(friction=(0.9, 1.1), armature=(0.9, 1.1), kp=(1.0, 1.0))),  # = v2b
    (600, dict(friction=(0.7, 1.3), armature=(0.7, 1.3), kp=(0.85, 1.2))),
    (1200, dict(friction=FRICTION_SCALE_RANGE, armature=ARMATURE_SCALE_RANGE, kp=KP_SCALE_RANGE)),
]
FORWARD_ENVS_STAGES = [(0, FORWARD_ENVS), (1200, FORWARD_ENVS_FINAL)]
STAND_TORQUE_STAGES = [(0, STAND_TORQUE_WEIGHT), (2500, STAND_TORQUE_WEIGHT_FINAL)]
NUM_STEPS_PER_ENV = 24  # curriculum steps are env steps: iteration x 24
# -- v2d (2026-10-18): turning in place from a standstill. Every run turned from rest
# at iteration 1000 (v2b@1000: +0.84 / -0.46 / +0.51 rad/s at +1.0 / -1.0 / +0.5) and
# folded it into standing by 2000, when the standing share ramps to 25 %; v2b@5500
# never turns from rest (harness v3). The stock track_angular_velocity (std 0.71)
# keeps 61 % of its reward for standing still at a 0.5 rad/s command. Same fix as the
# linear one: velstand's fine yaw term (std 0.35, weight 2) ADDED from step 0, on a
# 0.5 s EMA so the per-step yaw oscillation of a stepping turn is not priced. Standing
# at 0.5 rad/s then scores 0.13 instead of 1.0 on it: +1.7/step for turning.
YAW_FINE_WEIGHT = 2.0
YAW_FINE_STD = 0.35
YAW_FINE_TAU_S = 0.5


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
    cfg.rewards["track_yaw_rate_fine"] = RewardTermCfg(
        func=microduck_mdp.track_yaw_rate_fine,
        weight=YAW_FINE_WEIGHT,
        params={"std": YAW_FINE_STD, "command_name": "twist", "tau_s": YAW_FINE_TAU_S},
    )
    twist = deepcopy(cfg.commands["twist"])
    twist.rel_slow_envs = SLOW_ENVS
    twist.rel_forward_envs = FORWARD_ENVS
    cfg.commands["twist"] = twist
    # Standing torque: the legs' sum tau² at zero command (>= 0, negative weight).
    cfg.rewards["stand_torque"] = RewardTermCfg(
        func=microduck_mdp.joint_torques_l2_when_standing,
        weight=STAND_TORQUE_WEIGHT,
        params={"command_name": "twist"},
    )
    # Servo DR (v2c): the events start at stage 0 (= v2b) and the curricula widen them.
    dr0 = SERVO_DR_STAGES[0][1]
    friction = deepcopy(cfg.events["randomize_joint_friction"])
    friction.params["scale_range"] = dr0["friction"]
    cfg.events["randomize_joint_friction"] = friction
    armature = deepcopy(cfg.events["randomize_armature"])
    armature.params["ranges"] = dr0["armature"]
    cfg.events["randomize_armature"] = armature
    cfg.events["randomize_motor_gains"] = EventTermCfg(
        func=microduck_mdp.randomize_bam_gain_scale,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot"), "kp_range": dr0["kp"], "kd_range": (1.0, 1.0)},
    )
    steps = lambda it: it * NUM_STEPS_PER_ENV
    for event, key, param in (("randomize_joint_friction", "friction", "scale_range"),
                              ("randomize_armature", "armature", "ranges"),
                              ("randomize_motor_gains", "kp", "kp_range")):
        cfg.curriculum[f"{event}_stages"] = CurriculumTermCfg(
            func=microduck_mdp.event_params_curriculum,
            params={"event_name": event,
                    "stages": [{"step": steps(it), "params": {param: d[key]}} for it, d in SERVO_DR_STAGES]},
        )
    cfg.curriculum["forward_envs_stages"] = CurriculumTermCfg(
        func=microduck_mdp.command_attr_curriculum,
        params={"command_name": "twist", "attr": "rel_forward_envs",
                "stages": [{"step": steps(it), "value": v} for it, v in FORWARD_ENVS_STAGES]},
    )
    cfg.curriculum["stand_torque_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={"reward_name": "stand_torque",
                "weight_stages": [{"step": steps(it), "weight": w} for it, w in STAND_TORQUE_STAGES]},
    )
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
