"""HD-1910 variant: provisional actuator numbers, firmware limits, model and task wiring.

The HD-1910 BAM params are derived from the datasheet (reference: 6 V stall
10-12 kg.cm, 1.6 A stall current, 0.109 s/60 deg no-load) — these tests pin the
derivation so an edit to the JSON or the firmware limits can't silently drift
from the spec, and check that the registered tasks really run the HD-1910.
"""

import math

import mujoco
import numpy as np
import pytest
import torch

import mjlab_microduck.tasks  # noqa: F401  (load tasks first: avoids the constants<->tasks import cycle)
from bam.actuator import TorchBackend
from bam.model import load_model
from mjlab_microduck.robot import hd1910_pose
from mjlab_microduck.robot import microduck_hd1910_constants as hd
from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_XML

SPEC_STALL_6V = 0.99  # N.m, conservative 10 kg.cm figure
SPEC_STALL_CURRENT_6V = 1.6  # A
SPEC_NO_LOAD_6V = math.radians(60) / 0.109  # rad/s
XL330_STIFFNESS = 0.55  # N.m/rad, Microduck's XML-PD kp at kp_fw 200 / 7.4 V


@pytest.fixture(scope="module")
def act():
    """HD-1910 actuator exactly as training builds it (limits applied by the subclass)."""
    a = hd.hd1910_actuators.build(None, list(range(14)), [f"j{i}" for i in range(14)])
    bam = a._bam_model
    bam.actuator.backend = TorchBackend()  # what BamActuator.initialize does
    return bam


def torque(bam, vin, err, dq):
    """Motor torque on an (N=2, J=14) batch for position error err [rad] and velocity dq [rad/s]."""
    shape = (2, 14)
    act = bam.actuator
    act.vin = torch.full((2, 1), vin)
    q = torch.zeros(shape)
    ctrl = act.compute_control(q + err, q, torch.full(shape, dq), 0.02)
    tau = act.compute_torque(ctrl, True, q, torch.full(shape, dq))
    assert tau.shape == shape
    return tau, ctrl


def test_firmware_limits_are_hd1910_not_xl330(act):
    assert act.actuator.max_current is None  # Feetech protection current unloads, it does not clamp
    assert act.actuator.max_pwm == pytest.approx(0.98)  # reg 16 max torque 980 x 0.1 %
    assert act.actuator.kp == hd.HD1910_KP_FW
    assert act.kt.value == pytest.approx(0.62) and act.R.value == pytest.approx(3.75)


@pytest.mark.parametrize("vin,scale", [(6.0, 1.0), (7.4, 7.4 / 6.0)])
def test_stall_and_no_load_match_datasheet(act, vin, scale):
    # duty = err * kp_fw * error_gain = 0.45/rad: it only saturates past ~2.2 rad of error
    tau, ctrl = torque(act, vin, err=10.0, dq=0.0)
    assert tau.abs().max().item() == pytest.approx(SPEC_STALL_6V * scale, rel=0.05)
    current = ctrl / act.R.value
    assert current.max().item() == pytest.approx(SPEC_STALL_CURRENT_6V * scale, rel=0.05)
    no_load = 0.98 * vin / act.kt.value  # torque(dq) = 0
    assert torque(act, vin, err=10.0, dq=no_load)[0].abs().max().item() < 1e-6
    assert no_load == pytest.approx(SPEC_NO_LOAD_6V * scale, rel=0.05)


def test_stiffness_matches_microduck(act):
    tau, _ = torque(act, 7.4, err=0.01, dq=0.0)  # small error: linear region
    assert tau.max().item() / 0.01 == pytest.approx(XL330_STIFFNESS, rel=0.02)


def test_model_is_stock_body_with_hd1910_masses():
    stock = mujoco.MjModel.from_xml_path(str(MICRODUCK_WALK_XML))
    m = mujoco.MjModel.from_xml_path(str(hd.HD1910_WALK_XML))
    assert m.nu == stock.nu == 14
    assert [m.actuator(i).name for i in range(m.nu)] == [stock.actuator(i).name for i in range(stock.nu)]
    assert m.nbody == stock.nbody and m.njnt == stock.njnt
    assert m.body_subtreemass[1] - stock.body_subtreemass[1] == pytest.approx(15 * 0.003, abs=1e-6)
    np.testing.assert_allclose(m.jnt_range, stock.jnt_range)
    bl = mujoco.MjModel.from_xml_path(str(hd.HD1910_WALK_BACKLASH_XML))
    assert bl.nu == 14 and bl.njnt == stock.njnt + 14
    assert bl.body_subtreemass[1] == pytest.approx(m.body_subtreemass[1])


@pytest.mark.parametrize("task_id,robot_cfg", [
    ("Mjlab-Velocity-Flat-MicroDuck-HD1910", hd.HD1910_WALK_ROBOT_CFG),
    ("Mjlab-Velocity-Flat-Backlash-MicroDuck-HD1910", hd.HD1910_WALK_BACKLASH_ROBOT_CFG),
])
def test_registered_tasks_use_hd1910(task_id, robot_cfg):
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    for play in (False, True):
        robot = load_env_cfg(task_id, play=play).scene.entities["robot"]  # the registry deep-copies
        assert robot.spec_fn is robot_cfg.spec_fn
        assert robot.init_state == robot_cfg.init_state
        (actuator,) = robot.articulation.actuators
        assert type(actuator) is type(robot_cfg.articulation.actuators[0])
        assert actuator.json_path == str(hd.HD1910_PARAMS_JSON)
        assert actuator.kp_fw == hd.HD1910_KP_FW
    assert load_rl_cfg(task_id).experiment_name == "velocity_hd1910"


def test_cpu_rehearsal_mirrors_hd1910_training_cfg():
    """scripts/infer_policy.py --hd1910 must build the actuator the -HD1910 tasks train with."""
    import importlib.util
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("infer_policy", repo / "scripts" / "infer_policy.py")
    ip = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ip)

    cfg = hd.hd1910_actuators
    assert (repo / ip.HD1910_PARAMS_JSON).resolve() == hd.HD1910_PARAMS_JSON.resolve()
    assert (repo / ip.HD1910_XML).resolve() == hd.HD1910_SCENE_XML.resolve()
    assert ip.HD1910_KP_FW == cfg.kp_fw
    assert ip.HD1910_MAX_PWM == cfg.max_pwm
    assert ip.HD1910_MAX_CURRENT == cfg.max_current
    bam = ip.load_bam_model(ip.HD1910_KP_FW, 7.4, ip.HD1910_MAX_CURRENT,
                            json_path=str(repo / ip.HD1910_PARAMS_JSON), max_pwm=ip.HD1910_MAX_PWM)
    assert bam.kt.value == pytest.approx(0.62) and bam.actuator.max_pwm == pytest.approx(0.98)
    assert bam.actuator.max_current is None
    assert np.allclose(ip.HD1910_DEFAULT_POSE, hd1910_pose.HD1910_DEFAULT_POSE)


def test_hd1910_recipe_sets_spawn_height_without_touching_xl330_tasks():
    """The overlay (tasks/hd1910.py) spawns the taller stand pose on its soles; the base
    velocity task must keep HOME's (0.12, 0.13): base templates share cfg objects across
    make() calls, so a careless overlay would leak into the XL330 tasks."""
    from mjlab.tasks.registry import load_env_cfg

    z = lambda task: tuple(load_env_cfg(task).events["reset_base"].params["pose_range"]["z"])
    assert z("Mjlab-Velocity-Flat-MicroDuck-HD1910") == tuple(hd1910_pose.HD1910_SPAWN_Z)
    assert z("Mjlab-Velocity-Flat-Backlash-MicroDuck-HD1910") == tuple(hd1910_pose.HD1910_SPAWN_Z)
    assert z("Mjlab-Velocity-Flat-MicroDuck") == (0.12, 0.13)
    lo, hi = hd1910_pose.HD1910_SPAWN_Z
    assert hd1910_pose.HD1910_TRUNK_Z_CONTACT < lo < hi < hd1910_pose.HD1910_TRUNK_Z_CONTACT + 0.02


def test_hd1910_recipe_adds_fine_tracking_and_slow_bucket():
    """The slow-command fix lives in the overlay: both HD-1910 tasks carry it, the XL330 task
    does not, and the stock tracking term is untouched (the fine term is ADDED, not a swap)."""
    from mjlab.tasks.registry import load_env_cfg
    from mjlab_microduck.tasks import hd1910
    from mjlab_microduck.tasks import mdp as microduck_mdp

    for task in ("Mjlab-Velocity-Flat-MicroDuck-HD1910", "Mjlab-Velocity-Flat-Backlash-MicroDuck-HD1910"):
        cfg = load_env_cfg(task)
        fine = cfg.rewards["track_linear_velocity_fine"]
        assert fine.func is microduck_mdp.track_linear_velocity_fine
        assert fine.weight == hd1910.FINE_LIN_VEL_WEIGHT > 0
        assert fine.params == {"std": hd1910.FINE_LIN_VEL_STD, "command_name": "twist", "tau_s": hd1910.FINE_LIN_VEL_TAU_S}
        assert cfg.rewards["track_linear_velocity"].weight == 2.0
        assert cfg.rewards["track_linear_velocity"].params["std"] == pytest.approx(math.sqrt(0.1))
        tw = cfg.commands["twist"]
        assert tw.rel_slow_envs == hd1910.SLOW_ENVS == 0.15
        assert tuple(tw.slow_speed_range) == (0.05, 0.15) and tw.slow_lateral_frac == 0.25
        assert tw.rel_turn_in_place_envs > 0 and tw.rel_standing_envs > 0  # the other buckets stay
        stand = cfg.rewards["stand_torque"]
        assert stand.func is microduck_mdp.joint_torques_l2_when_standing
        assert stand.weight == hd1910.STAND_TORQUE_WEIGHT < 0  # a >= 0 cost needs a negative weight
    xl = load_env_cfg("Mjlab-Velocity-Flat-MicroDuck")
    assert "track_linear_velocity_fine" not in xl.rewards and "stand_torque" not in xl.rewards
    assert xl.commands["twist"].rel_slow_envs == 0.0


def test_joint_torques_l2_when_standing_prices_only_the_legs_at_zero_command():
    from types import SimpleNamespace
    from mjlab_microduck.tasks.mdp import joint_torques_l2_when_standing

    tau = torch.zeros(3, 14)
    tau[0, [1, 10]] = 0.12   # bracing hip rolls, standing
    tau[0, [3, 12]] = 0.08   # bracing knees
    tau[0, [5, 6]] = 0.5     # head torque must not count
    tau[1, [1, 10]] = 0.12   # same torques, but walking
    cmd = torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.0, 0.0]])
    env = SimpleNamespace(scene={"robot": SimpleNamespace(data=SimpleNamespace(actuator_force=tau))},
                          command_manager=SimpleNamespace(get_command=lambda name: cmd))
    cost = joint_torques_l2_when_standing(env)
    assert cost[0] == pytest.approx(2 * 0.12 ** 2 + 2 * 0.08 ** 2)
    assert cost[1] == 0.0 and cost[2] == 0.0
    assert (cost >= 0).all()


def test_track_linear_velocity_fine_prices_small_errors_through_an_ema():
    """exp(-|cmd - EMA(v)|²/std²): matching the command scores 1, standing at 0.10 scores
    exp(-1), the EMA (tau_s) ignores the stride oscillation, and a reset restarts it."""
    from types import SimpleNamespace
    from mjlab_microduck.tasks.mdp import track_linear_velocity_fine

    n, std, tau_s, dt = 4, 0.10, 0.5, 0.02
    cmd = torch.tensor([[0.10, 0.0, 0.0], [0.10, 0.0, 0.0], [0.0, 0.10, 0.0], [0.20, 0.0, 0.0]])
    data = SimpleNamespace(root_link_lin_vel_b=torch.zeros(n, 3))
    env = SimpleNamespace(scene={"robot": SimpleNamespace(data=data)},
                          command_manager=SimpleNamespace(get_command=lambda name: cmd),
                          episode_length_buf=torch.full((n,), 10), step_dt=dt)
    # Steady state: env 0 tracks, env 1 stands, env 2 tracks sideways, env 3 stands at 0.20.
    for k in range(200):
        osc = 0.05 * math.sin(2 * math.pi * k * dt / 0.25)  # stride-cycle oscillation, 4 Hz
        data.root_link_lin_vel_b = torch.tensor([[0.10 + osc, 0.0, 0.0], [0.0, 0.0, 0.0],
                                                 [0.0, 0.10 + osc, 0.0], [0.0, 0.0, 0.0]])
        r = track_linear_velocity_fine(env, std, "twist", tau_s)
    assert r.shape == (n,)
    assert r[0] > 0.95 and r[2] > 0.95, "the EMA must cancel the stride oscillation"
    assert r[1] == pytest.approx(math.exp(-1.0), abs=0.02)
    assert r[3] == pytest.approx(math.exp(-4.0), abs=0.01)
    # A reset env restarts the average from its measured velocity, not from the old one.
    env.episode_length_buf[0] = 0
    data.root_link_lin_vel_b = torch.zeros(n, 3)
    r = track_linear_velocity_fine(env, std, "twist", tau_s)
    assert r[0] == pytest.approx(math.exp(-1.0), abs=0.02)


def test_hd1910_v2c_levers_and_servo_dr_are_staged():
    """v2c: the HD-1910 tasks start exactly as v2b (stock servo DR, no gain DR, forward bucket 0.2,
    stand_torque -2) and curricula bring in the wide servo DR (600 / 1200), drop the forward-only
    bucket (1200) and raise stand_torque to -20 at 2500 (v2e's -8 was tried and rejected). The XL330 task gets none of it."""
    from mjlab.tasks.registry import load_env_cfg
    from mjlab_microduck.tasks import hd1910
    from mjlab_microduck.tasks import mdp as microduck_mdp

    it = lambda n: n * hd1910.NUM_STEPS_PER_ENV
    for task in ("Mjlab-Velocity-Flat-MicroDuck-HD1910", "Mjlab-Velocity-Flat-Backlash-MicroDuck-HD1910"):
        cfg = load_env_cfg(task)
        # stage 0 = v2b
        assert cfg.commands["twist"].rel_forward_envs == pytest.approx(0.2)
        assert cfg.rewards["stand_torque"].weight == -2.0
        assert tuple(cfg.events["randomize_joint_friction"].params["scale_range"]) == (0.9, 1.1)
        assert tuple(cfg.events["randomize_armature"].params["ranges"]) == (0.9, 1.1)
        gains = cfg.events["randomize_motor_gains"]
        assert gains.func is microduck_mdp.randomize_bam_gain_scale and gains.mode == "reset"
        assert tuple(gains.params["kp_range"]) == (1.0, 1.0) and tuple(gains.params["kd_range"]) == (1.0, 1.0)
        # the stages
        cur = cfg.curriculum
        fr = cur["randomize_joint_friction_stages"].params["stages"]
        assert [s["step"] for s in fr] == [0, it(600), it(1200)]
        assert fr[-1]["params"] == {"scale_range": hd1910.FRICTION_SCALE_RANGE} == {"scale_range": (0.5, 1.5)}
        assert cur["randomize_armature_stages"].params["stages"][-1]["params"] == {"ranges": (0.5, 1.5)}
        kp = cur["randomize_motor_gains_stages"].params["stages"]
        assert kp[0]["params"] == {"kp_range": (1.0, 1.0)} and kp[-1]["params"] == {"kp_range": (0.7, 1.4)}
        fw = cur["forward_envs_stages"].params
        assert fw["attr"] == "rel_forward_envs" and fw["stages"] == [{"step": 0, "value": 0.2}, {"step": it(1200), "value": 0.0}]
        st = cur["stand_torque_weight"].params
        assert st["reward_name"] == "stand_torque"
        assert st["weight_stages"] == [{"step": 0, "weight": -2.0}, {"step": it(2500), "weight": hd1910.STAND_TORQUE_WEIGHT_FINAL}]
        assert hd1910.STAND_TORQUE_WEIGHT_FINAL == -20.0  # v2d (v2e's -8 did not pass)
        for name in ("randomize_joint_friction_stages", "randomize_armature_stages", "randomize_motor_gains_stages"):
            assert cur[name].func is microduck_mdp.event_params_curriculum
            assert cur[name].params["stages"][0]["params"] == {  # stage 0 equals the event's own params
                k: tuple(v) for k, v in cfg.events[cur[name].params["event_name"]].params.items() if k in cur[name].params["stages"][0]["params"]}
    xl = load_env_cfg("Mjlab-Velocity-Flat-MicroDuck")
    assert xl.commands["twist"].rel_forward_envs == pytest.approx(0.2)
    assert tuple(xl.events["randomize_joint_friction"].params["scale_range"]) == (0.9, 1.1)
    assert "randomize_motor_gains" not in xl.events
    assert not {"forward_envs_stages", "stand_torque_weight", "randomize_joint_friction_stages"} & set(xl.curriculum)


def test_staged_curricula_mutate_the_live_terms():
    import types
    import torch
    from mjlab_microduck.tasks import mdp as microduck_mdp

    ev_cfg = types.SimpleNamespace(params={"scale_range": (0.9, 1.1), "asset_cfg": "robot"})
    cmd_cfg = types.SimpleNamespace(rel_forward_envs=0.2)
    env = types.SimpleNamespace(
        common_step_counter=0,
        event_manager=types.SimpleNamespace(get_term_cfg=lambda name: ev_cfg),
        command_manager=types.SimpleNamespace(get_term=lambda name: types.SimpleNamespace(cfg=cmd_cfg)))
    stages = [{"step": 0, "params": {"scale_range": (0.9, 1.1)}}, {"step": 100, "params": {"scale_range": (0.7, 1.3)}},
              {"step": 200, "params": {"scale_range": (0.5, 1.5)}}]
    fw = [{"step": 0, "value": 0.2}, {"step": 200, "value": 0.0}]
    for step, want_range, want_fw, want_idx in ((0, (0.9, 1.1), 0.2, 0), (150, (0.7, 1.3), 0.2, 1), (201, (0.5, 1.5), 0.0, 2)):
        env.common_step_counter = step
        idx = microduck_mdp.event_params_curriculum(env, None, "randomize_joint_friction", stages)
        val = microduck_mdp.command_attr_curriculum(env, None, "twist", "rel_forward_envs", fw)
        assert ev_cfg.params["scale_range"] == want_range and ev_cfg.params["asset_cfg"] == "robot"
        assert cmd_cfg.rel_forward_envs == want_fw and float(val) == pytest.approx(want_fw) and int(idx) == want_idx


def test_randomize_bam_gain_scale_spans_its_range():
    """One draw per env (not a mean of 14 per-joint draws, which collapses (0.7, 1.4) to ~1.05 +- 0.1),
    restored before each draw so it never accumulates."""
    import types
    import torch
    from bam.mjlab import BamActuator
    from mjlab_microduck.tasks import mdp as microduck_mdp

    class FakeBam(BamActuator):
        def __init__(self, n):  # skip BamActuator's init: only the gain bookkeeping is exercised
            self.kp_scale = torch.ones(n, 1)
            self.kd_scale = torch.ones(n, 1)
            self.default_kp_scale = self.kp_scale.clone()
            self.default_kd_scale = self.kd_scale.clone()

    n = 4096
    act = FakeBam(n)
    asset = types.SimpleNamespace(actuators=[act])
    env = types.SimpleNamespace(num_envs=n, device="cpu", scene={"robot": asset})
    torch.manual_seed(0)
    microduck_mdp.randomize_bam_gain_scale(env, None, kp_range=(0.7, 1.4))
    kp = act.kp_scale.squeeze(1)
    assert kp.min() >= 0.7 and kp.max() <= 1.4
    assert kp.min() < 0.72 and kp.max() > 1.38  # the whole range, not a collapsed mean
    assert kp.std() == pytest.approx(0.7 / math.sqrt(12), rel=0.05)
    assert torch.all(act.kd_scale == 1.0)
    ids = torch.arange(0, 10)
    before = act.kp_scale[10:].clone()
    microduck_mdp.randomize_bam_gain_scale(env, ids, kp_range=(1.0, 1.0))
    assert torch.all(act.kp_scale[:10] == 1.0) and torch.equal(act.kp_scale[10:], before)


def test_hd1910_v2d_adds_the_smoothed_fine_yaw_term():
    """v2d: both HD-1910 tasks carry track_yaw_rate_fine (std 0.35, weight 2, 0.5 s EMA) next to the stock
    track_angular_velocity, which is untouched; the XL330 task does not get it."""
    from mjlab.tasks.registry import load_env_cfg
    from mjlab_microduck.tasks import hd1910
    from mjlab_microduck.tasks import mdp as microduck_mdp

    for task in ("Mjlab-Velocity-Flat-MicroDuck-HD1910", "Mjlab-Velocity-Flat-Backlash-MicroDuck-HD1910"):
        cfg = load_env_cfg(task)
        t = cfg.rewards["track_yaw_rate_fine"]
        assert t.func is microduck_mdp.track_yaw_rate_fine and t.weight == hd1910.YAW_FINE_WEIGHT == 2.0
        assert t.params == {"std": 0.35, "command_name": "twist", "tau_s": 0.5}
        assert cfg.rewards["track_angular_velocity"].weight == 2.0
        assert cfg.rewards["track_angular_velocity"].params["std"] == pytest.approx(math.sqrt(0.5))
    assert "track_yaw_rate_fine" not in load_env_cfg("Mjlab-Velocity-Flat-MicroDuck").rewards


def test_track_yaw_rate_fine_ema_prices_the_dc_error_only():
    """With tau_s the term sees the yaw rate's running mean: a stepping turn oscillating around the command
    scores near 1, standing still at a 0.5 rad/s command scores exp(-0.25/0.35²) = 0.13; without tau_s
    (velstand) the instantaneous rate is used, as before."""
    import types
    import torch
    from mjlab_microduck.tasks import mdp as microduck_mdp

    n = 2
    data = types.SimpleNamespace(root_link_ang_vel_b=torch.zeros(n, 3))
    env = types.SimpleNamespace(step_dt=0.02, episode_length_buf=torch.full((n,), 100),
                                command_manager=types.SimpleNamespace(get_command=lambda name: torch.tensor([[0, 0, 0.5]] * n)),
                                scene={"robot": types.SimpleNamespace(data=data)})
    out = None
    for k in range(400):  # env 0 oscillates 0.5 +- 0.4 at 2.5 Hz (a stepping turn); env 1 stands still
        data.root_link_ang_vel_b[0, 2] = 0.5 + 0.4 * math.sin(2 * math.pi * 2.5 * k * 0.02)
        data.root_link_ang_vel_b[1, 2] = 0.0
        out = microduck_mdp.track_yaw_rate_fine(env, std=0.35, command_name="twist", tau_s=0.5)
    assert out[0] > 0.97 and out[1] == pytest.approx(math.exp(-0.25 / 0.35**2), rel=1e-3)
    data.root_link_ang_vel_b[0, 2] = 0.9  # instantaneous (velstand default) sees the peak
    inst = microduck_mdp.track_yaw_rate_fine(env, std=0.35, command_name="twist")
    assert inst[0] == pytest.approx(math.exp(-0.16 / 0.35**2), rel=1e-3)
