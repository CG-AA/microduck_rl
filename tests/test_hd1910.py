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
