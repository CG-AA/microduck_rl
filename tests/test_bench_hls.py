"""Bench kit for the HD-1910 (src/mjlab_microduck/bench, scripts/bench_*.py), tested on the simulated servo.

Locks the HLS-specific traps (sign-magnitude registers with the load sign at BIT10, gains in RAM 50-52,
address 44 = goal current, EEPROM writes only while unlocked), the measurement logic (gain map, two-sided
zero, recording in BAM's format) and the conversion of a fit into the simulator's actuator file (same
duty per radian). The full planted-servo recovery takes minutes and is a script check:
`bench_record.py --fake` + `bench_fit.py --truth` (docs/bench.md).
"""
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from mjlab_microduck.bench import bam_hd1910 as bh
from mjlab_microduck.bench import hls
from mjlab_microduck.bench.cli import find_zero
from mjlab_microduck.bench.fake import FakeClock, FakeHlsController

REPO = Path(__file__).resolve().parents[1]


def _script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _bus(**kw):
    clock = FakeClock()
    ctrl = FakeHlsController(clock, **kw)
    return hls.HlsBus(ctrl, [1], clock=clock), clock, ctrl


def test_sign_magnitude_registers():
    for bit in (10, 15):
        for v in (0, 1, -1, 37, -37, (1 << bit) - 1, -((1 << bit) - 1)):
            assert hls.to_signed(hls.from_signed(v, bit), bit) == v
    assert hls.from_signed(-5, 15) == 0x8005 and hls.from_signed(-5, 10) == 0x0405
    with pytest.raises(ValueError):
        hls.from_signed(1 << 10, 10)
    with pytest.raises(ValueError):
        hls.from_signed(-1, None)
    assert hls.rad_to_count(0.0) == 2048 and hls.count_to_rad(3072) == pytest.approx(math.pi / 2)


def test_decode_feedback_literal_block():
    # pos 3072 (+pi/2), speed -10 counts (BIT15), load -250 (BIT10 = 0x400), 7.4 V, 31 C, async 0,
    # status 0x08 (current), moving 1, two unused bytes, current -40 counts (BIT15).
    block = [0x00, 0x0C, 0x0A, 0x80, 0xFA, 0x04, 74, 31, 0, 0x08, 1, 0, 0, 0x28, 0x80]
    fb = hls.decode_feedback(block, t=1.5)
    assert fb.pos_rad == pytest.approx(math.pi / 2)
    assert fb.vel_rad_s == pytest.approx(-10 * 0.732 * 2 * math.pi / 60)
    assert fb.load_duty == pytest.approx(-0.250)
    assert fb.volts == pytest.approx(7.4) and fb.temp_c == 31 and fb.moving == 1 and fb.t == 1.5
    assert fb.current_a == pytest.approx(-40 * 0.0065)
    assert fb.faults() == ["current"]
    with pytest.raises(ValueError):
        hls.decode_feedback(block[:14])


def test_gains_go_to_ram_and_address_44_is_never_written():
    bus, clock, ctrl = _bus()
    bus.set_gains(1, 3, 0, 0)
    bus.goal(1, 0.5)
    bus.torque(1, hls.TORQUE_ON)
    bus.sync_goal({1: 0.4})
    addrs = [a for _, a, _ in ctrl.writes]
    assert {50, 51, 52} <= set(addrs) and 44 not in addrs and 21 not in addrs
    assert bus.read_reg(1, "p_gain") == 3 and bus.read_reg(1, "p_gain_eeprom") == 32  # RAM only
    with pytest.raises(ValueError, match="EEPROM"):
        bus.write_reg(1, "mode", 0)
    with pytest.raises(ValueError, match="read-only"):
        bus.write_reg(1, "present_position", 0)
    with pytest.raises(ValueError):
        bus.set_gains(1, 300)
    bus.write_eeprom(1, {"cw_dead": 0, "ccw_dead": 0})
    assert ctrl.eeprom_writes_while_locked() == [] and bus.read_reg(1, "lock") == 1
    assert bus.read_reg(1, "cw_dead") == 0


def test_configure_dry_run_writes_nothing_and_apply_converges():
    cfg = _script("bench_configure")
    bus, clock, ctrl = _bus()
    want = cfg.targets(json.loads(cfg.PROFILE.read_text()), kp=3, persist_gains=True)
    assert want["p_gain"] == 3 and want["p_gain_eeprom"] == 3 and want["d_gain"] == 0 and "goal_current" not in want
    todo = cfg.configure(bus, 1, want, apply=False, log=lambda *a: None)
    assert {n for n, _, _ in todo} >= {"p_gain", "d_gain", "cw_dead", "ccw_dead"} and ctrl.writes == []
    left = cfg.configure(bus, 1, want, apply=True, log=lambda *a: None)
    assert left == [] and ctrl.eeprom_writes_while_locked() == []
    assert bus.read_reg(1, "torque_enable") == 0  # torque cut before writing
    assert cfg.configure(bus, 1, want, apply=False, log=lambda *a: None) == []


def test_gainmap_recovers_the_duty_per_radian():
    st = _script("bench_static")
    bus, clock, ctrl = _bus(blocked=True)
    res = st.gainmap(bus, 1, clock, [4, 16], [5, 20], log=lambda *a: None)
    assert res["error_gain"] == pytest.approx(ctrl.model.actuator.error_gain, rel=0.01)
    assert res["max_duty"] <= 0.98 + 1e-9
    assert res["p_for_sim_exact"] == pytest.approx(bh.SIM_KP_FW * bh.XL330_ERROR_GAIN / res["error_gain"])
    assert bus.read_reg(1, "torque_enable") == 0
    synthetic = [dict(p=8, err=e, duty=float(np.clip(0.2 * 8 * e, -0.98, 0.98))) for e in (0.05, -0.05, 0.1, 1.0)]
    out = st.analyse_gainmap(synthetic)
    assert out["error_gain"] == pytest.approx(0.2) and out["n_linear_points"] == 3  # the saturated point dropped


def test_two_sided_zero_cancels_friction_and_flags_a_light_arm():
    bus, clock, ctrl = _bus(mass=0.3, length=0.08, theta0=0.3)
    zero, info = find_zero(bus, 1, clock)
    assert info["ok"] and zero == pytest.approx(0.3, abs=0.004) and info["band"] < 0.05
    bus, clock, ctrl = _bus(mass=0.05, length=0.08, theta0=0.3)
    zero, info = find_zero(bus, 1, clock)
    assert not info["ok"] and info["band"] > 0.15
    assert zero == pytest.approx(0.3, abs=0.02)  # the midpoint is still right; the logs would not be


def test_recording_is_a_bam_log_the_planted_model_reproduces(tmp_path):
    rec = _script("bench_record")
    bus, clock, ctrl = _bus(mass=0.15, length=0.08, arm_mass=0.01)
    written = rec.session(bus, 1, clock, tmp_path / "raw", 0.15, 0.08, 0.01, [8], ["sin_sin"], log=lambda *a: None)
    doc = json.loads(Path(written[0]).read_text())
    assert doc["kp"] == 8 and doc["arm_mass"] == 0.01 and doc["motor"] == "hd1910" and doc["vin"] == pytest.approx(7.4)
    e = doc["entries"]
    assert 1450 <= len(e) <= 1510 and {"position", "speed", "goal_position", "torque_enable", "current", "load"} <= set(e[0])
    assert np.mean(np.diff([x["timestamp"] for x in e])) == pytest.approx(1 / 250, rel=0.02)
    bh.process(tmp_path / "raw", tmp_path / "proc")
    truth = bh.model_from_json(dict(ctrl.truth, error_gain=ctrl.model.actuator.error_gain))
    batch = bh.load_logs(tmp_path / "proc").make_batch()
    good = bh.score(truth, batch)
    assert good < 0.01  # rad: the planted servo explains its own recording
    wrong = bh.model_from_json(dict(ctrl.truth, error_gain=ctrl.model.actuator.error_gain, kt=1.2 * ctrl.truth["kt"]))
    assert bh.score(wrong, batch) > good


def test_conversion_reproduces_the_duty_in_the_sim_actuator():
    from bam.model import load_model_from_dict
    fit = bh.params_dict(bh.make_model(values=dict(kt=0.7, R=4.0, error_gain_ratio=1.1)))
    sim, kp_fw = bh.to_xl330_law(fit, p_register=3)
    assert sim["actuator"] == "xl330" and sim["model"] == "m6" and sim["q_offset"] == 0.0
    m = load_model_from_dict(sim)  # what scripts/infer_policy.load_bam_model and mjlab's BAM actuator read
    m.actuator.kp, m.actuator.vin, m.actuator.max_pwm, m.actuator.max_current = kp_fw, 7.4, bh.MAX_PWM, None
    h = bh.model_from_json(fit)
    h.actuator.kp, h.actuator.vin = 3, 7.4
    for err in (0.01, -0.05, 0.2, 2.0):  # 2.0 rad saturates both at max_pwm
        assert m.actuator.compute_control(err, 0.0, 0.0, 0.005) == pytest.approx(
            h.actuator.compute_control(err, 0.0, 0.0, 0.005), rel=1e-9)
    assert m.kt.value == 0.7 and m.R.value == 4.0
    exact, nearest = bh.p_register_for_sim(fit)
    assert exact == pytest.approx(bh.SIM_KP_FW * bh.XL330_ERROR_GAIN / (bh.ERROR_GAIN_PLACEHOLDER * 1.1))
    assert nearest == max(1, round(exact))


def test_register_makes_hd1910_loadable_by_bam():
    from bam.model import load_model_from_dict
    bh.register()
    m = load_model_from_dict(bh.params_dict(bh.make_model(values=dict(kt=0.8))))
    assert type(m.actuator).__name__ == "HD1910Actuator" and m.kt.value == 0.8


def test_thermal_first_order_fit():
    th = _script("bench_thermal")
    t = np.arange(0, 900, 1.0)
    temp = 63.0 - (63.0 - 25.0) * np.exp(-t / 300.0)
    tinf, tau, t0 = th.fit_first_order(t, np.round(temp))
    assert tinf == pytest.approx(63.0, abs=0.5) and tau == pytest.approx(300, rel=0.05)
    assert th.fit_first_order(t[:5], temp[:5]) is None
