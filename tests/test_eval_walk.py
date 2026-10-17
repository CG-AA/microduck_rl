"""Regression tests for scripts/eval_walk.py (harness v2).

Locks the parts that were wrong or missing in v1: the body-frame rates (free-joint angular qvel is
already body-frame; rotating it again distorted yaw rate whenever the trunk tilted), the per-substep
target delay that reproduces training's actuator delay, the step-response metrics, and the JSON
keys the old evaluations carry (so v1 results stay comparable).
"""
import importlib.util
import json
import os
from pathlib import Path

import mujoco
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def ev():
    spec = importlib.util.spec_from_file_location("eval_walk", REPO / "scripts" / "eval_walk.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _find_onnx():
    env = os.environ.get("MICRODUCK_TEST_ONNX")
    if env:
        return env
    cands = sorted((REPO / "runs").glob("hd1910_*.onnx"), key=lambda p: p.stat().st_mtime)
    return str(cands[-1]) if cands else None


FREE_BODY = """
<mujoco><worldbody>
  <body name="trunk_base" pos="0 0 1"><freejoint name="trunk_base_freejoint"/><geom type="box" size="0.05 0.05 0.05" mass="1"/></body>
</worldbody></mujoco>
"""


def test_body_rates_use_body_frame_angular_velocity(ev):
    model = mujoco.MjModel.from_xml_string(FREE_BODY)
    data = mujoco.MjData(model)
    trunk = model.body("trunk_base").id
    dof = model.jnt_dofadr[model.joint("trunk_base_freejoint").id]
    # Yaw the trunk 90 deg and tilt it 30 deg in pitch: world x-velocity must read as body -y,
    # and the angular slots must pass through untouched (they are already body-frame).
    quat = np.zeros(4)
    mujoco.mju_euler2Quat(quat, np.radians([0.0, 30.0, 90.0]), "xyz")
    data.qpos[3:7] = quat
    data.qvel[dof:dof + 3] = [1.0, 0.0, 0.0]
    data.qvel[dof + 3:dof + 6] = [0.1, 0.2, 1.0]
    mujoco.mj_forward(model, data)
    v_body, w_body, R = ev.body_rates(data, trunk, dof)
    np.testing.assert_allclose(w_body, [0.1, 0.2, 1.0], atol=1e-12)
    np.testing.assert_allclose(v_body, R.T @ np.array([1.0, 0.0, 0.0]), atol=1e-12)
    assert abs(v_body[1] + np.cos(np.radians(0.0))) < 0.6  # mostly -y after the 90 deg yaw
    assert abs(np.linalg.norm(v_body) - 1.0) < 1e-9  # a rotation, not a scaling


def test_delay_line_lags_within_range_and_is_reproducible(ev):
    init = np.zeros(3)
    a = ev.DelayLine(init, (3, 6), np.random.default_rng(7))
    b = ev.DelayLine(init, (3, 6), np.random.default_rng(7))
    lags_a, lags_b, hist = [], [], [init.copy()]
    for k in range(1, 60):
        target = np.full(3, float(k))
        out_a, lag_a = a.push_and_get(target)
        out_b, lag_b = b.push_and_get(target)
        hist.append(target)
        lags_a.append(lag_a)
        lags_b.append(lag_b)
        assert 3 <= lag_a <= 6
        expected = hist[-1 - lag_a] if lag_a < len(hist) else init  # target pushed `lag` substeps ago
        np.testing.assert_array_equal(out_a, expected)
        np.testing.assert_array_equal(out_a, out_b)
    assert lags_a == lags_b
    assert len(set(lags_a)) == 4  # every lag in {3,4,5,6} gets sampled
    passthrough = ev.DelayLine(init, (0, 0))
    np.testing.assert_array_equal(passthrough.push_and_get(np.ones(3))[0], np.ones(3))
    assert passthrough.push_and_get(np.ones(3))[1] == 0


def _rows(dt, segments):
    """Synthetic (segment, cmd, _vel) rows: first-order response of vx with tau=0.2 s to each command."""
    rows, v = [], 0.0
    for name, seconds, cmd in segments:
        n = int(round(seconds / dt))
        vel = np.zeros((n, 3))
        for k in range(n):
            v += (cmd - v) * dt / 0.2
            vel[k, 0] = v
        rows.append(dict(segment=name, cmd=[cmd, 0.0, 0.0], _vel=vel))
    return rows


def test_step_metrics_on_first_order_response(ev):
    dt = 0.02
    rows = _rows(dt, [("stand", 2.0, 0.0), ("fwd 0.20", 4.0, 0.2), ("stand", 3.0, 0.0)])
    up, down = ev.step_metrics(rows, dt)
    assert up["step"] == "stand -> fwd 0.20" and up["axis"] == "vx"
    # tau = 0.2 s: t10 ~ 0.02 s, t90 ~ 0.46 s, plus the 0.3 s moving-average lag (about half of it)
    assert 0.0 < up["t10"] < 0.3 and 0.4 < up["t90"] < 0.8 and up["rise"] == pytest.approx(up["t90"] - up["t10"])
    assert up["overshoot"] == pytest.approx(0.0, abs=1e-6)
    assert up["settle_t"] is not None and up["settle_t"] < 1.5
    assert abs(up["ss_err"]) < 1e-3 and up["ss_std"] < 1e-3
    assert up["t_stop"] is None
    assert down["from_meas"] == pytest.approx(0.2, abs=1e-3)
    assert down["t_stop"] is not None and 0.3 < down["t_stop"] < 1.0


def test_step_metrics_ignore_a_step_the_robot_never_took(ev):
    dt = 0.02
    rows = _rows(dt, [("stand", 2.0, 0.0), ("fwd 0.10", 4.0, 0.0), ("stand", 3.0, 0.0)])
    rows[1]["cmd"] = [0.10, 0.0, 0.0]  # commanded 0.1, achieved 0 (the v1 behaviour)
    up, down = ev.step_metrics(rows, dt)
    assert up["t90"] is None and up["ss_err"] == pytest.approx(0.10, abs=1e-9)
    assert down["t10"] is None and down["t90"] is None  # 0.1 -> 0 from a standstill is not a response


def test_aggregate_means_rows_and_keeps_falls(ev):
    r = lambda seg, vx, fell=None: dict(segment=seg, cmd=[vx, 0, 0], fell_at=fell, mean=[vx, 0.0, 0.0],
                                         torque_rms={"a": vx, "b": 2 * vx})
    agg = ev.aggregate([[r("s", 0.1), r("f", 0.2)], [r("s", 0.3), r("f", 0.4, fell=1.5)], [r("s", 0.5)]])
    assert [a["segment"] for a in agg] == ["s", "f"]
    assert agg[0]["mean"][0] == pytest.approx(0.3) and agg[0]["n"] == 3 and agg[0]["fell_at"] is None
    assert agg[1]["torque_rms"]["b"] == pytest.approx(0.6) and agg[1]["n"] == 2
    assert agg[1]["fell_at"] == 1.5 and agg[1]["fell_count"] == 1


def test_gate_table_reads_v1_json_layout(ev):
    v1 = REPO / "runs" / "hd1910_v1_final_eval.json"
    if not v1.exists():
        pytest.skip("no v1 eval JSON in runs/")
    doc = json.loads(v1.read_text())
    rows = ev.gate_rows(doc, doc)
    by = {name: (x, y, v) for name, x, y, v in rows}
    assert by["fwd 0.10 vx [m/s]"][0] == pytest.approx(doc["schedule"][1]["mean"][0])
    assert by["step 0->0.10 t90 [s]"][2] == "n/a"  # v1 had no step rollout
    assert by["stand torque max joint [N.m]"][0] == pytest.approx(max(doc["schedule"][0]["torque_rms"].values()))


def test_rollout_rows_keep_v1_keys(ev):
    onnx = _find_onnx()
    if onnx is None:
        pytest.skip("no runs/hd1910_*.onnx")
    import argparse
    args = argparse.Namespace(walking=onnx, hd1910=True, vin=7.4, vin_drop_gain=0.1, kp_fw=None, continuous=0.29)
    bam_model, model, data, bam_ctrl, policy = ev.build(args)
    assert policy.default_pose_source == "onnx"
    assert policy.joint_vel_lag == ev.JOINT_VEL_LAG
    ev.reset(model, data, bam_ctrl, policy)
    delay = ev.DelayLine(policy.default_pose, ev.ACTION_DELAY, np.random.default_rng(0))
    rows = ev.strip(ev.run([("stand", 0.6, 0.0, 0.0, 0.0)], bam_model, model, data, bam_ctrl, policy, delay=delay))
    row = rows[0]
    for key in ("segment", "cmd", "fell_at", "mean", "torque_rms", "torque_peak", "saturated"):  # v1
        assert key in row
    for key in ("torque_rms_settled", "torque_mean_abs_settled", "torque_peak_substep", "over_continuous_frac",
                "speed_util_max", "speed_util_frac", "torque_over_stall_max", "lag_mean"):
        assert key in row
    assert set(row["torque_rms"]) == set(policy.actuator_names)
    assert 3.0 <= row["lag_mean"] <= 6.0
    assert row["fell_at"] is None
    assert policy.walking_session.get_providers() == ["CPUExecutionProvider"]


# ---- iteration 2: servo-model overrides, backlash encoder view, sensitivity table ---------------------------

def test_bam_override_parsing(ev):
    specs = ev.parse_bam_overrides(["kt*=1.2", "R=3.0", "friction*=0.5", " kp *= 2 "])
    assert specs == [("kt", "*", 1.2), ("R", "=", 3.0), ("friction", "*", 0.5), ("kp", "*", 2.0)]
    assert ev.parse_bam_overrides(None) == [] and ev.parse_bam_overrides([]) == []
    with pytest.raises(ValueError, match="use --vin"):
        ev.parse_bam_overrides(["vin=6.5"])
    with pytest.raises(ValueError, match="KEY=VALUE"):
        ev.parse_bam_overrides(["kt"])


def test_bam_overrides_mutate_the_model(ev):
    from bam.model import load_model
    m = load_model(str(REPO / ev.ip.HD1910_PARAMS_JSON))
    m.actuator.kp = 156.0
    base = {n: p.value for n, p in m.get_parameters().items()}
    changed = ev.apply_bam_overrides(m, ev.parse_bam_overrides(["friction*=0.5", "kt=0.7", "kp*=2"]))
    for name in ev.FRICTION_COEFFS:
        assert m.get_parameters()[name].value == pytest.approx(0.5 * base[name])
    for name in ("dtheta_stribeck", "alpha", "friction_viscous", "R", "armature"):  # friction_scale leaves these
        assert m.get_parameters()[name].value == base[name]
    assert m.kt.value == 0.7 and m.actuator.kp == 312.0
    assert changed["kt"] == (pytest.approx(base["kt"]), 0.7)
    with pytest.raises(ValueError, match="unknown key"):
        ev.apply_bam_overrides(m, [("not_a_param", "*", 2.0)])


def _ns(onnx, **kw):
    import argparse
    return argparse.Namespace(walking=onnx, hd1910=True, vin=7.4, vin_drop_gain=0.1, kp_fw=None, continuous=0.29, **kw)


def test_bam_overrides_reach_the_controller(ev):
    onnx = _find_onnx()
    if onnx is None:
        pytest.skip("no runs/hd1910_*.onnx")
    ref = ev.build(_ns(onnx))
    bam_model, model, data, bam_ctrl, policy = ev.build(
        _ns(onnx, bam=["kt*=1.2", "armature*=2", "kp*=2", "friction*=0.5"]))
    kt0, kt = ref[0].kt.value, bam_model.kt.value
    assert kt == pytest.approx(1.2 * kt0) and bam_ctrl.model is bam_model
    assert model.dof_armature[bam_ctrl.dof_indexes] == pytest.approx(2 * ref[1].dof_armature[ref[3].dof_indexes])
    assert model.actuator_forcerange[0, 1] == pytest.approx(7.4 * kt / bam_model.R.value)  # read after the override
    assert policy.limits["duty_per_rad"] == pytest.approx(2 * ref[4].limits["duty_per_rad"])
    assert policy.bam_resolved["friction_base"] == pytest.approx(0.5 * ref[4].bam_resolved["friction_base"])
    assert set(policy.bam_overrides) >= {"kt", "armature", "kp", *ev.FRICTION_COEFFS}


def test_backlash_encoder_view(ev):
    onnx = _find_onnx()
    if onnx is None:
        pytest.skip("no runs/hd1910_*.onnx")
    bam_model, model, data, bam_ctrl, policy = ev.build(_ns(onnx, backlash=True))
    assert type(bam_ctrl).__name__ == "BacklashEncoderController"
    ev.reset(model, data, bam_ctrl, policy)
    j = 3  # left knee
    b_adr = policy.backlash_qpos_indices[j]
    assert model.joint(int(model.actuator_trnid[j, 0])).name == "left_knee"
    assert b_adr == bam_ctrl.backlash_qpos_indexes[j]
    data.qpos[b_adr] = 0.01
    data.qvel[policy.backlash_qvel_indices[j]] = 0.5
    mujoco.mj_forward(model, data)
    servo = data.qpos[policy.joint_qpos_indices]
    np.testing.assert_allclose(policy.get_joint_pos_relative()[j], servo[j] + 0.01 - policy.default_pose[j], atol=1e-6)
    policy.joint_vel_lag = 0
    assert policy.get_joint_vel()[j] == pytest.approx(data.qvel[policy.joint_qvel_indices[j]] + 0.5, abs=1e-6)
    # The firmware loop reads servo + backlash: with the target at the servo angle, the knee is driven
    # back by the 0.01 rad of play (duty = -0.01 * duty_per_rad; torque = kt * V / R with dq = 0).
    target = bam_ctrl.q_target
    target[:] = servo
    data.qvel[:] = 0.0
    bam_ctrl.vin_drop_gain = None
    bam_ctrl.update()
    act = bam_model.actuator
    expected = -0.01 * act.kp * act.error_gain * act.vin * bam_model.kt.value / bam_model.R.value
    assert data.ctrl[bam_ctrl.act_indexes[j]] == pytest.approx(expected, rel=1e-6)
    assert bam_ctrl.q_target is target  # the array control_step writes into survives update()
    np.testing.assert_allclose(target, servo)
    ev.reset(model, data, bam_ctrl, policy)
    delay = ev.DelayLine(policy.default_pose, ev.ACTION_DELAY, np.random.default_rng(0))
    rows = ev.run([("stand", 0.6, 0.0, 0.0, 0.0)], bam_model, model, data, bam_ctrl, policy, delay=delay)
    assert rows[0]["fell_at"] is None


def _sens():
    spec = importlib.util.spec_from_file_location("sensitivity", REPO / "scripts" / "sensitivity.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _synthetic_doc(fwd10, legs, fell=None):
    """A minimal harness-v2 JSON: the schedule rows gate_rows reads, plus one push rollout."""
    sched = [dict(segment="stand", cmd=[0, 0, 0], fell_at=None, mean=[0.0, 0.0, 0.0],
                  torque_mean_abs_settled={"left_knee": legs / 2, "right_knee": legs / 2, "head_pitch": 0.01}),
             dict(segment="fwd 0.10", cmd=[0.1, 0, 0], fell_at=None, mean=[fwd10, 0, 0]),
             dict(segment="back 0.10", cmd=[-0.1, 0, 0], fell_at=None, mean=[-0.1, 0, 0]),
             dict(segment="left 0.10", cmd=[0, 0.1, 0], fell_at=None, mean=[0, 0.1, 0]),
             dict(segment="fwd 0.20", cmd=[0.2, 0, 0], fell_at=None, mean=[0.2, 0, 0]),
             dict(segment="fwd 0.30", cmd=[0.3, 0, 0], fell_at=None, mean=[0.3, 0, 0]),
             dict(segment="turn +1.0", cmd=[0, 0, 1], fell_at=None, mean=[0, 0, 1.0]),
             dict(segment="turn -1.0", cmd=[0, 0, -1], fell_at=None, mean=[0, 0, -1.0])]
    push = [dict(segment="push walk", cmd=[0.1, 0, 0], fell_at=fell, mean=[0.1, 0, 0])]
    return dict(args={"continuous": 0.29}, summary=dict(schedule=sched, push=push, steps=[], push_stand=[]),
                replicas=dict(schedule=[sched], push=[push]))


def test_sensitivity_table_and_verdicts(tmp_path):
    sens = _sens()
    base = tmp_path / "base.json"
    base.write_text(json.dumps(_synthetic_doc(0.0, 0.6)))
    (tmp_path / "nominal.json").write_text(json.dumps(_synthetic_doc(0.095, 0.30)))
    (tmp_path / "kp_x0.7.json").write_text(json.dumps(_synthetic_doc(0.08, 0.30, fell=6.5)))
    (tmp_path / "friction_x1.5.json").write_text(json.dumps(_synthetic_doc(0.09, 0.40)))
    (tmp_path / "kp_x0.5.json").write_text(json.dumps(_synthetic_doc(0.01, 0.30)))
    conds = [("nominal", []), ("kp_x0.7", []), ("friction_x1.5", []), ("kp_x0.5", []), ("absent", [])]
    text, verdicts = sens.table(tmp_path, base, conds)
    assert verdicts == {"nominal": "robust", "kp_x0.7": "fails", "friction_x1.5": "degraded", "kp_x0.5": "fails"}
    assert (tmp_path / "table.md").read_text() == text
    line = {l.split(" | ")[0].lstrip("| "): l for l in text.splitlines() if l.startswith("| ")}
    assert "| push x1 |" in line["kp_x0.7"] and "0.080" in line["kp_x0.7"]
    assert "0.010 ✗" in line["kp_x0.5"] and "fwd 0.10 vx" in line["kp_x0.5"]
    assert "missing" in line["absent"]
    assert "**fails** (2): kp_x0.7, kp_x0.5" in text
