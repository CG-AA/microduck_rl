"""Pre-flight pose test for the HD-1910 stand pose (robot/hd1910_pose.py).

AGENTS.md: a target pose must be a stable equilibrium; hold its ctrl for 3 s from noisy inits and
check TILT, not just height. One twist on this robot: at the training gain the servo PD alone
cannot hold any pose (joint stiffness 0.55 N.m/rad against a gravity stiffness of ~1.07), so the
holds run at kp x8, which reads the gravity moment the policy has to supply on average; the last
test documents that fact so nobody "fixes" these tests by lowering the gain.
"""
import importlib.util
import os
from pathlib import Path

import mujoco
import numpy as np
import pytest

from mjlab_microduck.robot.hd1910_pose import (
    HD1910_DEFAULT_POSE,
    HD1910_STAND,
    HD1910_TRUNK_Z_CONTACT,
    JOINT_NAMES,
    NECK_PITCH,
)

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def st():
    spec = importlib.util.spec_from_file_location("stand_torque", REPO / "scripts" / "stand_torque.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _find_onnx():
    env = os.environ.get("MICRODUCK_TEST_ONNX")
    if env:
        return env
    cands = sorted((REPO / "runs").glob("hd1910_*.onnx"), key=lambda p: p.stat().st_mtime)
    return str(cands[-1]) if cands else None


def test_pose_is_flat_sole_mirror_symmetric_and_in_actuator_order():
    assert HD1910_STAND["ankle"] == pytest.approx(HD1910_STAND["knee"] - HD1910_STAND["hip_pitch"], abs=1e-4)
    p = HD1910_DEFAULT_POSE
    assert p.shape == (14,) and p.dtype == np.float32
    np.testing.assert_allclose(p[9:14], -p[0:5], atol=1e-7)
    assert p[5] == pytest.approx(NECK_PITCH) and p[6] == pytest.approx(NECK_PITCH) and p[7] == p[8] == 0.0
    assert JOINT_NAMES[3] == "left_knee" and p[3] == pytest.approx(-0.6)


def test_stand_pose_is_a_static_equilibrium(st):
    starts = st.perturbed_starts(n=8, joint_noise=0.05, tilt_deg=3.0, seed=0)
    ev = st.evaluate(HD1910_DEFAULT_POSE, starts, kp_scale=8.0, seconds=3.0)
    assert ev["passed"], f"perturbed starts {ev['failing']} did not settle upright"
    for r in [ev["nominal"]] + ev["starts"]:
        assert not r["fell"]
        assert r["tilt_end"] < 2.0 and r["tilt_max"] < 8.0
        assert r["settled"] and r["tau_drift"] < 0.005
        assert max(r["sole_tilt_left"], r["sole_tilt_right"]) < 1.0
        assert max(r["q_err"]) < 0.03
    n = ev["nominal"]
    assert n["sumsq_legs"] < 0.004, "legs' standing heat should be well below HOME's 0.0090"
    assert n["maxabs"] < 0.04
    assert abs(n["com_ankle_mm"]) < 3.0
    assert abs(n["z_end"] - HD1910_TRUNK_Z_CONTACT) < 0.004


def test_backlash_twin_holds_with_the_same_torque(st):
    # +-1 deg of play on five joints per leg lets gravity move the body through the play, so the
    # backlash model never meets a 2 deg tilt gate (HOME tilts 5.4 deg on it); the pose must still
    # stand and cost the same torque.
    r = st.hold(HD1910_DEFAULT_POSE, backlash=True, kp_scale=8.0, seconds=3.0)
    assert not r["fell"] and r["tilt_end"] < 3.0
    assert r["sumsq_legs"] < 0.004 and r["maxabs"] < 0.04


def test_gravity_exceeds_joint_stiffness_at_training_gain(st):
    bam, model, data, _, _ = st.build(kp_scale=1.0)
    st.place(model, data, HD1910_DEFAULT_POSE, 0.0)
    mujoco.mj_comPos(model, data)
    trunk = model.body("trunk_base").id
    gravity_stiffness = float(model.body_subtreemass[trunk]) * -model.opt.gravity[2] * float(data.subtree_com[trunk][2])
    act = bam.actuator
    joint_stiffness = act.kp * act.error_gain * act.vin * bam.kt.value / bam.R.value
    assert joint_stiffness == pytest.approx(0.55, rel=0.05)
    assert gravity_stiffness > 1.5 * joint_stiffness, "if this ever fails, the kp x8 holds above are no longer needed"


def test_policy_holds_stand_from_perturbed_starts(st):
    onnx = _find_onnx()
    if onnx is None:
        pytest.skip("no runs/hd1910_*.onnx")
    import onnxruntime as ort
    meta = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"]).get_modelmeta().custom_metadata_map
    trained = np.array([float(x) for x in meta.get("default_joint_pos", "").split(",")] or [np.nan])
    if trained.shape != (14,) or np.abs(trained - HD1910_DEFAULT_POSE).max() > 5e-4:
        pytest.skip("the newest ONNX was trained on another default pose")
    for s in [dict()] + st.perturbed_starts(n=8, joint_noise=0.05, tilt_deg=3.0, seed=0):
        r = st.hold(HD1910_DEFAULT_POSE, kp_scale=1.0, seconds=3.0, policy_onnx=onnx, **s)
        assert not r["fell"] and r["tilt_end"] < 5.0 and r["z_end"] > 0.10
