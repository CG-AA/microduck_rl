"""Two rollouts of the same policy and schedule must be bit-identical.

Nondeterminism hides bugs behind noise. The CPU rehearsal has no intended randomness beyond the
seeded target delay, but onnxruntime threads and BAM's per-step rewrites of dof_frictionloss /
dof_damping (which mj_resetData does not undo) can both break reproducibility, so each rollout
here builds a fresh model, data, controller and session.
"""
import argparse
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SEGMENTS = [("stand", 1.0, 0.0, 0.0, 0.0), ("fwd", 1.5, 0.2, 0.0, 0.0), ("turn", 0.5, 0.0, 0.0, 1.0)]


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


def rollout(ev, onnx):
    args = argparse.Namespace(walking=onnx, hd1910=True, vin=7.4, vin_drop_gain=0.1, kp_fw=None, continuous=0.29)
    bam_model, model, data, bam_ctrl, policy = ev.build(args)
    ev.reset(model, data, bam_ctrl, policy)
    delay = ev.DelayLine(policy.default_pose, ev.ACTION_DELAY, np.random.default_rng(0))
    dt = ev.DECIMATION * model.opt.timestep
    obs, actions, qpos, ctrl = [], [], [], []
    for _, seconds, vx, vy, wz in SEGMENTS:
        policy.set_vel_cmd(vx, vy, wz)
        for _ in range(int(round(seconds / dt))):
            smp = ev.control_step(policy, bam_ctrl, model, data, delay)
            obs.append(policy.last_obs.copy())
            actions.append(smp["action"].copy())
            qpos.append(data.qpos.copy())
            ctrl.append(data.ctrl.copy())
    return policy, np.array(obs), np.array(actions), np.array(qpos), np.array(ctrl)


def test_two_rollouts_are_bit_identical(ev):
    onnx = _find_onnx()
    if onnx is None:
        pytest.skip("no runs/hd1910_*.onnx")
    policy, *a = rollout(ev, onnx)
    _, *b = rollout(ev, onnx)
    for name, x, y in zip(("obs", "actions", "qpos", "ctrl"), a, b):
        assert np.array_equal(x, y), f"{name} differs between two identical rollouts"
    assert a[0].shape == (150, 61)
    assert np.abs(a[1]).max() > 0 and np.abs(a[3]).max() > 0  # a real trajectory, not a frozen one
    assert policy.walking_session.get_providers() == ["CPUExecutionProvider"]
    assert policy.walking_session.get_session_options().intra_op_num_threads == 1
