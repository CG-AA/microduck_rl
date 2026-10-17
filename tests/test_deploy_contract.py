"""The deployment contract of an HD-1910 policy file (docs/runtime_spec_hd1910.md), checked on the ONNX.

A runtime builds the 61-D observation and applies the 14 actions from this file and its metadata
alone, so everything it relies on is locked here: shapes, the joint order, the stand pose (which is NOT
the XL330 HOME pose), the action scale, the observation block order, the baked-in normalizer, and sane
outputs at the stand. Runs on the newest `runs/hd1910_*.onnx` or `MICRODUCK_TEST_ONNX`.
"""
import os
from pathlib import Path

import numpy as np
import pytest

from mjlab_microduck.robot.hd1910_pose import HD1910_DEFAULT_POSE, JOINT_NAMES

REPO = Path(__file__).resolve().parents[1]
OBS_BLOCKS = ["base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions", "command", "head_command",
              "body_command"]
OBS_WIDTHS = [3, 3, 14, 14, 14, 3, 4, 6]


def _find_onnx():
    env = os.environ.get("MICRODUCK_TEST_ONNX")
    if env:
        return env
    cands = sorted((REPO / "runs").glob("hd1910_*.onnx"), key=lambda p: p.stat().st_mtime)
    return str(cands[-1]) if cands else None


@pytest.fixture(scope="module")
def policy():
    path = _find_onnx()
    if path is None:
        pytest.skip("no runs/hd1910_*.onnx (export one, or set MICRODUCK_TEST_ONNX)")
    import onnx
    import onnxruntime as ort
    model = onnx.load(path)
    meta = {p.key: p.value for p in model.metadata_props}
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
    return path, model, meta, sess


def test_shapes(policy):
    _, model, _, sess = policy
    (inp,), (out,) = sess.get_inputs(), sess.get_outputs()
    assert inp.shape[-1] == 61 and out.shape[-1] == 14
    assert sum(OBS_WIDTHS) == 61


def test_metadata_is_the_hd1910_contract(policy):
    path, _, meta, _ = policy
    names = meta["joint_names"].split(",")
    assert [n for n in names if not n.startswith("passive_")] == list(JOINT_NAMES)
    assert len(names) == 14, "a policy trained on the backlash body lists passive joints: export from the walk task"
    pose = np.array([float(x) for x in meta["default_joint_pos"].split(",")])
    np.testing.assert_allclose(pose, HD1910_DEFAULT_POSE, atol=5e-4)  # exported at 3 decimals
    assert float(meta["action_scale"]) == 1.0
    assert meta["observation_names"].split(",") == OBS_BLOCKS
    assert meta["command_names"].split(",") == ["twist", "head_pose", "body_pose"]
    assert "hd1910" in meta.get("run_path", "hd1910").lower()


def test_normalizer_is_baked_in(policy):
    _, model, _, _ = policy
    ops = [n.op_type for n in model.graph.node[:2]]
    assert ops == ["Sub", "Div"], f"the graph should start with the observation normalizer, found {ops}"
    assert model.graph.node[0].input[0] == model.graph.input[0].name


def _stand_obs():
    obs = np.zeros((1, 61), dtype=np.float32)
    obs[0, 3:6] = [0.0, 0.0, -1.0]  # upright: gravity straight down in the trunk frame
    return obs


def test_outputs_are_finite_and_small_at_the_stand(policy):
    """At the stand pose with zero command, the targets stay near the stand pose (a policy that wants
    to throw a joint across its range from rest is miswired or mis-normalized)."""
    _, _, _, sess = policy
    name = sess.get_inputs()[0].name
    for obs in (np.zeros((1, 61), dtype=np.float32), _stand_obs()):
        act = sess.run(None, {name: obs})[0]
        assert np.all(np.isfinite(act))
    act = sess.run(None, {name: _stand_obs()})[0][0]
    assert np.max(np.abs(act)) < 0.5, f"stand action {np.round(act, 3)}"
    tilted = _stand_obs()
    tilted[0, 3:6] = [np.sin(0.2), 0.0, -np.cos(0.2)]  # trunk pitched: the policy must react
    assert np.max(np.abs(sess.run(None, {name: tilted})[0][0] - act)) > 1e-3
