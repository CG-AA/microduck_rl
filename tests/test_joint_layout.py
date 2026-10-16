"""Joint order and mirror sign, locked across every place they are spelled out.

Action i of a policy is the i-th actuated joint in joint-tree order for training
(mjlab ``Entity.find_joints_by_actuator_names``), the i-th ``<actuator>`` for the
CPU rehearsal (``scripts/infer_policy.py`` maps through ``actuator_trnid``), the
i-th entry of ``DEFAULT_POSE`` / ``HOME_FRAME`` / the scene STAND keyframe, and
the i-th slot of the mirror tables in ``tasks/symmetry.py``. Nothing but
convention keeps those in agreement: a reordered Onshape export, a renamed joint
or a flipped hinge axis trains a policy that looks fine in the viewer and falls
over on hardware. These tests pin all of them to ``CANONICAL`` and derive the
mirror signs from the actual hinge axes instead of trusting the hand-written table.
"""

import glob
import importlib.util
import os
import re
from pathlib import Path

import mujoco
import numpy as np
import pytest

import mjlab_microduck.tasks  # noqa: F401  (load tasks first: avoids the constants<->tasks import cycle)
from mjlab.entity import Entity
from mjlab.utils.string import resolve_expr
from mjlab_microduck.robot import microduck_constants as md
from mjlab_microduck.robot import microduck_hd1910_constants as hd
from mjlab_microduck.tasks.symmetry import _JOINT_PERM, _JOINT_SIGN

REPO = Path(__file__).resolve().parents[1]

CANONICAL = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]

MODELS = {
    "hd1910_walk": hd.HD1910_WALK_XML,
    "hd1910_backlash": hd.HD1910_WALK_BACKLASH_XML,
    "hd1910_scene": hd.HD1910_SCENE_XML,
    "stock_walk": md.MICRODUCK_WALK_XML,
    "stock_backlash": md.MICRODUCK_WALK_BACKLASH_XML,
    "stock_scene": md.MICRODUCK_WALK_XML.parent / "scene_walk.xml",
    "stock_scene_groundcontact": md.MICRODUCK_WALK_XML.parent / "scene.xml",  # infer_policy's default
}

MIRROR = np.diag([1.0, -1.0, 1.0])  # sagittal-plane reflection, y -> -y


def _find_onnx():
    """HD-1910 ONNX for the metadata / rehearsal tests: env override, else newest runs/hd1910_*.onnx."""
    path = os.environ.get("MICRODUCK_TEST_ONNX")
    if path:
        return path
    found = sorted(glob.glob(str(REPO / "runs" / "hd1910_*.onnx")), key=os.path.getmtime)
    return found[-1] if found else None


def _hinges(model):
    return [model.joint(j).name for j in range(model.njnt) if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]


def _resolve(init_state):
    """init_state.joint_pos over CANONICAL, first-match-wins exactly as Entity resolves it."""
    return np.array(resolve_expr(init_state.joint_pos, tuple(CANONICAL), 0.0))


@pytest.fixture(scope="module")
def ip():
    spec = importlib.util.spec_from_file_location("infer_policy", REPO / "scripts" / "infer_policy.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def robots():
    """Training-side entities (CPU, no sim) for the two HD-1910 cfgs."""
    return {"walk": Entity(hd.HD1910_WALK_ROBOT_CFG), "backlash": Entity(hd.HD1910_WALK_BACKLASH_ROBOT_CFG)}


@pytest.fixture(scope="module")
def home(robots):
    """HD-1910 walk init_state over CANONICAL, cross-checked against the compiled init_state keyframe."""
    pose = _resolve(hd.HD1910_WALK_ROBOT_CFG.init_state)
    key = robots["walk"].spec.compile().key("init_state")
    np.testing.assert_allclose(key.qpos[7:], pose, atol=1e-6)
    np.testing.assert_allclose(key.ctrl, pose, atol=1e-6)
    return pose


@pytest.fixture(scope="module")
def cpu_sim(ip):
    """The --hd1910 CPU rehearsal model (BAM-converted actuators) and its actuator names."""
    bam = ip.load_bam_model(ip.HD1910_KP_FW, 7.4, ip.HD1910_MAX_CURRENT,
                            json_path=str(REPO / ip.HD1910_PARAMS_JSON), max_pwm=ip.HD1910_MAX_PWM)
    return ip.load_mujoco_with_bam(str(REPO / ip.HD1910_XML), bam, 0.005, None, 6.0)


@pytest.mark.parametrize("name", list(MODELS))
def test_xml_actuator_and_joint_order(name):
    model = mujoco.MjModel.from_xml_path(str(MODELS[name]))
    hinges = _hinges(model)
    servo = [n for n in hinges if not n.startswith("passive_")]
    actuators = [model.actuator(i).name for i in range(model.nu)]
    driven = [model.joint(int(model.actuator_trnid[i, 0])).name for i in range(model.nu)]
    assert actuators == driven == servo == CANONICAL
    passive = [n for n in hinges if n.startswith("passive_")]
    if "backlash" in name:
        assert len(passive) == 14 and all(re.fullmatch(r"passive_.*_backlash", n) for n in passive)
        assert hinges == [x for n in CANONICAL for x in (n, f"passive_{n}_backlash")]  # servo, then its play
        stride = 2
    else:
        assert passive == []
        stride = 1  # ctrl idx == joint idx (AGENTS.md)
    assert [int(model.jnt_qposadr[model.joint(n).id]) for n in CANONICAL] == [7 + stride * i for i in range(14)]


def test_training_action_order_matches_rehearsal(robots, cpu_sim):
    expr = hd.hd1910_actuators.target_names_expr
    assert expr == hd.hd1910_backlash_actuators.target_names_expr == md._BAM_ACTUATOR_KWARGS["target_names_expr"]
    for robot in robots.values():
        ids, names = robot.find_joints_by_actuator_names(expr)
        assert names == CANONICAL
        assert [robot.joint_names[i] for i in ids] == CANONICAL
        assert list(robot.actuator_names) == CANONICAL
    model, _, _, names = cpu_sim
    assert list(names) == CANONICAL
    assert [model.joint(int(model.actuator_trnid[i, 0])).name for i in range(model.nu)] == CANONICAL


def test_rehearsal_qpos_indices_follow_actuator_order(ip, cpu_sim):
    onnx = _find_onnx()
    if onnx is None:
        pytest.skip("no HD-1910 ONNX: set MICRODUCK_TEST_ONNX or export one to runs/hd1910_*.onnx")
    pytest.importorskip("onnxruntime")
    model, data, bam_ctrl, _ = cpu_sim
    policy = ip.PolicyInference(model, data, walking_onnx_path=onnx, bam_ctrl=bam_ctrl,
                                new_cmd_obs=True, use_projected_gravity=True)
    assert policy.joint_qpos_indices == [int(model.jnt_qposadr[model.joint(n).id]) for n in CANONICAL]
    assert policy.joint_qvel_indices == [int(model.jnt_dofadr[model.joint(n).id]) for n in CANONICAL]


def test_onnx_metadata_joint_layout(home):
    onnx = _find_onnx()
    if onnx is None:
        pytest.skip("no HD-1910 ONNX: set MICRODUCK_TEST_ONNX or export one to runs/hd1910_*.onnx")
    ort = pytest.importorskip("onnxruntime")
    meta = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"]).get_modelmeta().custom_metadata_map
    assert meta["joint_names"].split(",") == CANONICAL
    exported = [float(x) for x in meta["default_joint_pos"].split(",")]  # rounded to 3 decimals at export
    np.testing.assert_allclose(exported, home, atol=5e-4)
    if "observation_names" in meta:
        obs = meta["observation_names"].split(",")
        assert obs and obs.index("joint_pos") < obs.index("joint_vel") < obs.index("actions")


@pytest.mark.parametrize("xml", [hd.HD1910_WALK_XML, md.MICRODUCK_WALK_XML], ids=["hd1910", "stock"])
def test_mirror_sign_table_matches_geometry(xml):
    """Derive the symmetry signs from the hinge axes at qpos0 and compare with _JOINT_SIGN.

    Reflecting a rotation by theta about axis a through the sagittal plane gives a
    rotation by -theta about M.a. So if the partner's axis equals M.a the mirrored
    joint angle is negated (sign -1: yaw, roll, and the pitch pairs whose axes point
    opposite ways); if it equals -M.a the angle is kept (sign +1: midline pitch).
    """
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    jid = [model.joint(n).id for n in CANONICAL]
    anchor, axis = data.xanchor[jid], data.xaxis[jid]
    swap = lambda n: n.replace("left_", "X").replace("right_", "left_").replace("X", "right_")  # noqa: E731
    assert [CANONICAL[j] for j in _JOINT_PERM] == [swap(n) for n in CANONICAL]
    assert [_JOINT_PERM[j] for j in _JOINT_PERM] == list(range(14))
    signs = []
    for i, j in enumerate(_JOINT_PERM):
        # The mirrored anchor must lie on the partner's hinge line (a hinge anchor is free along its axis).
        assert np.linalg.norm(np.cross(MIRROR @ anchor[i] - anchor[j], axis[j])) < 1e-6, CANONICAL[i]
        if np.allclose(axis[j], MIRROR @ axis[i], atol=1e-6):
            signs.append(-1.0)
        else:
            assert np.allclose(axis[j], -(MIRROR @ axis[i]), atol=1e-6), CANONICAL[i]
            signs.append(1.0)
    assert signs == _JOINT_SIGN


def test_home_poses_are_mirror_symmetric(ip, home):
    perm, sign = np.array(_JOINT_PERM), np.array(_JOINT_SIGN)
    poses = {
        "infer_policy.DEFAULT_POSE": ip.DEFAULT_POSE,
        "HOME_FRAME": _resolve(md.HOME_FRAME),
        "BACKLASH_HOME_FRAME": _resolve(md.BACKLASH_HOME_FRAME),
        "HD1910_WALK init_state": home,
        "HD1910_WALK_BACKLASH init_state": _resolve(hd.HD1910_WALK_BACKLASH_ROBOT_CFG.init_state),
    }
    for name, pose in poses.items():
        pose = np.asarray(pose, dtype=np.float64)
        assert pose.shape == (14,), name
        np.testing.assert_allclose(pose, sign * pose[perm], atol=1e-6, err_msg=name)


def test_scene_stand_keyframe_matches_init_state(ip, home):
    key = mujoco.MjModel.from_xml_path(str(hd.HD1910_SCENE_XML)).key("STAND")
    np.testing.assert_allclose(key.qpos[7:], home, atol=1e-4)
    np.testing.assert_allclose(key.ctrl, home, atol=1e-4)
    # The backlash HOME must pin passive_*_backlash first: HOME's unanchored patterns would
    # otherwise push a +-1 deg hinge to the servo's home angle.
    assert next(iter(hd.HD1910_WALK_BACKLASH_ROBOT_CFG.init_state.joint_pos)) == r".*_backlash$"
    assert hd.HD1910_WALK_ROBOT_CFG.init_state == md.HOME_FRAME  # same body, same home as stock
    # Stock side: DEFAULT_POSE (CPU rehearsal), HOME_FRAME (training) and both STAND keyframes agree.
    stock_home = _resolve(md.HOME_FRAME)
    np.testing.assert_allclose(ip.DEFAULT_POSE, stock_home, atol=1e-4)
    for scene in (MODELS["stock_scene"], MODELS["stock_scene_groundcontact"]):
        key = mujoco.MjModel.from_xml_path(str(scene)).key("STAND")
        np.testing.assert_allclose(key.qpos[7:], stock_home, atol=1e-4, err_msg=str(scene))
        np.testing.assert_allclose(key.ctrl, stock_home, atol=1e-4, err_msg=str(scene))
