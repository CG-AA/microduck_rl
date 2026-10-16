#!/usr/bin/env python3
"""Headless walking evaluation on CPU MuJoCo with the BAM actuators (no viewer, no keyboard).

Runs an exported walking ONNX (61-D obs, --new-cmd-obs layout) through a fixed command schedule,
a step-response schedule and two push tests, using scripts/infer_policy.py's own model loading,
observation and action code so it is the same rehearsal, scripted. Harness v2 reproduces what
training puts around the policy:

  * the position target reaches the servo model 3-6 physics substeps late, lag resampled every
    substep (mjlab's actuator delay, microduck_constants._BAM_ACTUATOR_KWARGS);
  * joint velocities in the observation are one control step old;
  * the default pose (action offset, joint_pos reference) is the policy's own, read from the ONNX
    metadata, so a policy trained on another stand pose is evaluated under it.

Per segment:
  tracking     mean body-frame vx / vy / yaw rate over the settled part of the segment vs command
  fell         trunk tilt > 45 deg or trunk below 6 cm (time of the fall)
  torque       per-joint |motor torque| [N.m] (BAM output, data.ctrl): v1's control-step RMS and
               peak over the whole segment (torque_rms, torque_peak) and, over the settled window at
               the substep, torque_rms_settled, torque_mean_abs_settled, torque_peak_substep and
               over_continuous_frac (fraction of substeps above the continuous rating)
  speed_util   max |qdot| / no-load speed and the fraction of substeps above 0.8 of it
  saturated    v1's duty-cycle proxy, kept for compatibility; on the HD-1910 it needs 2.18 rad of
               position error (0.98 / (156 x 0.00288)) and never fires
and, for the STEPS schedule, each command step's t10 / t90 / rise / overshoot / settle_t / ss_err /
ss_std on a 0.3 s causal moving average of the stepped axis.

Replicas perturb the initial joint positions by +-0.01 rad and reseed the delay. The JSON keeps the
v1 keys (`schedule`, `push` = replica 0) and adds `steps`, `push_stand`, every replica, and a mean
over replicas (`summary`), which `--compare` reads.

usage:
  uv run scripts/eval_walk.py --walking policy.onnx [--hd1910] [--replicas 3] [--json out.json] [--trace out.npz]
  uv run scripts/eval_walk.py --compare v1.json v2.json      # gate table, v2 judged against v1
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ip = _load("infer_policy")
st = _load("stand_torque")  # contact_height: spawn the robot on its soles whatever its default pose

HARNESS_VERSION = 2
DECIMATION = 4  # 50 Hz control on a 5 ms physics step, as in training
SETTLE = 0.4  # fraction of each segment ignored before measuring
MA_SECONDS = 0.3  # step responses: causal moving average, longer than a stride, shorter than a segment
ACTION_DELAY = (3, 6)  # physics substeps, as training (lag resampled every substep)
JOINT_VEL_LAG = 1  # control steps, as training's lagged joint_vel observation
SPAWN_Z = 0.125  # trunk height at reset for the stock stand pose
LEGS = ("left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
        "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle")
# (name, seconds, vx, vy, wz)
SCHEDULE = [
    ("stand", 3.0, 0.0, 0.0, 0.0),
    ("fwd 0.10", 6.0, 0.10, 0.0, 0.0),
    ("fwd 0.20", 6.0, 0.20, 0.0, 0.0),
    ("fwd 0.30", 6.0, 0.30, 0.0, 0.0),
    ("back 0.10", 5.0, -0.10, 0.0, 0.0),
    ("left 0.10", 5.0, 0.0, 0.10, 0.0),
    ("turn +1.0", 5.0, 0.0, 0.0, 1.0),
    ("turn -1.0", 5.0, 0.0, 0.0, -1.0),
    ("stand", 3.0, 0.0, 0.0, 0.0),
]
# Command steps from and to rest, and between speeds; every transition is one axis.
STEPS = [
    ("stand", 3.0, 0.0, 0.0, 0.0),
    ("fwd 0.10", 6.0, 0.10, 0.0, 0.0),
    ("stand", 3.0, 0.0, 0.0, 0.0),
    ("fwd 0.20", 6.0, 0.20, 0.0, 0.0),
    ("stand", 3.0, 0.0, 0.0, 0.0),
    ("fwd 0.30", 6.0, 0.30, 0.0, 0.0),
    ("stand", 3.0, 0.0, 0.0, 0.0),
    ("fwd 0.10", 4.0, 0.10, 0.0, 0.0),
    ("fwd 0.20", 4.0, 0.20, 0.0, 0.0),
    ("stand", 3.0, 0.0, 0.0, 0.0),
    ("back 0.10", 5.0, -0.10, 0.0, 0.0),
    ("stand", 3.0, 0.0, 0.0, 0.0),
    ("left 0.10", 5.0, 0.0, 0.10, 0.0),
    ("stand", 3.0, 0.0, 0.0, 0.0),
]
PUSHES = (0.2, 0.3, 0.4)  # m/s lateral velocity kicks, one per 3 s
PUSH_SECONDS = 3.0 * (len(PUSHES) + 1)
ROLLOUTS = (  # key, segments, pushes
    ("schedule", SCHEDULE, ()),
    ("steps", STEPS, ()),
    ("push", [("push walk", PUSH_SECONDS, 0.10, 0.0, 0.0)], PUSHES),
    ("push_stand", [("push stand", PUSH_SECONDS, 0.0, 0.0, 0.0)], PUSHES),
)


def actuator_limits(bam_model, continuous):
    """The limits that can bind on this servo: thermal (continuous), stall, no-load speed, and v1's duty proxy."""
    act = bam_model.actuator
    kt, R = float(bam_model.kt.value), float(bam_model.R.value)
    return dict(continuous=continuous, omega_nl=act.max_pwm * act.vin / kt, stall=kt * act.max_pwm * act.vin / R,
                duty_per_rad=act.kp * act.error_gain, max_pwm=act.max_pwm)


def onnx_default_pose(session, names):
    """The policy's default joint positions from scripts/export.py's metadata, in actuator order, or None."""
    meta = session.get_modelmeta().custom_metadata_map
    if "default_joint_pos" not in meta or "joint_names" not in meta:
        return None
    by_name = dict(zip(meta["joint_names"].split(","), (float(x) for x in meta["default_joint_pos"].split(","))))
    if any(n not in by_name for n in names):
        return None
    return np.array([by_name[n] for n in names], dtype=np.float32)


def build(args):
    kp_fw = args.kp_fw if args.kp_fw is not None else (ip.HD1910_KP_FW if args.hd1910 else ip.BAM_KP_FW)
    if args.hd1910:
        bam_model = ip.load_bam_model(kp_fw, args.vin, ip.HD1910_MAX_CURRENT,
                                      json_path=str(REPO / ip.HD1910_PARAMS_JSON), max_pwm=ip.HD1910_MAX_PWM)
        xml = REPO / ip.HD1910_XML
    else:
        bam_model = ip.load_bam_model(kp_fw, args.vin, ip.BAM_MAX_CURRENT)
        xml = REPO / ip.MICRODUCK_XML
    model, data, bam_ctrl, names = ip.load_mujoco_with_bam(
        str(xml), bam_model, 0.005, args.vin_drop_gain if args.vin_drop_gain > 0 else None, ip.BAM_VIN_MIN)
    policy = ip.PolicyInference(model, data, walking_onnx_path=args.walking, bam_ctrl=bam_ctrl, new_cmd_obs=True,
                                use_projected_gravity=True,  # infer_policy default (--raw-accelerometer off)
                                joint_vel_lag=getattr(args, "joint_vel_lag", JOINT_VEL_LAG),
                                default_pose=ip.HD1910_DEFAULT_POSE if args.hd1910 else None)
    pose = None if getattr(args, "script_pose", False) else onnx_default_pose(policy.walking_session, names)
    if pose is None:
        policy.default_pose_source = "script"
        print("default pose: the script's (not taken from the ONNX metadata)")
    else:
        policy.default_pose = pose
        policy.default_pose_source = "onnx"
        print(f"default pose from the ONNX metadata: {np.round(pose, 4).tolist()}")
    policy.spawn_z = st.contact_height(model, policy.default_pose) + 0.008  # soles 8 mm up, as 0.125 was for HOME
    policy.actuator_names = names
    policy.limits = actuator_limits(bam_model, getattr(args, "continuous", 0.29))
    policy.model_init = (model.dof_frictionloss.copy(), model.dof_damping.copy())
    return bam_model, model, data, bam_ctrl, policy


def reset(model, data, bam_ctrl, policy, joint_noise=0.0, rng=None):
    """Robot at rest in its default pose, trunk at spawn height; everything stateful cleared."""
    mujoco.mj_resetData(model, data)
    init = getattr(policy, "model_init", None)
    if init is not None:  # BAM rewrites these every update(); mj_resetData does not restore them
        model.dof_frictionloss[:], model.dof_damping[:] = init
    adr = model.jnt_qposadr[model.joint("trunk_base_freejoint").id]
    data.qpos[adr:adr + 7] = [0, 0, getattr(policy, "spawn_z", SPAWN_Z), 1, 0, 0, 0]
    q = policy.default_pose.astype(np.float64).copy()
    if joint_noise > 0:
        q += rng.uniform(-joint_noise, joint_noise, q.shape)
    data.qpos[policy.joint_qpos_indices] = q
    bam_ctrl.reset(data.qpos)
    bam_ctrl.last_ts = data.time
    policy.last_action = np.zeros(policy.n_joints, dtype=np.float32)
    policy.reset_history()
    policy.set_position_targets(policy.default_pose)
    mujoco.mj_forward(model, data)


class DelayLine:
    """Per-substep history of position targets; the servo sees the target `lag` substeps old (mjlab's actuator delay)."""

    def __init__(self, init, lag_range=ACTION_DELAY, rng=None):
        self.lo, self.hi = int(lag_range[0]), int(lag_range[1])
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.hist = np.tile(np.asarray(init, dtype=np.float64), (self.hi + 1, 1))
        self.i = 0

    def push_and_get(self, target):
        self.hist[self.i] = target
        lag = int(self.rng.integers(self.lo, self.hi + 1)) if self.hi > 0 else 0
        out = self.hist[(self.i - lag) % len(self.hist)]
        self.i = (self.i + 1) % len(self.hist)
        return out, lag


def control_step(policy, bam_ctrl, model, data, delay=None):
    """One 50 Hz control step: infer, then DECIMATION substeps on the (delayed) target. Returns substep samples."""
    action = policy.infer()
    target = (policy.default_pose + action * policy.action_scale).astype(np.float64)
    lim = getattr(policy, "limits", None)
    nu = policy.n_joints
    tau_abs, tau_sq, tau_max, qd_max = (np.zeros(nu) for _ in range(4))
    over, fast, lags = np.zeros(nu), np.zeros(nu), []
    for _ in range(DECIMATION):
        applied, lag = delay.push_and_get(target) if delay is not None else (target, 0)
        bam_ctrl.q_target[:] = applied
        bam_ctrl.update()
        mujoco.mj_step(model, data)
        tau = np.abs(data.ctrl)
        qd = np.abs(data.qvel[policy.joint_qvel_indices])
        tau_abs += tau
        tau_sq += tau ** 2
        np.maximum(tau_max, tau, out=tau_max)
        np.maximum(qd_max, qd, out=qd_max)
        if lim is not None:
            over += tau > lim["continuous"]
            fast += qd > 0.8 * lim["omega_nl"]
        lags.append(lag)
    return dict(action=action, target=target, applied=applied, tau=np.abs(data.ctrl).copy(),
                tau_abs=tau_abs / DECIMATION, tau_sq=tau_sq / DECIMATION, tau_max=tau_max, qd_max=qd_max,
                over=over / DECIMATION, fast=fast / DECIMATION, lag=float(np.mean(lags)))


def tilt_deg(R):
    return math.degrees(math.acos(float(np.clip(R[2, 2], -1.0, 1.0))))


def body_rates(data, trunk, dof):
    """Trunk linear and angular velocity in the body frame.

    MuJoCo's free-joint qvel holds the linear velocity in the WORLD frame and the angular velocity
    already in the BODY frame (mjlab's root_link_lin_vel_w / root_link_ang_vel_b read the same slots).
    """
    R = data.xmat[trunk].reshape(3, 3)
    return R.T @ data.qvel[dof:dof + 3], data.qvel[dof + 3:dof + 6].copy(), R


def run(segments, bam_model, model, data, bam_ctrl, policy, pushes=(), delay=None, trace=None):
    """Step through (name, seconds, vx, vy, wz) segments; returns per-segment metrics (+ `_vel` series)."""
    trunk = model.body("trunk_base").id
    dof = model.jnt_dofadr[model.joint("trunk_base_freejoint").id]
    lim = policy.limits
    names = policy.actuator_names
    dt = DECIMATION * model.opt.timestep
    out, fell_at, t = [], None, 0.0
    push_times = {round(3.0 * (i + 1) / dt): v for i, v in enumerate(pushes)}
    for si, (name, seconds, vx, vy, wz) in enumerate(segments):
        policy.set_vel_cmd(vx, vy, wz)
        n = int(round(seconds / dt))
        s = int(SETTLE * n)
        vel, tau, sat, tau_abs, tau_sq, tau_max, qd_max, over, fast, lag = ([] for _ in range(10))
        for k in range(n):
            if k in push_times:
                data.qvel[dof + 1] += push_times[k]  # world y kick
            smp = control_step(policy, bam_ctrl, model, data, delay)
            t += dt
            v_body, w_body, R = body_rates(data, trunk, dof)
            vel.append((v_body[0], v_body[1], w_body[2]))
            tau.append(smp["tau"])
            err = np.abs(np.asarray(bam_ctrl.q_target) - data.qpos[policy.joint_qpos_indices])
            sat.append(err * lim["duty_per_rad"] >= lim["max_pwm"])
            for buf, key in ((tau_abs, "tau_abs"), (tau_sq, "tau_sq"), (tau_max, "tau_max"), (qd_max, "qd_max"),
                             (over, "over"), (fast, "fast"), (lag, "lag")):
                buf.append(smp[key])
            tilt = tilt_deg(R)
            if fell_at is None and (tilt > 45.0 or data.xpos[trunk][2] < 0.06):
                fell_at = t
            if trace is not None:
                for key, val in (("t", t), ("seg", si), ("cmd", (vx, vy, wz)), ("v_body", vel[-1]),
                                 ("trunk_z", data.xpos[trunk][2]), ("tilt", tilt),
                                 ("qpos", data.qpos[policy.joint_qpos_indices].copy()),
                                 ("qvel", data.qvel[policy.joint_qvel_indices].copy()),
                                 ("action", smp["action"]), ("target", smp["target"]), ("tau", smp["tau"]),
                                 ("tau_submax", smp["tau_max"]), ("tau_subsq", smp["tau_sq"]), ("qd_submax", smp["qd_max"])):
                    trace.setdefault(key, []).append(np.asarray(val))
        vel, tau, sat = np.array(vel), np.array(tau), np.array(sat)
        tau_abs, tau_sq, tau_max, qd_max, over, fast = (np.array(b) for b in (tau_abs, tau_sq, tau_max, qd_max, over, fast))
        j = lambda a: dict(zip(names, np.round(a, 4).tolist()))
        out.append(dict(
            segment=name, cmd=[vx, vy, wz], fell_at=fell_at,
            mean=[float(x) for x in vel[s:].mean(0)],
            torque_rms=j(np.sqrt((tau ** 2).mean(0))),
            torque_peak=j(tau.max(0)),
            saturated=j(sat.mean(0)),
            torque_rms_settled=j(np.sqrt(tau_sq[s:].mean(0))),
            torque_mean_abs_settled=j(tau_abs[s:].mean(0)),
            torque_peak_substep=j(tau_max.max(0)),
            over_continuous_frac=j(over[s:].mean(0)),
            speed_util_max=float(qd_max.max() / lim["omega_nl"]),
            speed_util_frac=float(fast[s:].mean()),
            torque_over_stall_max=float(tau_max.max() / lim["stall"]),
            lag_mean=float(np.mean(lag)),
            _vel=vel,
        ))
        if fell_at is not None:
            break
    return out


def step_metrics(rows, dt):
    """Per command step: response of the stepped axis on a MA_SECONDS causal moving average."""
    m = max(1, int(round(MA_SECONDS / dt)))
    out = []
    for i in range(1, len(rows)):
        prev, cur = rows[i - 1], rows[i]
        d = np.asarray(cur["cmd"]) - np.asarray(prev["cmd"])
        axis = int(np.argmax(np.abs(d)))
        if abs(d[axis]) < 1e-9:
            continue
        raw = cur["_vel"][:, axis]
        prefix = prev["_vel"][-(m - 1):, axis] if m > 1 else raw[:0]
        if len(prefix) < m - 1:
            prefix = np.concatenate([np.full(m - 1 - len(prefix), prefix[0] if len(prefix) else raw[0]), prefix])
        r = np.convolve(np.concatenate([prefix, raw]), np.ones(m) / m, mode="valid")
        t = (np.arange(len(raw)) + 1) * dt
        c0, c1 = float(prev["cmd"][axis]), float(cur["cmd"][axis])
        # The response starts from what the robot was actually doing, not from the previous
        # command (a policy that ignored 0.1 m/s must not score an instant 0.1 -> 0 step).
        r0 = float(prefix[-min(len(prefix), m):].mean()) if len(prefix) else float(raw[0])
        d_eff = c1 - r0

        def first(mask):
            idx = np.flatnonzero(mask)
            return float(t[idx[0]]) if len(idx) else None

        def first_lasting(mask):  # first time after which mask holds for the rest of the segment
            ok = np.flip(np.cumprod(np.flip(mask))).astype(bool)
            return first(ok)

        if abs(d_eff) >= 0.02:
            frac = (r - r0) / d_eff
            t10, t90 = first(frac >= 0.1), first(frac >= 0.9)
            overshoot = float(max(0.0, frac.max() - 1.0))
        else:  # nothing to respond to: the robot already was (within 2 cm/s) where the command points
            t10 = t90 = overshoot = None
        tol = max(0.03, 0.2 * abs(d[axis]))
        half = len(raw) // 2
        out.append(dict(
            step=f"{prev['segment']} -> {cur['segment']}", axis=("vx", "vy", "wz")[axis],
            from_cmd=c0, from_meas=r0, to_cmd=c1,
            t10=t10, t90=t90, rise=(t90 - t10) if (t10 is not None and t90 is not None) else None,
            overshoot=overshoot,
            settle_t=first_lasting(np.abs(r - c1) < tol),
            ss_err=float(c1 - raw[half:].mean()), ss_std=float(raw[half:].std()),
            t_stop=first_lasting(np.abs(r) < 0.02) if abs(c1) < 1e-9 else None,
        ))
    return out


def strip(rows):
    return [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]


def _mean_tree(items):
    """Elementwise mean over same-shaped nested dict/list/number structures; None is skipped, strings kept."""
    items = [x for x in items if x is not None]
    if not items:
        return None
    x = items[0]
    if isinstance(x, dict):
        return {k: _mean_tree([it.get(k) for it in items]) for k in x}
    if isinstance(x, (list, tuple)):
        return [_mean_tree([it[i] for it in items if i < len(it)]) for i in range(len(x))]
    if isinstance(x, bool) or isinstance(x, str):
        return x
    if isinstance(x, (int, float)):
        return float(np.mean(items))
    return x


def aggregate(reps):
    """Mean over replicas, segment by segment (a replica that fell contributes the segments it reached)."""
    out = []
    for jdx in range(max(len(r) for r in reps)):
        have = [r[jdx] for r in reps if len(r) > jdx]
        row = _mean_tree([{k: v for k, v in h.items() if k != "fell_at"} for h in have])
        fells = [h["fell_at"] for h in have if h.get("fell_at") is not None]
        row["fell_at"] = min(fells) if fells else None
        row["fell_count"], row["n"] = len(fells), len(have)
        out.append(row)
    return out


def report(title, rows, continuous):
    print(f"\n== {title}")
    print(f"{'segment':11s} {'cmd vx vy wz':>17s} {'got vx vy wz':>20s}  fell   settled rms/joint  peak(sub)  speed util")
    for r in rows:
        rms = max(r["torque_rms_settled"].items(), key=lambda kv: kv[1])
        peak = max(r["torque_peak_substep"].values())
        fell = "-" if r["fell_at"] is None else f"{r['fell_at']:.1f}s"
        print(f"{r['segment']:11s} {r['cmd'][0]:+5.2f} {r['cmd'][1]:+5.2f} {r['cmd'][2]:+5.2f}  "
              f"{r['mean'][0]:+6.3f} {r['mean'][1]:+6.3f} {r['mean'][2]:+6.2f}  {fell:5s}  "
              f"{rms[1]:.3f} {rms[0]:15s} {peak:.3f}      {r['speed_util_max']:.2f}")
    worst = max((v, k) for r in rows for k, v in r["torque_rms_settled"].items())
    print(f"worst settled RMS torque {worst[0]:.3f} N.m ({worst[1]}) vs {continuous} N.m continuous rating")


def report_steps(metrics):
    print("\n== step responses (0.3 s moving average of the stepped axis)")
    print(f"{'step':24s} {'axis':4s} {'t10':>6s} {'t90':>6s} {'over':>6s} {'settle':>7s} {'ss_err':>7s} {'ss_std':>7s} {'t_stop':>7s}")
    f = lambda v, w=6: f"{v:{w}.2f}" if v is not None else f"{'-':>{w}s}"
    for s in metrics:
        print(f"{s['step']:24s} {s['axis']:4s} {f(s['t10'])} {f(s['t90'])} {f(s['overshoot'])} {f(s['settle_t'], 7)} "
              f"{f(s['ss_err'], 7)} {f(s['ss_std'], 7)} {f(s['t_stop'], 7)}")


# ---- comparison against a baseline JSON (the gate table of the first plan) ----------------------

def _rows(doc, key):
    summ = doc.get("summary", {})
    return summ.get(key) if key in summ else doc.get(key)


def _seg(rows, name, nth=0):
    hits = [r for r in (rows or []) if r.get("segment") == name]
    return hits[nth] if len(hits) > nth else None


def _stand_torque(doc):
    row = _seg(_rows(doc, "schedule"), "stand")
    if row is None:
        return None, None
    tq = row.get("torque_mean_abs_settled") or row.get("torque_rms")
    return max(tq.values()), sum(v for k, v in tq.items() if k in LEGS)


def _worst(doc, key, fallback):
    vals = [max((r.get(key) or r.get(fallback)).values()) for k in ("schedule", "steps") for r in (_rows(doc, k) or [])]
    return max(vals) if vals else None


def _falls(doc):
    """(falls at the schedule/steps/push_stand rollouts or the 0.2/0.3 kicks, falls at the 0.4 kick)"""
    hard, soft = 0, 0
    for key in ("schedule", "steps", "push_stand"):
        hard += sum(1 for r in (_rows(doc, key) or []) if r.get("fell_at") is not None)
    for r in (_rows(doc, "push") or []):
        if r.get("fell_at") is not None:
            if r["fell_at"] <= 3.0 * len(PUSHES) + 0.05:
                hard += 1
            else:
                soft += 1
    return hard, soft


def _step(doc, name, field):
    for s in (_rows(doc, "steps_metrics") or []):
        if s["step"] == name:
            return s.get(field)
    return None


def gate_rows(a, b):
    """[(metric, value_a, value_b, verdict)] — b judged against a and against the plan's thresholds."""
    cont = float(b.get("args", {}).get("continuous", 0.29))
    g = lambda doc, key, name, i, nth=0: (lambda r: None if r is None else r["mean"][i])(_seg(_rows(doc, key), name, nth))
    turn = lambda doc: [abs(x) for x in (g(doc, "schedule", "turn +1.0", 2), g(doc, "schedule", "turn -1.0", 2)) if x is not None]
    drift = lambda doc: (lambda r: None if r is None else math.hypot(r["mean"][0], r["mean"][1]))(_seg(_rows(doc, "schedule"), "stand"))
    specs = [
        ("fwd 0.10 vx [m/s]", lambda d: g(d, "schedule", "fwd 0.10", 0), lambda x, y: y >= 0.07),
        ("back 0.10 vx [m/s]", lambda d: g(d, "schedule", "back 0.10", 0), lambda x, y: y <= -0.06),
        ("left 0.10 vy [m/s]", lambda d: g(d, "schedule", "left 0.10", 1), lambda x, y: y >= 0.06),
        ("step 0->0.10 t90 [s]", lambda d: _step(d, "stand -> fwd 0.10", "t90"), lambda x, y: y <= 1.5),
        ("step 0->0.10 |ss_err| [m/s]", lambda d: (lambda v: None if v is None else abs(v))(_step(d, "stand -> fwd 0.10", "ss_err")), lambda x, y: y <= 0.03),
        ("step 0.20->0 t_stop [s]", lambda d: _step(d, "fwd 0.20 -> stand", "t_stop"), lambda x, y: y <= 1.0),
        ("fwd 0.20 vx [m/s]", lambda d: g(d, "schedule", "fwd 0.20", 0), lambda x, y: x is not None and y >= x - 0.02),
        ("fwd 0.30 vx [m/s]", lambda d: g(d, "schedule", "fwd 0.30", 0), lambda x, y: x is not None and y >= x - 0.03),
        ("turn |wz| mean [rad/s]", lambda d: (lambda v: float(np.mean(v)) if v else None)(turn(d)), lambda x, y: y >= 0.5 and (x is None or y >= x - 0.25)),
        ("turn |wz| min [rad/s]", lambda d: (lambda v: float(min(v)) if v else None)(turn(d)), lambda x, y: y >= 0.3 and (x is None or y >= x - 0.25)),
        ("stand drift |v_xy| [m/s]", drift, lambda x, y: y <= 0.01),
        ("stand drift |wz| [rad/s]", lambda d: (lambda v: None if v is None else abs(v))(g(d, "schedule", "stand", 2)), lambda x, y: y <= 0.02),
        ("falls (sched/steps/stand/kick<=0.3)", lambda d: _falls(d)[0], lambda x, y: y == 0),
        ("falls at the 0.4 kick (report)", lambda d: _falls(d)[1], None),
        ("stand torque max joint [N.m]", lambda d: _stand_torque(d)[0], lambda x, y: x is not None and y <= 0.6 * x),
        ("stand torque leg sum [N.m]", lambda d: _stand_torque(d)[1], lambda x, y: x is not None and y <= 0.6 * x),
        ("settled RMS worst [N.m]", lambda d: _worst(d, "torque_rms_settled", "torque_rms"), lambda x, y: y <= cont),
        ("substep peak worst [N.m]", lambda d: _worst(d, "torque_peak_substep", "torque_peak"), lambda x, y: "warn" if 0.6 < y <= 0.9 else y <= 0.6),
        ("speed_util_max (report)", lambda d: max((r["speed_util_max"] for k in ("schedule", "steps") for r in (_rows(d, k) or []) if "speed_util_max" in r), default=None), None),
        ("speed_util_frac", lambda d: max((r["speed_util_frac"] for k in ("schedule", "steps") for r in (_rows(d, k) or []) if "speed_util_frac" in r), default=None), lambda x, y: y <= 0.05),
    ]
    rows = []
    for name, getter, rule in specs:
        try:
            x = getter(a)
        except Exception:
            x = None
        try:
            y = getter(b)
        except Exception:
            y = None
        if rule is None or y is None:
            verdict = "report" if rule is None else "n/a"
        else:
            v = rule(x, y)
            verdict = v if isinstance(v, str) else ("pass" if v else "FAIL")
        rows.append((name, x, y, verdict))
    return rows


def compare(path_a, path_b):
    a, b = json.loads(Path(path_a).read_text()), json.loads(Path(path_b).read_text())
    fmt = lambda v: "n/a" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
    print(f"{'metric':36s} {'A':>9s} {'B':>9s}  verdict   (A = {path_a}, B = {path_b})")
    rows = gate_rows(a, b)
    for name, x, y, verdict in rows:
        print(f"{name:36s} {fmt(x):>9s} {fmt(y):>9s}  {verdict}")
    fails = [r[0] for r in rows if r[3] == "FAIL"]
    print(f"\n{len(fails)} failing: {fails}" if fails else "\nevery gated row passes")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--walking", help="walking policy ONNX (exported with scripts/export.py)")
    ap.add_argument("--hd1910", action="store_true", help="HD-1910 model + actuator (as the -HD1910 tasks)")
    ap.add_argument("--vin", type=float, default=7.4)
    ap.add_argument("--vin-drop-gain", type=float, default=0.1)
    ap.add_argument("--kp-fw", type=float, default=None)
    ap.add_argument("--continuous", type=float, default=0.29, help="continuous torque rating to report against [N.m]")
    ap.add_argument("--replicas", type=int, default=3, help="rollouts per schedule; replica k>0 perturbs the start pose by +-0.01 rad")
    ap.add_argument("--action-delay", type=int, nargs=2, default=list(ACTION_DELAY), metavar=("MIN", "MAX"),
                    help="target delay in physics substeps, resampled every substep (training: 3 6; 0 0 = none)")
    ap.add_argument("--joint-vel-lag", type=int, default=JOINT_VEL_LAG, help="joint_vel obs lag in control steps (training: 1)")
    ap.add_argument("--script-pose", action="store_true", help="ignore the ONNX default_joint_pos, use the script's DEFAULT_POSE")
    ap.add_argument("--json", help="write all metrics here")
    ap.add_argument("--trace", help="write per-control-step time series here (.npz)")
    ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"), help="print the gate table of B against A and exit")
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
        return
    if not args.walking:
        ap.error("--walking is required (or --compare A B)")

    bam_model, model, data, bam_ctrl, policy = build(args)
    dt = DECIMATION * model.opt.timestep
    results = {key: [] for key, _, _ in ROLLOUTS}
    results["steps_metrics"] = []
    trace_all = {}
    for rep in range(args.replicas):
        rng = np.random.default_rng(rep)
        for ri, (key, segments, pushes) in enumerate(ROLLOUTS):
            reset(model, data, bam_ctrl, policy, joint_noise=0.01 if rep else 0.0, rng=rng)
            delay = DelayLine(policy.default_pose, args.action_delay, np.random.default_rng(1000 * rep + ri))
            trace = {} if args.trace else None
            rows = run(segments, bam_model, model, data, bam_ctrl, policy, pushes=pushes, delay=delay, trace=trace)
            if key == "steps":
                results["steps_metrics"].append(step_metrics(rows, dt))
            results[key].append(strip(rows))
            if trace:
                for f, vals in trace.items():
                    trace_all[f"{key}/r{rep}/{f}"] = np.array(vals)
    summary = {key: aggregate(reps) for key, reps in results.items()}

    report("command schedule (replica mean)", summary["schedule"], args.continuous)
    report_steps(summary["steps_metrics"])
    report(f"push test: lateral kicks {PUSHES} m/s every 3 s at 0.1 m/s", summary["push"], args.continuous)
    report(f"push test: lateral kicks {PUSHES} m/s every 3 s standing", summary["push_stand"], args.continuous)
    falls = [(key, k, r["fell_at"]) for key, reps in results.items() if key != "steps_metrics"
             for k, rep in enumerate(reps) for r in rep if r["fell_at"] is not None]
    print(f"\nfalls: {falls if falls else 'none'}   default pose: {policy.default_pose_source}")
    if args.json:
        doc = dict(args=vars(args), harness_version=HARNESS_VERSION, default_pose_source=policy.default_pose_source,
                   default_pose=np.round(policy.default_pose, 6).tolist(), actuator_names=policy.actuator_names,
                   limits=policy.limits, dt=dt,
                   schedule=results["schedule"][0], push=results["push"][0],
                   steps=results["steps"][0], push_stand=results["push_stand"][0],
                   replicas=results, summary=summary)
        Path(args.json).write_text(json.dumps(doc, indent=1))
        print("wrote", args.json)
    if args.trace:
        trace_all["names"] = np.array(policy.actuator_names)
        trace_all["dt"] = np.array(dt)
        trace_all["rollouts"] = np.array(json.dumps({k: s for k, s, _ in ROLLOUTS}))
        np.savez_compressed(args.trace, **trace_all)
        print("wrote", args.trace)


if __name__ == "__main__":
    main()
