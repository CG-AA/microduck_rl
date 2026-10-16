#!/usr/bin/env python3
"""Headless walking evaluation on CPU MuJoCo with the BAM actuators (no viewer, no keyboard).

Runs an exported walking ONNX (61-D obs, --new-cmd-obs layout) through a fixed command schedule
and a push test, using scripts/infer_policy.py's own model loading, observation and action code
so it is the same rehearsal, scripted. Reports per segment:

  tracking   mean body-frame vx / vy / yaw rate over the settled part of the segment vs command
  fell       trunk tilt > 45 deg or trunk below 6 cm
  torque     per-joint RMS and peak |motor torque| [N.m] (BAM output, data.ctrl)
  saturated  fraction of time each joint's firmware duty cycle sits at max_pwm

usage:
  uv run scripts/eval_walk.py --walking policy.onnx [--hd1910] [--vin 7.4] [--json out.json]
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("infer_policy", REPO / "scripts" / "infer_policy.py")
ip = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ip)

DECIMATION = 4  # 50 Hz control on a 5 ms physics step, as in training
SETTLE = 0.4  # fraction of each segment ignored before measuring tracking
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
PUSHES = (0.2, 0.3, 0.4)  # m/s lateral velocity kicks while walking at 0.1 m/s, one per 3 s


def build(args):
    kp_fw = args.kp_fw if args.kp_fw is not None else (ip.HD1910_KP_FW if args.hd1910 else ip.BAM_KP_FW)
    if args.hd1910:
        bam_model = ip.load_bam_model(kp_fw, args.vin, ip.HD1910_MAX_CURRENT,
                                      json_path=str(REPO / ip.HD1910_PARAMS_JSON), max_pwm=ip.HD1910_MAX_PWM)
        xml = REPO / ip.HD1910_XML
    else:
        bam_model = ip.load_bam_model(kp_fw, args.vin, ip.BAM_MAX_CURRENT)
        xml = REPO / ip.MICRODUCK_XML
    model, data, bam_ctrl, _ = ip.load_mujoco_with_bam(
        str(xml), bam_model, 0.005, args.vin_drop_gain if args.vin_drop_gain > 0 else None, ip.BAM_VIN_MIN)
    policy = ip.PolicyInference(model, data, walking_onnx_path=args.walking, bam_ctrl=bam_ctrl, new_cmd_obs=True,
                                use_projected_gravity=True)  # infer_policy default (--raw-accelerometer off)
    return bam_model, model, data, bam_ctrl, policy


def reset(model, data, bam_ctrl, policy):
    mujoco.mj_resetData(model, data)
    adr = model.jnt_qposadr[model.joint("trunk_base_freejoint").id]
    data.qpos[adr:adr + 7] = [0, 0, 0.125, 1, 0, 0, 0]
    data.qpos[policy.joint_qpos_indices] = policy.default_pose
    bam_ctrl.reset(data.qpos)
    policy.last_action = np.zeros(policy.n_joints, dtype=np.float32)
    policy.set_position_targets(policy.default_pose)
    mujoco.mj_forward(model, data)


def run(segments, bam_model, model, data, bam_ctrl, policy, pushes=()):
    """Step through (name, seconds, vx, vy, wz) segments; returns per-segment metrics."""
    trunk = model.body("trunk_base").id
    dof = model.jnt_dofadr[model.joint("trunk_base_freejoint").id]
    act = bam_model.actuator
    duty_per_rad = act.kp * act.error_gain
    names = [model.actuator(i).name for i in range(model.nu)]
    dt = DECIMATION * model.opt.timestep
    out, fell_at, t = [], None, 0.0
    push_times = {round(3.0 * (i + 1) / dt): v for i, v in enumerate(pushes)}
    for name, seconds, vx, vy, wz in segments:
        policy.set_vel_cmd(vx, vy, wz)
        n = int(round(seconds / dt))
        vel, tau, sat = [], [], []
        for k in range(n):
            if k in push_times:
                data.qvel[dof + 1] += push_times[k]  # world y kick
            policy.apply_action(policy.infer())
            for _ in range(DECIMATION):
                bam_ctrl.update()
                mujoco.mj_step(model, data)
            t += dt
            R = data.xmat[trunk].reshape(3, 3)
            v_body = R.T @ data.qvel[dof:dof + 3]
            w_body = R.T @ data.qvel[dof + 3:dof + 6]
            vel.append((v_body[0], v_body[1], w_body[2]))
            tau.append(np.abs(data.ctrl).copy())
            err = np.abs(np.asarray(bam_ctrl.q_target) - data.qpos[policy.joint_qpos_indices])
            sat.append(err * duty_per_rad >= act.max_pwm)
            if fell_at is None and (R[2, 2] < math.cos(math.radians(45)) or data.xpos[trunk][2] < 0.06):
                fell_at = t
        vel, tau, sat = np.array(vel), np.array(tau), np.array(sat)
        s = int(SETTLE * n)
        out.append(dict(
            segment=name, cmd=[vx, vy, wz], fell_at=fell_at,
            mean=[float(x) for x in vel[s:].mean(0)],
            torque_rms=dict(zip(names, np.sqrt((tau ** 2).mean(0)).round(4).tolist())),
            torque_peak=dict(zip(names, tau.max(0).round(4).tolist())),
            saturated=dict(zip(names, sat.mean(0).round(4).tolist())),
        ))
        if fell_at is not None:
            break
    return out


def report(title, rows, continuous):
    print(f"\n== {title}")
    print(f"{'segment':11s} {'cmd vx vy wz':>17s} {'got vx vy wz':>20s}  fell   max rms/joint   max peak   max sat")
    for r in rows:
        rms, peak, sat = (max(r[k].items(), key=lambda kv: kv[1]) for k in ("torque_rms", "torque_peak", "saturated"))
        print(f"{r['segment']:11s} {r['cmd'][0]:+5.2f} {r['cmd'][1]:+5.2f} {r['cmd'][2]:+5.2f}  "
              f"{r['mean'][0]:+6.3f} {r['mean'][1]:+6.3f} {r['mean'][2]:+6.2f}  "
              f"{'-' if r['fell_at'] is None else f'{r['fell_at']:.1f}s':5s}  "
              f"{rms[1]:.3f} {rms[0]:15s} {peak[1]:.3f}      {sat[1]:.0%} {sat[0]}")
    worst = max((v, k) for r in rows for k, v in r["torque_rms"].items())
    print(f"worst per-segment RMS torque {worst[0]:.3f} N.m ({worst[1]}) vs {continuous} N.m continuous rating")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--walking", required=True, help="walking policy ONNX (exported with scripts/export.py)")
    ap.add_argument("--hd1910", action="store_true", help="HD-1910 model + actuator (as the -HD1910 tasks)")
    ap.add_argument("--vin", type=float, default=7.4)
    ap.add_argument("--vin-drop-gain", type=float, default=0.1)
    ap.add_argument("--kp-fw", type=float, default=None)
    ap.add_argument("--continuous", type=float, default=0.29, help="continuous torque rating to report against [N.m]")
    ap.add_argument("--json", help="write all metrics here")
    args = ap.parse_args()

    bam_model, model, data, bam_ctrl, policy = build(args)
    reset(model, data, bam_ctrl, policy)
    sched = run(SCHEDULE, bam_model, model, data, bam_ctrl, policy)
    reset(model, data, bam_ctrl, policy)
    push = run([("push walk", 3.0 * (len(PUSHES) + 1), 0.10, 0.0, 0.0)], bam_model, model, data, bam_ctrl,
               policy, pushes=PUSHES)
    report("command schedule", sched, args.continuous)
    report(f"push test: lateral kicks {PUSHES} m/s every 3 s at 0.1 m/s", push, args.continuous)
    if args.json:
        Path(args.json).write_text(json.dumps(dict(args=vars(args), schedule=sched, push=push), indent=1))
        print("wrote", args.json)


if __name__ == "__main__":
    main()
