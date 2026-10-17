#!/usr/bin/env python3
"""Stand-pose torque tool: which standing pose costs the HD-1910 servos the least heat?

Holds a candidate pose on CPU MuJoCo with the BAM servo model and reads the motor torque each
joint needs to keep it, then sweeps a small grid of poses and ranks them. Two facts shape it:

  * At the training gain (0.55 N.m/rad per joint) the servo PD alone cannot hold any standing
    pose: the robot's gravity stiffness m*g*h is about 1.07 N.m/rad, so a PD-only hold tips over
    in ~1.4 s (HOME included). The policy holds the pose by acting, so what matters is the gravity
    moment at each joint in that pose. A stiff hold (kp_scale 8) reads exactly that, and the
    reading has converged by kp_scale 4.
  * The sagittal chain is hip_pitch, knee, ankle(pitch); there is no ankle roll. A flat sole means
    ankle = knee - hip_pitch (left-leg sign; the right leg is the mirror). The 5 deg inward
    hip_roll is what sets the sole flat laterally, so it is pinned unless --sweep-roll shows
    a better value.

Objective: the sum over the ten leg joints of the mean squared torque (heat), tie-broken by the
largest joint mean |torque|. A candidate passes when it is a static equilibrium from the nominal
start and from perturbed starts (trunk tilted +-3 deg, joint noise +-0.05 rad): no fall, final
tilt < 2 deg, settled, no torque creep.

usage:
  uv run scripts/stand_torque.py --pose "-0.4579,-0.0049"           # HOME: one candidate, full report
  uv run scripts/stand_torque.py --sweep-roll --json runs/stand_torque_v2.json
  uv run scripts/stand_torque.py --backlash --pose "<hp>,<knee>[,<roll_deg>]"   # confirm the winner
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

JOINTS = ("left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
          "neck_pitch", "head_pitch", "head_yaw", "head_roll",
          "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle")
LEG_IDX = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
NECK_PITCH = HEAD_PITCH = 0.3491
HIP_ROLL_DEG = 5.0  # microduck_constants.HOME_FRAME: the sole sits flat at this inward roll
HOME = dict(hip_pitch=-0.4579, knee=-0.0049, hip_roll_deg=HIP_ROLL_DEG)  # STAND2, left-leg sign
FOOT_GEOMS = ("left_foot_collision", "right_foot_collision")
TIMESTEP = 0.005
ROBOT_DIR = REPO / "src" / "mjlab_microduck" / "robot" / "microduck_hd1910"
BACKLASH_XML = ROBOT_DIR / "robot_walk_hd1910_backlash.xml"


# ---- poses --------------------------------------------------------------------------------------

def pose_vector(hip_pitch, knee, hip_roll_deg=HIP_ROLL_DEG, hip_yaw=0.0, ankle=None):
    """14 joint targets in actuator order from the left-leg chain; right leg mirrored; flat sole by default."""
    ankle = knee - hip_pitch if ankle is None else ankle
    left = [hip_yaw, -math.radians(hip_roll_deg), hip_pitch, knee, ankle]
    return np.array(left + [NECK_PITCH, HEAD_PITCH, 0.0, 0.0] + [-x for x in left], dtype=np.float32)


def chain_of(pose):
    return dict(hip_yaw=float(pose[0]), hip_roll=float(pose[1]), hip_pitch=float(pose[2]), knee=float(pose[3]),
                ankle=float(pose[4]))


def init_state_dict(pose, ndigits=4):
    """HOME_FRAME-style {joint regex: value} (plain names; mirrored right leg)."""
    return {name: round(float(v), ndigits) for name, v in zip(JOINTS, pose)}


def parse_range(text):
    """'a:b:step' -> inclusive grid, or a single number."""
    parts = [float(x) for x in text.split(":")]
    if len(parts) == 1:
        return [parts[0]]
    a, b, step = parts
    n = int(round(abs(b - a) / abs(step))) + 1
    return [round(a + i * math.copysign(abs(step), b - a), 6) for i in range(n)]


# ---- simulation ---------------------------------------------------------------------------------

def build(hd1910=True, kp_scale=8.0, vin=7.4, vin_drop_gain=None, backlash=False):
    """Fresh (bam_model, model, data, ctrl, names); BAM mutates the model every step, so never reuse one."""
    kp = (ip.HD1910_KP_FW if hd1910 else ip.BAM_KP_FW) * kp_scale
    if hd1910:
        bam = ip.load_bam_model(kp, vin, ip.HD1910_MAX_CURRENT, json_path=str(REPO / ip.HD1910_PARAMS_JSON),
                                max_pwm=ip.HD1910_MAX_PWM)
        xml = REPO / ip.HD1910_XML
    else:
        bam = ip.load_bam_model(kp, vin, ip.BAM_MAX_CURRENT)
        xml = REPO / ip.MICRODUCK_XML
    spec = None
    if backlash:
        if not hd1910:
            raise ValueError("--backlash is only wired for the HD-1910 model")
        spec = hd1910_backlash_spec()
    model, data, ctrl, names = ip.load_mujoco_with_bam(str(xml), bam, TIMESTEP, vin_drop_gain, ip.BAM_VIN_MIN, spec=spec)
    return bam, model, data, ctrl, names


def hd1910_backlash_spec():
    """The HD-1910 backlash robot (a passive_<joint>_backlash hinge of +-1 deg behind every servo) as an
    MjSpec with the scene's floor added: the robot XML has no world, and no scene XML wraps it."""
    spec = mujoco.MjSpec.from_file(str(BACKLASH_XML))
    floor = spec.worldbody.add_geom()
    floor.name, floor.type, floor.size = "floor", mujoco.mjtGeom.mjGEOM_PLANE, [0.0, 0.0, 0.05]
    return spec


def sole_points(model, data, geom_name):
    g = model.geom(geom_name).id
    mid = model.geom_dataid[g]
    va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    return data.geom_xpos[g] + model.mesh_vert[va:va + vn] @ data.geom_xmat[g].reshape(3, 3).T


def sole_plane(model, data, geom_name, band=1.5e-3):
    """(tilt_deg, lowest z, contact-patch centroid) of a sole from the mesh vertices within `band` of its lowest point."""
    p = sole_points(model, data, geom_name)
    low = p[p[:, 2] < p[:, 2].min() + band]
    c = low.mean(0)
    n = np.linalg.svd(low - c)[2][2]
    n = n if n[2] > 0 else -n
    return math.degrees(math.acos(float(np.clip(n[2], -1.0, 1.0)))), float(p[:, 2].min()), c


def place(model, data, pose, z_extra=0.005, trunk_rpy_deg=(0.0, 0.0, 0.0)):
    """Set qpos: joints at `pose`, trunk rotated by rpy, lowest sole point z_extra above the floor. Returns trunk z."""
    jid = model.joint("trunk_base_freejoint").id
    adr = model.jnt_qposadr[jid]
    quat = np.zeros(4)
    mujoco.mju_euler2Quat(quat, np.radians(list(trunk_rpy_deg) + [0.0] * (3 - len(trunk_rpy_deg))), "xyz")
    data.qpos[adr:adr + 3] = [0.0, 0.0, 0.2]
    data.qpos[adr + 3:adr + 7] = quat
    data.qpos[[int(model.jnt_qposadr[model.actuator_trnid[i, 0]]) for i in range(model.nu)]] = pose
    data.qvel[:] = 0.0
    mujoco.mj_kinematics(model, data)
    zmin = min(sole_plane(model, data, g)[1] for g in FOOT_GEOMS)
    data.qpos[adr + 2] += z_extra - zmin
    mujoco.mj_kinematics(model, data)
    return float(data.qpos[adr + 2])


def contact_height(model, pose):
    """Trunk z at which the lowest sole point touches the floor with the trunk level (side-effect free)."""
    return place(model, mujoco.MjData(model), pose, z_extra=0.0)


def com_ankle_mm(model, data):
    """Whole-robot CoM minus the left ankle axis, forward direction, mm (needs kinematics)."""
    mujoco.mj_comPos(model, data)
    trunk = model.body("trunk_base").id
    return 1000.0 * float(data.subtree_com[trunk][0] - data.xanchor[model.joint("left_ankle").id][0])


def solve_hip_pitch(model, knee, com_mm, hip_roll_deg=HIP_ROLL_DEG, lo=-1.2, hi=0.6):
    """hip_pitch (left sign) such that the CoM sits com_mm ahead of the ankle axis, with the sole flat and the trunk level."""
    data = mujoco.MjData(model)

    def f(hp):
        place(model, data, pose_vector(hp, knee, hip_roll_deg), 0.0)
        return com_ankle_mm(model, data) - com_mm

    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        raise ValueError(f"no hip_pitch in [{lo}, {hi}] puts the CoM {com_mm} mm ahead of the ankle at knee {knee}")
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if flo * fm <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)


def knee_off_line_mm(model, data):
    """Sagittal-plane distance of the left knee axis from the left hip_pitch -> ankle line (needs kinematics)."""
    j = lambda n: data.xanchor[model.joint(n).id][[0, 2]]  # x, z: the pitch axes are lateral
    h, k, a = j("left_hip_pitch"), j("left_knee"), j("left_ankle")
    d, e = a - h, k - h
    return 1000.0 * float(abs(e[0] * d[1] - e[1] * d[0]) / np.linalg.norm(d))


SWEPT_IDX = [2, 3, 4, 11, 12, 13]  # hip_pitch, knee, ankle on both legs: the joints the sweep moves


def soft_limit_margin(model, pose):
    """Smallest distance of a swept joint from its soft limit (0.9 of the range about the midpoint), rad."""
    margin = math.inf
    for i in SWEPT_IDX:
        lo, hi = model.jnt_range[model.actuator_trnid[i, 0]]
        mid, half = 0.5 * (lo + hi), 0.45 * (hi - lo)
        margin = min(margin, (mid + half) - pose[i], pose[i] - (mid - half))
    return float(margin)


def analytic_moments(model, data):
    """Gravity moment each joint must hold with the ground reaction (M g / 2 per foot) at each sole's contact centroid.

    Static, contact-model-free: tau_j = a_j . [ (c_foot - p_j) x F_foot - sum_{b distal of j} (x_b - p_j) x m_b g z ].
    Compared with the stiff hold it shows whether the simulated CoP sits at the sole centre (it does not for the
    hip_roll: the CoP can move to the sole edge and remove that moment).
    """
    mujoco.mj_comPos(model, data)
    g = -model.opt.gravity[2]
    total = float(model.body_subtreemass[model.body("trunk_base").id])
    cop = {side: sole_plane(model, data, f"{side}_foot_collision")[2] for side in ("left", "right")}
    F = np.array([0.0, 0.0, 0.5 * total * g])
    children = {b: [c for c in range(model.nbody) if model.body_parentid[c] == b] for b in range(model.nbody)}

    def descendants(b):
        out, stack = [], [b]
        while stack:
            x = stack.pop()
            out.append(x)
            stack.extend(children[x])
        return out

    out = np.zeros(model.nu)
    for i in range(model.nu):
        jid = model.actuator_trnid[i, 0]
        axis, p = data.xaxis[jid], data.xanchor[jid]
        tau = 0.0
        for b in descendants(model.jnt_bodyid[jid]):
            tau += axis @ np.cross(data.xipos[b] - p, np.array([0.0, 0.0, -model.body_mass[b] * g]))
        name = JOINTS[i]
        if name.startswith(("left_", "right_")):
            tau += axis @ np.cross(cop[name.split("_")[0]] - p, F)
        out[i] = -tau
    return out


def hold(pose, *, hd1910=True, backlash=False, kp_scale=8.0, seconds=3.0, trunk_rpy_deg=(0.0, 0.0),
         joint_noise=0.0, seed=0, z_extra=0.005, policy_onnx=None, vin=7.4):
    """Hold `pose` from a (possibly perturbed) start; report stability and the torque it costs."""
    bam, model, data, ctrl, names = build(hd1910, kp_scale, vin, None, backlash)
    rng = np.random.default_rng(seed)
    q0 = np.asarray(pose, dtype=np.float64) + (rng.uniform(-joint_noise, joint_noise, len(pose)) if joint_noise else 0.0)
    z0 = place(model, data, q0, z_extra, trunk_rpy_deg)
    ctrl.reset(data.qpos)
    ctrl.last_ts = data.time
    ctrl.q_target[:] = pose
    policy = None
    if policy_onnx:
        policy = ip.PolicyInference(model, data, walking_onnx_path=policy_onnx, bam_ctrl=ctrl, new_cmd_obs=True,
                                    use_projected_gravity=True, default_pose=pose, joint_vel_lag=1)
        policy.set_vel_cmd(0.0, 0.0, 0.0)
    mujoco.mj_forward(model, data)
    trunk = model.body("trunk_base").id
    qidx, didx = ctrl.qpos_indexes, ctrl.dof_indexes
    n = int(round(seconds / model.opt.timestep))
    taus, tilts, zs, qds = [], [], [], []
    fell_t = None
    for k in range(n):
        if policy is not None and k % 4 == 0:
            policy.set_position_targets(policy.default_pose + policy.infer() * policy.action_scale)
        ctrl.update()
        mujoco.mj_step(model, data)
        R = data.xmat[trunk].reshape(3, 3)
        tilt = math.degrees(math.acos(float(np.clip(R[2, 2], -1.0, 1.0))))
        taus.append(np.abs(data.ctrl).copy())
        tilts.append(tilt)
        zs.append(float(data.xpos[trunk][2]))
        qds.append(np.abs(data.qvel[didx]).max())
        if fell_t is None and (tilt > 45.0 or data.xpos[trunk][2] < 0.06):
            fell_t = data.time
    taus, tilts, zs, qds = np.array(taus), np.array(tilts), np.array(zs), np.array(qds)
    last, half = taus[-200:], taus[-100:]
    prev = taus[-200:-100]
    tau_mean, tau_max = last.mean(0), last.max(0)
    sole = {s: sole_plane(model, data, f"{s}_foot_collision")[0] for s in ("left", "right")}
    pitch = math.degrees(math.atan2(-R[2, 0], R[2, 2]))
    return dict(
        fell=fell_t is not None, time_to_fall=fell_t, tilt_end=float(tilts[-1]), tilt_max=float(tilts.max()),
        pitch_end=pitch, z_start=z0, z_end=float(zs[-1]),
        settled=bool(qds[-100:].max() < 0.05 and abs(zs[-1] - zs[-100]) < 0.5e-3),
        tau_drift=float(np.abs(half.mean(0) - prev.mean(0)).max()),
        sole_tilt_left=sole["left"], sole_tilt_right=sole["right"],
        tau_mean=tau_mean.round(4).tolist(), tau_max=tau_max.round(4).tolist(),
        sumsq=float((last ** 2).mean(0).sum()), sumsq_legs=float((last[:, LEG_IDX] ** 2).mean(0).sum()),
        maxabs=float(tau_mean.max()), maxabs_joint=JOINTS[int(tau_mean.argmax())],
        q_err=np.abs(data.qpos[qidx] - np.asarray(pose)).round(4).tolist(),
        com_ankle_mm=com_ankle_mm(model, data), knee_off_line_mm=knee_off_line_mm(model, data),
        analytic=analytic_moments(model, data).round(4).tolist(),
    )


def perturbed_starts(n=8, joint_noise=0.05, tilt_deg=3.0, seed=0):
    combos = [(tilt_deg, 0), (-tilt_deg, 0), (0, tilt_deg), (0, -tilt_deg),
              (tilt_deg, tilt_deg), (-tilt_deg, -tilt_deg), (tilt_deg, -tilt_deg), (-tilt_deg, tilt_deg)]
    return [dict(trunk_rpy_deg=combos[i % len(combos)], joint_noise=joint_noise, seed=seed + i) for i in range(n)]


def evaluate(pose, starts, **kw):
    """Nominal hold (the metrics) plus every perturbed start (the pass/fail)."""
    nominal = hold(pose, **kw)
    runs = [hold(pose, **{**kw, **s}) for s in starts]
    ok = lambda r: (not r["fell"]) and r["tilt_end"] < 2.0 and r["settled"]
    failing = [i for i, r in enumerate(runs) if not ok(r)]
    return dict(nominal=nominal, starts=runs, passed=ok(nominal) and not failing, failing=failing)


# ---- sweep and report ---------------------------------------------------------------------------

def candidate(model, knee, com_mm, roll_deg):
    hp = solve_hip_pitch(model, knee, com_mm, roll_deg)
    pose = pose_vector(hp, knee, roll_deg)
    data = mujoco.MjData(model)
    place(model, data, pose, 0.0)
    return dict(knee=knee, com_mm=com_mm, roll_deg=roll_deg, hip_pitch=hp, ankle=float(pose[4]), pose=pose,
                z_contact=contact_height(model, pose), knee_off_line_mm=knee_off_line_mm(model, data),
                margin=soft_limit_margin(model, pose))


def reject_reason(c, home_sumsq=None):
    if c["knee_off_line_mm"] < 3.0:
        return "knee within 3 mm of the hip-ankle line (singular)"
    if c["margin"] < 0.5:
        return f"hip_pitch/knee/ankle within {c['margin']:.2f} rad of a soft limit"
    return None


def print_table(rows):
    cols = "knee  hip_p  ankle roll  com  kneeoff  z_ct  | hip_roll hip_p  knee  ankle | legsum2  maxabs(joint)    tilt sole | result"
    print(cols)
    for r in rows:
        n = r.get("nominal")
        if n is None:
            print(f"{r['knee']:+.2f} {r['hip_pitch']:+.3f} {r['ankle']:+.3f} {r['roll_deg']:4.1f} {r['com_mm']:+4.1f} "
                  f"{r['knee_off_line_mm']:6.1f} {r['z_contact']*1000:5.1f} | rejected: {r['reject']}")
            continue
        tm = n["tau_mean"]
        print(f"{r['knee']:+.2f} {r['hip_pitch']:+.3f} {r['ankle']:+.3f} {r['roll_deg']:4.1f} {r['com_mm']:+4.1f} "
              f"{r['knee_off_line_mm']:6.1f} {r['z_contact']*1000:5.1f} | {tm[1]:.3f}    {tm[2]:.3f} {tm[3]:.3f} {tm[4]:.3f} | "
              f"{n['sumsq_legs']:.4f}  {n['maxabs']:.3f} {n['maxabs_joint']:<15s} {n['tilt_end']:4.2f} "
              f"{max(n['sole_tilt_left'], n['sole_tilt_right']):4.2f} | "
              f"{'pass' if r['passed'] else 'FAIL ' + str(r['failing'])}")


def winner_block(c, hold_result):
    pose = c["pose"]
    z = c["z_contact"]
    chain = chain_of(pose)
    lines = ["", "== winner",
             f"chain (left sign): hip_pitch {chain['hip_pitch']:+.4f}  knee {chain['knee']:+.4f}  ankle {chain['ankle']:+.4f}  "
             f"hip_roll {chain['hip_roll']:+.4f} ({c['roll_deg']:.1f} deg)",
             f"HD1910_STAND = dict(hip_yaw=0.0, hip_roll={chain['hip_roll']:+.4f}, hip_pitch={chain['hip_pitch']:+.4f}, "
             f"knee={chain['knee']:+.4f}, ankle={chain['ankle']:+.4f})",
             "init_state joint_pos = " + json.dumps(init_state_dict(pose)),
             f"trunk z at contact {z:.4f} m; training spawn z range ({z + 0.003:.4f}, {z + 0.013:.4f}); CPU spawn z {z + 0.008:.4f}",
             "scene keyframe:",
             f'  <key name="STAND" qpos="0 0 {z + 0.003:.4f} 1 0 0 0 ' + " ".join(f"{v:.4f}" for v in pose) + '"',
             '       ctrl="' + " ".join(f"{v:.4f}" for v in pose) + '"/>',
             f"torque per joint (mean |tau|, N.m): " + ", ".join(f"{n} {t:.3f}" for n, t in zip(JOINTS, hold_result['tau_mean']) if t > 0.005),
             f"analytic (CoP at sole centre): " + ", ".join(f"{n} {abs(t):.3f}" for n, t in zip(JOINTS, hold_result['analytic']) if abs(t) > 0.005)]
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xl330", action="store_true", help="stock XL330 model instead of the HD-1910")
    ap.add_argument("--backlash", action="store_true", help="HD-1910 backlash twin (robot XML + a floor)")
    ap.add_argument("--kp-scale", type=float, default=8.0, help="PD gain multiplier for the stiff hold (>= 4 reads the gravity moment)")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--starts", type=int, default=8, help="perturbed starts per candidate (0 = nominal only)")
    ap.add_argument("--knee", default="0.0:-0.8:0.2", help="left-sign knee grid a:b:step (negative = extension)")
    ap.add_argument("--com-mm", default="-2:6:2", help="CoM ahead of the ankle axis, mm, grid a:b:step")
    ap.add_argument("--hip-roll-deg", type=float, default=HIP_ROLL_DEG)
    ap.add_argument("--sweep-roll", action="store_true", help="also sweep hip_roll over 4.0, 4.5, 5.0, 5.5 deg")
    ap.add_argument("--pose", help='one explicit candidate "hip_pitch,knee[,roll_deg]" (left sign) instead of the grid')
    ap.add_argument("--policy", help="also hold with this walking ONNX at kp_scale 1 (zero command) from the same starts")
    ap.add_argument("--passive", action="store_true", help="also report time-to-fall of the PD-only hold at kp_scale 1")
    ap.add_argument("--json", help="write every candidate's metrics here")
    args = ap.parse_args()
    hd1910 = not args.xl330
    kw = dict(hd1910=hd1910, backlash=args.backlash, kp_scale=args.kp_scale, seconds=args.seconds)
    starts = perturbed_starts(args.starts) if args.starts > 0 else []

    _, model, _, _, _ = build(hd1910, args.kp_scale, backlash=args.backlash)
    if args.pose:
        parts = [float(x) for x in args.pose.split(",")]
        roll = parts[2] if len(parts) > 2 else args.hip_roll_deg
        pose = pose_vector(parts[0], parts[1], roll)
        data = mujoco.MjData(model)
        place(model, data, pose, 0.0)
        cands = [dict(knee=parts[1], com_mm=com_ankle_mm(model, data), roll_deg=roll, hip_pitch=parts[0], ankle=float(pose[4]),
                      pose=pose, z_contact=contact_height(model, pose), knee_off_line_mm=knee_off_line_mm(model, data),
                      margin=soft_limit_margin(model, pose))]
    else:
        rolls = [4.0, 4.5, 5.0, 5.5] if args.sweep_roll else [args.hip_roll_deg]
        cands = [candidate(model, k, c, r) for r in rolls for k in parse_range(args.knee) for c in parse_range(args.com_mm)]
    home_pose = pose_vector(HOME["hip_pitch"], HOME["knee"], HOME["hip_roll_deg"])
    data = mujoco.MjData(model)
    place(model, data, home_pose, 0.0)
    home = dict(knee=HOME["knee"], com_mm=com_ankle_mm(model, data), roll_deg=HOME["hip_roll_deg"], hip_pitch=HOME["hip_pitch"],
                ankle=float(home_pose[4]), pose=home_pose, z_contact=contact_height(model, home_pose),
                knee_off_line_mm=knee_off_line_mm(model, data), margin=soft_limit_margin(model, home_pose))

    rows = []
    for c in [home] + cands:
        c["reject"] = reject_reason(c) if c is not home else None
        if c["reject"]:
            rows.append(c)
            continue
        ev = evaluate(c["pose"], starts, **kw)
        c.update(nominal=ev["nominal"], passed=ev["passed"], failing=ev["failing"], starts=ev["starts"])
        if args.passive:
            c["passive_time_to_fall"] = hold(c["pose"], **{**kw, "kp_scale": 1.0})["time_to_fall"]
        if args.policy:
            pol = [hold(c["pose"], **{**kw, "kp_scale": 1.0, "policy_onnx": args.policy, **s}) for s in [dict()] + starts]
            c["policy_holds"] = dict(fell=[r["fell"] for r in pol], tilt_end=[r["tilt_end"] for r in pol], z_end=[r["z_end"] for r in pol])
        rows.append(c)
        print(f"  {'HOME' if c is home else 'cand'} knee {c['knee']:+.2f} com {c['com_mm']:+.1f} roll {c['roll_deg']:.1f}: "
              f"legs sum tau^2 {ev['nominal']['sumsq_legs']:.4f}  max {ev['nominal']['maxabs']:.3f} ({ev['nominal']['maxabs_joint']})  "
              f"{'pass' if ev['passed'] else 'FAIL'}", flush=True)

    print(f"\n== candidates ({'HD-1910' if hd1910 else 'XL330'}{' backlash' if args.backlash else ''}, kp x{args.kp_scale:g}, "
          f"{len(starts)} perturbed starts); first row is HOME")
    print_table(rows)
    passing = [r for r in rows[1:] if r.get("passed")]
    home_n = rows[0]["nominal"]
    best = None
    if passing:
        best = min(passing, key=lambda r: (r["nominal"]["sumsq_legs"], r["nominal"]["maxabs"]))
        red = 1.0 - best["nominal"]["sumsq_legs"] / home_n["sumsq_legs"]
        print(f"\nHOME legs sum tau^2 {home_n['sumsq_legs']:.4f} (max {home_n['maxabs']:.3f}); best candidate "
              f"{best['nominal']['sumsq_legs']:.4f} (max {best['nominal']['maxabs']:.3f}): {red:+.0%}")
        adopt = red >= 0.30 and best["nominal"]["maxabs"] <= home_n["maxabs"] and max(best["nominal"]["sole_tilt_left"], best["nominal"]["sole_tilt_right"]) <= 1.0
        print("adopt: " + ("yes" if adopt else "NO (needs >= 30 % less heat, no larger worst joint, sole within 1 deg)"))
        winner_block(best, best["nominal"])
    else:
        print("\nno candidate passed the perturbed-start holds")
    if args.json:
        doc = dict(args=vars(args), joints=JOINTS, home=_jsonable(rows[0]), candidates=[_jsonable(r) for r in rows[1:]],
                   winner=_jsonable(best) if best else None)
        Path(args.json).write_text(json.dumps(doc, indent=1))
        print("wrote", args.json)


def _jsonable(r):
    out = {}
    for k, v in r.items():
        if isinstance(v, np.ndarray):
            out[k] = v.round(6).tolist()
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = v.item()
        else:
            out[k] = v
    return out


if __name__ == "__main__":
    main()
