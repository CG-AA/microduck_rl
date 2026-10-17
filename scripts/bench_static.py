"""Static HD-1910 measurements: the P-register gain map (blocked rotor) and a torque/current hold point.

gainmap  The output horn is clamped so it cannot turn. For each P in --p and each error in --err-deg
         (both signs) the goal is set off the clamped position for --hold seconds and the reported duty
         (present load, 0.1 %) is averaged. duty = error_gain * P * error fits through the origin on the
         unsaturated points: `error_gain` is the duty per radian per P unit, the number that converts
         the simulator's stiffness (kp_fw 156 = 0.449 duty/rad) into a P register value. Also reports the
         saturation duty (max_torque 980 should give 0.98) and whether duty is linear in P.
hold     A pendulum arm of known mass held at --angles from hanging. Each angle is approached from above
         and from below and the two are averaged to cancel friction. Static: I = duty*V/R and
         tau = kt*I, so the hold gives kt = tau_gravity / I and R = duty*V / I.

Short pulses, a rest between them, a temperature guard and a stop on any status flag. Results go to
--out (JSON) and are printed.

usage:
  uv run scripts/bench_static.py gainmap --port /dev/ttyUSB0 --out runs/bench/gainmap.json
  uv run scripts/bench_static.py hold --port /dev/ttyUSB0 --mass 0.1 --length 0.08 --arm-mass 0.012 --p 16
  uv run scripts/bench_static.py gainmap --fake            # rehearsal on the simulated servo
"""
import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

from mjlab_microduck.bench import hls
from mjlab_microduck.bench.bam_hd1910 import SIM_KP_FW, XL330_ERROR_GAIN
from mjlab_microduck.bench.cli import (SafetyStop, add_bus_args, add_pendulum_args, average, gravity_torque,
                                       measure_zero, move_to, open_from_args, safe_off)


def gainmap(bus, sid, clock, p_values, err_deg, hold=0.3, rest=0.5, temp_limit=55.0, log=print) -> dict:
    fb = bus.feedback(sid)
    home = fb.pos_rad
    bus.set_gains(sid, min(p_values), 0, 0)
    bus.goal(sid, home)
    bus.torque(sid, hls.TORQUE_ON)
    points = []
    try:
        for p in p_values:
            bus.set_gains(sid, p, 0, 0)
            for e in err_deg:
                for sign in (1, -1):
                    goal = home + sign * math.radians(e)
                    bus.goal(sid, goal)
                    clock.sleep(hold * 0.5)
                    m = average(bus, sid, clock, hold * 0.5, n=5, temp_limit=temp_limit)
                    bus.goal(sid, home)
                    clock.sleep(rest)
                    err = goal - m["pos_rad"]
                    points.append(dict(p=p, err_cmd=sign * math.radians(e), err=err, duty=m["load_duty"],
                                       current=m["current_a"], volts=m["volts"], temp=m["temp_c"]))
                    log(f"  P {p:3d}  err {math.degrees(err):+6.2f} deg  duty {m['load_duty']:+.3f}  "
                        f"I {m['current_a']:+.3f} A  {m['temp_c']} C")
    finally:
        safe_off(bus, sid)
    return analyse_gainmap(points)


def analyse_gainmap(points) -> dict:
    duties = np.array([pt["duty"] for pt in points])
    max_duty = float(np.max(np.abs(duties))) if len(duties) else None
    lin = [pt for pt in points if abs(pt["duty"]) < 0.9 * (max_duty or 1.0) and abs(pt["err"]) > 1e-4]
    x = np.array([pt["p"] * pt["err"] for pt in lin])
    y = np.array([pt["duty"] for pt in lin])
    eg = float(x @ y / (x @ x)) if len(lin) else None
    resid = float(np.sqrt(np.mean((y - eg * x) ** 2))) if eg is not None else None
    per_p = {}
    for p in sorted({pt["p"] for pt in lin}):
        xs = np.array([pt["err"] for pt in lin if pt["p"] == p])
        ys = np.array([pt["duty"] for pt in lin if pt["p"] == p])
        per_p[str(p)] = float(xs @ ys / (xs @ xs)) / p  # error_gain estimated at this P alone
    out = dict(points=points, error_gain=eg, rms_residual_duty=resid, max_duty=max_duty, per_p_error_gain=per_p,
               n_linear_points=len(lin))
    if eg:
        exact = SIM_KP_FW * XL330_ERROR_GAIN / eg
        out.update(sim_duty_per_rad=SIM_KP_FW * XL330_ERROR_GAIN, p_for_sim_exact=exact,
                   p_for_sim=max(1, int(round(exact))))
    return out


def hold(bus, sid, clock, mass, arm_mass, length, angles_deg, p, approach=0.15, settle=1.5, temp_limit=55.0,
         log=print) -> dict:
    zero = measure_zero(bus, sid, clock, temp_limit=temp_limit)
    bus.set_gains(sid, p, 0, 0)
    bus.goal(sid, bus.feedback(sid).pos_rad)
    bus.torque(sid, hls.TORQUE_ON)
    rows = []
    try:
        for a in angles_deg:
            target = zero + math.radians(a)
            pair = []
            for side in (1, -1):  # come from above, then from below: friction flips sign
                move_to(bus, sid, clock, target + side * approach, temp_limit=temp_limit)
                move_to(bus, sid, clock, target, speed=0.3, temp_limit=temp_limit)
                clock.sleep(settle)
                pair.append(average(bus, sid, clock, 0.5, n=20, temp_limit=temp_limit))
            m = {k: float(np.mean([r[k] for r in pair])) for k in ("pos_rad", "load_duty", "current_a", "volts")}
            q = m["pos_rad"] - zero
            tau = gravity_torque(mass, arm_mass, length, q)
            kt = tau / abs(m["current_a"]) if abs(m["current_a"]) > 1e-3 else None
            R = abs(m["load_duty"]) * m["volts"] / abs(m["current_a"]) if abs(m["current_a"]) > 1e-3 else None
            rows.append(dict(angle_deg=a, q=q, tau=tau, **m, kt=kt, R=R,
                             friction_band_duty=abs(pair[0]["load_duty"] - pair[1]["load_duty"]) / 2))
            log(f"  {a:5.1f} deg: tau {tau:.4f} N.m  I {m['current_a']:+.3f} A  duty {m['load_duty']:+.3f}  "
                f"{m['volts']:.2f} V  -> kt {kt if kt is None else round(kt, 3)}  R {R if R is None else round(R, 2)}")
        move_to(bus, sid, clock, zero, temp_limit=temp_limit)
    finally:
        safe_off(bus, sid)
    kts = [r["kt"] for r in rows if r["kt"]]
    Rs = [r["R"] for r in rows if r["R"]]
    return dict(zero=zero, p=p, rows=rows, kt=float(np.mean(kts)) if kts else None, R=float(np.mean(Rs)) if Rs else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("gainmap", "hold"))
    add_bus_args(ap)
    add_pendulum_args(ap, mass_required=False)
    ap.add_argument("--p", type=int, nargs="+", help="P register values (gainmap default 2 4 8 16 32; hold one value)")
    ap.add_argument("--err-deg", type=float, nargs="+", default=[2, 5, 10, 20, 40])
    ap.add_argument("--hold", type=float, default=0.3, help="seconds per gainmap pulse")
    ap.add_argument("--rest", type=float, default=0.5, help="seconds between pulses")
    ap.add_argument("--angles", type=float, nargs="+", default=[45, 90], help="hold angles from hanging [deg]")
    ap.add_argument("--out", help="JSON output (default runs/bench/<mode>_<time>.json)")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.out or f"runs/bench/{args.mode}_{stamp}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        if args.mode == "gainmap":
            bus, clock, _ = open_from_args(args, blocked=True)
            res = gainmap(bus, args.id, clock, args.p or [2, 4, 8, 16, 32], args.err_deg, args.hold, args.rest,
                          args.temp_limit)
            print(f"\nerror_gain {res['error_gain']:.5f} duty/rad per P unit (rms residual {res['rms_residual_duty']:.4f}), "
                  f"max duty {res['max_duty']:.3f}")
            print("per P:", {k: round(v, 5) for k, v in res["per_p_error_gain"].items()}, "(equal = duty linear in P)")
            print(f"sim stiffness {res['sim_duty_per_rad']:.3f} duty/rad -> P {res['p_for_sim_exact']:.2f}, "
                  f"nearest integer P {res['p_for_sim']}")
        else:
            if args.mass is None or args.length is None:
                ap.error("hold needs --mass and --length")
            bus, clock, _ = open_from_args(args)
            res = hold(bus, args.id, clock, args.mass, args.arm_mass, args.length, args.angles, (args.p or [16])[0],
                       temp_limit=args.temp_limit)
            print(f"\nkt {res['kt']} N.m/A, R {res['R']} ohm (static, friction-averaged)")
    except SafetyStop as e:
        raise SystemExit(f"STOPPED: {e}")
    res["args"] = {k: v for k, v in vars(args).items()}
    out.write_text(json.dumps(res, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
