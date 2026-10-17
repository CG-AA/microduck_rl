"""Thermal run: hold a known torque on the HD-1910 and log temperature until it levels off or hits a limit.

The datasheet's 0.29 N.m "rated torque" is a retailer number; this measures what the servo sustains.
The arm (known mass and length) is held at --angle from hanging; temperature, current, duty and voltage
are logged every second until --minutes pass or the temperature reaches --limit. A first-order fit
T(t) = T_inf - (T_inf - T0) exp(-t / tau) gives the settling temperature at this torque; with heating
proportional to I^2, the torque that would settle at --limit is tau_hold * sqrt((limit - T_amb) /
(T_inf - T_amb)): the measured continuous rating at that ambient. Run it at more than one torque.

usage:
  uv run scripts/bench_thermal.py --port /dev/ttyUSB0 --mass 0.2 --length 0.08 --arm-mass 0.012 --angle 90 --p 16
  uv run scripts/bench_thermal.py --fake --mass 0.2 --length 0.08 --minutes 3
"""
import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

from mjlab_microduck.bench import hls
from mjlab_microduck.bench.cli import (SafetyStop, add_bus_args, add_pendulum_args, average, gravity_torque,
                                       measure_zero, move_to, open_from_args, safe_off)


def fit_first_order(t, temp):
    """(T_inf, tau, T0) of T(t) = T_inf - (T_inf - T0) exp(-t / tau); None if it cannot be fitted."""
    from scipy.optimize import curve_fit
    t, temp = np.asarray(t, float), np.asarray(temp, float)
    if len(t) < 10 or temp[-1] - temp[0] < 1.0:
        return None
    f = lambda tt, tinf, tau, t0: tinf - (tinf - t0) * np.exp(-tt / tau)
    try:
        (tinf, tau, t0), _ = curve_fit(f, t, temp, p0=(temp[-1] + 5, max(t[-1] / 2, 30.0), temp[0]),
                                       bounds=([temp[0], 1.0, temp[0] - 5], [200.0, 1e5, temp[0] + 5]))
    except RuntimeError:
        return None
    return float(tinf), float(tau), float(t0)


def run(bus, sid, clock, mass, arm_mass, length, angle_deg, p, minutes, limit, ambient=None, log=print) -> dict:
    zero = measure_zero(bus, sid, clock, temp_limit=limit + 5)
    bus.set_gains(sid, p, 0, 0)
    bus.goal(sid, bus.feedback(sid).pos_rad)
    bus.torque(sid, hls.TORQUE_ON)
    rows = []
    try:
        move_to(bus, sid, clock, zero + math.radians(angle_deg), temp_limit=limit + 5)
        t0 = clock.monotonic()
        while clock.monotonic() - t0 < minutes * 60:
            m = average(bus, sid, clock, 1.0, n=5, temp_limit=limit + 5)
            q = m["pos_rad"] - zero
            rows.append(dict(t=clock.monotonic() - t0, temp=m["temp_c"], current=m["current_a"], duty=m["load_duty"],
                             volts=m["volts"], q=q, tau=gravity_torque(mass, arm_mass, length, q)))
            if len(rows) % 30 == 0:
                log(f"  {rows[-1]['t'] / 60:5.1f} min  {m['temp_c']} C  I {m['current_a']:+.3f} A  {m['volts']:.2f} V")
            if m["temp_c"] >= limit:
                log(f"  reached {limit} C")
                break
        move_to(bus, sid, clock, zero, temp_limit=limit + 10)
    finally:
        safe_off(bus, sid)
    t = [r["t"] for r in rows]
    temp = [r["temp"] for r in rows]
    tau_hold = float(np.mean([r["tau"] for r in rows]))
    amb = float(ambient if ambient is not None else temp[0])
    res = dict(rows=rows, tau_hold=tau_hold, current=float(np.mean([r["current"] for r in rows])), ambient=amb,
               limit=limit, minutes=t[-1] / 60, reached_limit=temp[-1] >= limit)
    fitted = fit_first_order(t, temp)
    if fitted:
        tinf, tc, _ = fitted
        res.update(t_inf=tinf, time_constant_s=tc)
        if tinf > amb + 0.5:
            res["continuous_torque_at_limit"] = tau_hold * math.sqrt((limit - amb) / (tinf - amb))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_bus_args(ap)
    add_pendulum_args(ap)
    ap.add_argument("--angle", type=float, default=90.0, help="hold angle from hanging [deg]")
    ap.add_argument("--p", type=int, default=16)
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--limit", type=float, default=60.0, help="stop at this temperature [C]")
    ap.add_argument("--ambient", type=float, help="ambient temperature [C] (default: the servo's first reading)")
    ap.add_argument("--out", help="JSON (default runs/bench/thermal_<time>.json)")
    args = ap.parse_args()

    bus, clock, _ = open_from_args(args)
    try:
        res = run(bus, args.id, clock, args.mass, args.arm_mass, args.length, args.angle, args.p, args.minutes,
                  args.limit, args.ambient)
    except SafetyStop as e:
        raise SystemExit(f"STOPPED: {e}")
    print(f"held {res['tau_hold']:.3f} N.m at {res['current']:.3f} A for {res['minutes']:.1f} min")
    if "t_inf" in res:
        print(f"settles at {res['t_inf']:.1f} C (time constant {res['time_constant_s']:.0f} s)"
              + (f"; continuous torque at {args.limit:g} C: {res['continuous_torque_at_limit']:.3f} N.m"
                 if "continuous_torque_at_limit" in res else ""))
    out = Path(args.out or f"runs/bench/thermal_{datetime.now():%Y%m%d_%H%M%S}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    res["args"] = vars(args)
    out.write_text(json.dumps(res, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
