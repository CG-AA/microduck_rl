"""Record HD-1910 pendulum trajectories as BAM logs (the input of scripts/bench_fit.py).

Rig: the servo clamped with its shaft horizontal, an arm of known length and mass on the horn, a tip
mass; the arm must hang and swing freely through at least +-100 deg. Each run: find the hanging zero
(release the arm from above and below, take the midpoint; stop if friction holds it far from
hanging, i.e. the tip mass is too light), then for
every P in --kp and every trajectory: ramp to the trajectory's start, settle, run it at --rate Hz
(one goal write + one 15-byte feedback read per tick), return to zero, rest. Trajectories are BAM's
(bam.trajectory: lift_and_drop, sin_sin, up_and_down, sin_time_square; 6 s each, torque cut in
lift_and_drop's drop).

One invocation = one tip mass at one supply voltage; swap the mass / set the supply and run again.
Logs: <logdir>/<trajectory>_p<P>_m<grams>g_<V>V_<time>.json, BAM's raw format with arm_mass, the
measured supply voltage, current and duty per sample, and the P register as "kp".

usage:
  uv run scripts/bench_record.py --port /dev/ttyUSB0 --mass 0.05 --length 0.08 --arm-mass 0.012 --kp 4 8 16 --logdir runs/bench/raw
  uv run scripts/bench_record.py --fake --mass 0.05 --length 0.08 --kp 4 8 16 --logdir runs/bench_sim/raw   # planted servo
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from mjlab_microduck.bench import hls
from mjlab_microduck.bench.cli import (SafetyStop, add_bus_args, add_pendulum_args, check, find_zero, move_to,
                                       open_from_args, safe_off, zero_problem)

TRAJECTORIES = ("lift_and_drop", "sin_sin", "up_and_down", "sin_time_square")


def record_one(bus, sid, clock, zero, trajectory, p, kd=0, rate=250.0, temp_limit=55.0) -> list:
    """One trajectory at P; returns BAM raw entries (position relative to the hanging zero)."""
    from bam.trajectory import trajectories
    traj = trajectories[trajectory]
    bus.set_gains(sid, p, kd, 0)
    bus.goal(sid, bus.feedback(sid).pos_rad)
    bus.torque(sid, hls.TORQUE_ON)
    a0, _ = traj(0.0)
    move_to(bus, sid, clock, zero + a0, temp_limit=temp_limit)
    clock.sleep(1.0)
    entries, enabled = [], True
    t_start = clock.monotonic()
    k = 0
    while True:
        t = clock.monotonic() - t_start
        if t > traj.duration:
            break
        angle, te = traj(t)
        if te != enabled:
            bus.torque(sid, hls.TORQUE_ON if te else hls.TORQUE_OFF)
            enabled = te
        if te:
            bus.goal(sid, zero + angle)
        fb = bus.feedback(sid)
        if k % 25 == 0:
            check(fb, temp_limit)
        entries.append(dict(position=fb.pos_rad - zero, speed=fb.vel_rad_s, load=fb.load_duty, current=fb.current_a,
                            input_volts=fb.volts, temp=fb.temp_c, timestamp=fb.t - t_start,
                            goal_position=float(angle), torque_enable=bool(te)))
        k += 1
        clock.sleep(t_start + k / rate - clock.monotonic())
    bus.goal(sid, bus.feedback(sid).pos_rad)
    bus.torque(sid, hls.TORQUE_ON)
    move_to(bus, sid, clock, zero, temp_limit=temp_limit)
    safe_off(bus, sid)
    return entries


def session(bus, sid, clock, logdir, mass, length, arm_mass, kps, trajectories=TRAJECTORIES, kd=0, rate=250.0,
            rest=2.0, temp_limit=55.0, log=print) -> list:
    logdir = Path(logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    zero, info = find_zero(bus, sid, clock, temp_limit=temp_limit)
    log(f"zero {zero:+.4f} rad (released from above / below: {info['above']:+.3f} / {info['below']:+.3f}, "
        f"friction band {info['band']:.3f} rad)")
    if not info["ok"]:
        raise SafetyStop(zero_problem(info))
    written = []
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for p in kps:
        for name in trajectories:
            entries = record_one(bus, sid, clock, zero, name, p, kd, rate, temp_limit)
            vin = float(np.mean([e["input_volts"] for e in entries]))
            doc = dict(mass=mass, length=length, arm_mass=arm_mass, kp=p, kd=kd, vin=vin, motor="hd1910",
                       trajectory=name, zero_rad=zero, rate=rate, entries=entries)
            path = logdir / f"{name}_p{p}_m{round(mass * 1000)}g_{vin:.1f}V_{stamp}.json"
            path.write_text(json.dumps(doc))
            dts = np.diff([e["timestamp"] for e in entries])
            log(f"  {path.name}: {len(entries)} samples, {1 / np.mean(dts):.0f} Hz (worst gap {dts.max() * 1000:.1f} ms), "
                f"{vin:.2f} V, max |I| {max(abs(e['current']) for e in entries):.2f} A, {entries[-1]['temp']} C")
            written.append(str(path))
            clock.sleep(rest)
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_bus_args(ap, fake_help="record from the simulated servo (planted model; see bench/fake.py)")
    add_pendulum_args(ap)
    ap.add_argument("--kp", type=int, nargs="+", default=[4, 8, 16], help="P register values to record")
    ap.add_argument("--kd", type=int, default=0, help="D register (0 = the sim's law; record 40 once to check D)")
    ap.add_argument("--trajectories", nargs="+", default=list(TRAJECTORIES), choices=TRAJECTORIES)
    ap.add_argument("--rate", type=float, default=250.0)
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--truth", help="--fake only: JSON of planted parameters (default bench.fake.DEFAULT_TRUTH)")
    args = ap.parse_args()

    fake_kw = {}
    if args.fake and args.truth:
        fake_kw["truth"] = json.loads(Path(args.truth).read_text())
    bus, clock, ctrl = open_from_args(args, **fake_kw)
    try:
        written = session(bus, args.id, clock, args.logdir, args.mass, args.length, args.arm_mass, args.kp,
                          args.trajectories, args.kd, args.rate, temp_limit=args.temp_limit)
    except SafetyStop as e:
        safe_off(bus, args.id)
        raise SystemExit(f"STOPPED: {e}")
    if args.fake:  # what the fit should recover
        truth = dict(ctrl.truth, error_gain=ctrl.model.actuator.error_gain)
        (Path(args.logdir) / "truth.json.planted").write_text(json.dumps(truth, indent=1))
    print(f"{len(written)} logs in {args.logdir}")


if __name__ == "__main__":
    main()
