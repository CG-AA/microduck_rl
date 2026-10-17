"""Shared pieces of the bench scripts: opening a real or fake bus, safety checks, slow moves, the
pendulum zero."""
from __future__ import annotations

import math

import numpy as np

from mjlab_microduck.bench import hls

G = 9.80665


class SafetyStop(RuntimeError):
    pass


def add_bus_args(ap, fake_help="run against the simulated servo (no hardware)"):
    ap.add_argument("--port", help="serial port of the half-duplex TTL adapter, e.g. /dev/ttyUSB0")
    ap.add_argument("--fake", action="store_true", help=fake_help)
    ap.add_argument("--id", type=int, default=1, help="servo id (default 1)")
    ap.add_argument("--baud", type=int, default=hls.BAUD_1M)
    ap.add_argument("--temp-limit", type=float, default=55.0, help="stop above this servo temperature [C]")


def add_pendulum_args(ap, mass_required=True):
    ap.add_argument("--mass", type=float, required=mass_required, help="tip mass [kg]")
    ap.add_argument("--length", type=float, required=mass_required, help="pivot to tip-mass centre [m]")
    ap.add_argument("--arm-mass", type=float, default=0.0, help="arm mass without the tip mass [kg]")


def open_from_args(args, **fake_kw):
    """(bus, clock, ctrl). The fake gets the script's pendulum (mass/length/arm_mass) if it has one."""
    if args.fake:
        from mjlab_microduck.bench.fake import FakeClock, FakeHlsController
        clock = FakeClock()
        for k in ("mass", "length", "arm_mass"):
            if getattr(args, k, None) is not None and k not in fake_kw:
                fake_kw[k] = getattr(args, k)
        ctrl = FakeHlsController(clock, ids=[args.id], **fake_kw)
        return hls.HlsBus(ctrl, [args.id], clock=clock), clock, ctrl
    if not args.port:
        raise SystemExit("--port is required (or --fake)")
    bus = hls.open_bus(args.port, [args.id], args.baud)
    return bus, bus.clock, bus.ctrl


def check(fb: hls.Feedback, temp_limit: float):
    faults = fb.faults()
    if faults:
        raise SafetyStop(f"servo status flags {faults} (status {fb.status:#04x})")
    if fb.temp_c >= temp_limit:
        raise SafetyStop(f"servo at {fb.temp_c} C (limit {temp_limit} C): let it cool")


def safe_off(bus, sid):
    try:
        bus.torque(sid, hls.TORQUE_OFF)
    except Exception as e:  # never mask the original error
        print(f"warning: could not turn torque off: {e}")


def average(bus, sid, clock, seconds, n=10, temp_limit=55.0):
    """Mean feedback over `seconds` (n reads)."""
    rows = []
    for _ in range(n):
        clock.sleep(seconds / n)
        fb = bus.feedback(sid)
        check(fb, temp_limit)
        rows.append(fb)
    keys = ("pos_rad", "vel_rad_s", "load_duty", "volts", "current_a")
    out = {k: float(np.mean([getattr(r, k) for r in rows])) for k in keys}
    out["temp_c"] = max(r.temp_c for r in rows)
    return out


def move_to(bus, sid, clock, target_rad, speed=1.0, rate=100.0, temp_limit=55.0):
    """Ramp the goal from the present position to target at `speed` rad/s (torque must be on)."""
    start = bus.feedback(sid).pos_rad
    n = max(1, int(abs(target_rad - start) / speed * rate))
    for k in range(1, n + 1):
        bus.goal(sid, start + (target_rad - start) * k / n)
        clock.sleep(1.0 / rate)
        if k % 10 == 0:
            check(bus.feedback(sid), temp_limit)


def _release_from(bus, sid, clock, target, p_gain, temp_limit, settle=1.5):
    """Hold the arm at target, cut torque, let it settle; the resting encoder angle."""
    bus.set_gains(sid, p_gain, 0, 0)
    bus.goal(sid, bus.feedback(sid).pos_rad)
    bus.torque(sid, hls.TORQUE_ON)
    move_to(bus, sid, clock, target, temp_limit=temp_limit)
    clock.sleep(0.5)
    bus.torque(sid, hls.TORQUE_OFF)
    clock.sleep(settle)
    return average(bus, sid, clock, 0.3, n=6, temp_limit=temp_limit)["pos_rad"]


def find_zero(bus, sid, clock, lift=0.4, p_gain=8, temp_limit=55.0) -> tuple[float, dict]:
    """The hanging angle (the pendulum's q = 0) and a friction check.

    Friction stops a released arm anywhere inside a band around the true hanging angle, so one torque-off
    reading can be off by the whole band. The arm is released from `lift` above and below the rough rest
    position; the midpoint of the two rests is the zero (exact for symmetric friction), and half their
    gap is the friction band. A band over 0.15 rad means the tip mass is too light for the servo's
    friction (the logs would mostly measure friction): use a heavier mass or a longer arm.
    """
    bus.torque(sid, hls.TORQUE_OFF)
    clock.sleep(1.0)
    rough = average(bus, sid, clock, 0.3, n=6, temp_limit=temp_limit)["pos_rad"]
    above = _release_from(bus, sid, clock, rough + lift, p_gain, temp_limit)
    below = _release_from(bus, sid, clock, rough - lift, p_gain, temp_limit)
    zero, band = 0.5 * (above + below), 0.5 * abs(above - below)
    moved = min(abs(rough + lift - above), abs(rough - lift - below))
    info = dict(rough=rough, above=above, below=below, band=band, moved=moved,
                ok=band < 0.15 and moved > 0.5 * lift)
    return zero, info


def measure_zero(bus, sid, clock, temp_limit=55.0) -> float:
    zero, info = find_zero(bus, sid, clock, temp_limit=temp_limit)
    if not info["ok"]:
        raise SafetyStop(zero_problem(info))
    return zero


def zero_problem(info) -> str:
    return (f"the released arm stops up to {info['band']:.2f} rad ({info['band'] * 57.3:.0f} deg) from hanging and moved "
            f"only {info['moved']:.2f} rad: friction beats gravity (tip mass too light for this servo: use a heavier "
            "mass or a longer arm, gravity torque with the arm horizontal at least ~8x the friction), or the arm "
            "is jammed / touching something")


def gravity_torque(mass, arm_mass, length, q):
    """Pendulum gravity torque magnitude at angle q from hanging [N.m] (bam.testbench.Pendulum)."""
    return (mass + arm_mass / 2) * G * length * abs(math.sin(q))
