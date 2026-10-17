"""Read an HD-1910's registers, diff them against a settings profile, and (only with --apply) write it.

The default profile (src/mjlab_microduck/bench/profiles/hd1910_sim_match.json) makes the servo behave the
way the simulation assumes: position mode, no deadband, no acceleration profile, full duty, P from the
gain map, D = I = 0. EEPROM writes unlock (55 = 0) and re-lock; RAM gains revert to the EEPROM copies at
power-up, so pass --persist-gains to also write EEPROM 21/22/23.

usage:
  uv run scripts/bench_configure.py --port /dev/ttyUSB0                      # read and diff, writes nothing
  uv run scripts/bench_configure.py --port /dev/ttyUSB0 --kp 3 --apply       # write the profile, P = 3
  uv run scripts/bench_configure.py --port /dev/ttyUSB0 --set-id 20 --apply  # one servo on the bus only
  uv run scripts/bench_configure.py --fake --kp 3 --apply                    # rehearse without hardware
"""
import argparse
import json
import sys
from pathlib import Path

from mjlab_microduck.bench import hls
from mjlab_microduck.bench.cli import add_bus_args, open_from_args, safe_off

PROFILE = Path(__file__).resolve().parents[1] / "src" / "mjlab_microduck" / "bench" / "profiles" / "hd1910_sim_match.json"


def targets(profile: dict, kp=None, kd=None, ki=None, persist_gains=False) -> dict:
    """{register: value} the profile asks for (None entries skipped), CLI gains overriding it."""
    eeprom = {k: v for k, v in profile.get("eeprom", {}).items() if v is not None}
    ram = {k: v for k, v in profile.get("ram", {}).items() if v is not None}
    for name, v in (("p_gain", kp), ("d_gain", kd), ("i_gain", ki)):
        if v is not None:
            ram[name] = v
    if persist_gains:
        for name in ("p_gain", "d_gain", "i_gain"):
            if name in ram:
                eeprom[f"{name}_eeprom"] = ram[name]
    return {**eeprom, **ram}


def diff(table: dict, want: dict) -> list:
    return [(name, table.get(name), v) for name, v in want.items() if table.get(name) != v]


def configure(bus, sid, want: dict, apply=False, log=print) -> list:
    """Diff the servo against `want`; with apply, write (torque off first) and return what still differs."""
    table = bus.read_table(sid)
    model = table["model"]
    if model != hls.MODEL_NUMBER:
        log(f"warning: model number {model:#06x}, expected {hls.MODEL_NUMBER:#06x} (HD-1910-C001)")
    todo = diff(table, want)
    for name, now, v in todo:
        log(f"  {name:22s} {now!s:>6} -> {v}")
    if not todo:
        log("  matches the profile")
    if not apply or not todo:
        if todo:
            log("dry run: nothing written (pass --apply)")
        return todo
    safe_off(bus, sid)
    eeprom = {n: v for n, _, v in todo if hls.REG[n].area == "eeprom"}
    if eeprom:
        bus.write_eeprom(sid, eeprom)
    for n, _, v in todo:
        if hls.REG[n].area == "ram":
            bus.write_reg(sid, n, v)
    left = diff(bus.read_table(sid), want)
    log("written and verified" if not left else f"STILL DIFFERENT after writing: {left}")
    return left


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_bus_args(ap)
    ap.add_argument("--profile", default=str(PROFILE))
    ap.add_argument("--kp", type=int, help="P gain (RAM 50); the gain map says which P matches the sim")
    ap.add_argument("--kd", type=int)
    ap.add_argument("--ki", type=int)
    ap.add_argument("--persist-gains", action="store_true", help="also write the EEPROM gains (21/22/23)")
    ap.add_argument("--set-id", type=int, help="change the servo id (EEPROM 5); one servo on the bus")
    ap.add_argument("--apply", action="store_true", help="write; without it nothing is written")
    ap.add_argument("--json", help="save the full register table here")
    args = ap.parse_args()

    bus, clock, ctrl = open_from_args(args)
    sid = args.id
    if not bus.ping(sid):
        sys.exit(f"no answer from id {sid}")
    profile = json.loads(Path(args.profile).read_text())
    table = bus.read_table(sid)
    print(f"id {sid}: model {table['model']:#06x}, mode {table['mode']}, torque {table['torque_enable']}, "
          f"P/D/I RAM {table['p_gain']}/{table['d_gain']}/{table['i_gain']} "
          f"EEPROM {table['p_gain_eeprom']}/{table['d_gain_eeprom']}/{table['i_gain_eeprom']}, "
          f"{table['present_voltage'] / 10:.1f} V, {table['present_temperature']} C")
    for name in profile.get("inspect", []):
        print(f"  (inspect) {name:22s} {table[name]}")
    if args.json:
        Path(args.json).write_text(json.dumps(table, indent=1))
    want = targets(profile, args.kp, args.kd, args.ki, args.persist_gains)
    left = configure(bus, sid, want, apply=args.apply)
    if args.set_id is not None:
        if not args.apply:
            print(f"dry run: would set id {sid} -> {args.set_id} (pass --apply)")
        else:
            bus.write_eeprom(sid, {"id": args.set_id})
            print(f"id {sid} -> {args.set_id}; reconnect with --id {args.set_id}")
    sys.exit(1 if (args.apply and left) else 0)


if __name__ == "__main__":
    main()
