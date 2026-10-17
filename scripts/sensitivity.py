"""Servo-model sensitivity matrix: one policy, the servo model deliberately wrong in one way at a time.

Every condition is one harness-v2 evaluation (`scripts/eval_walk.py --hd1910`) with a single change:
battery voltage and sag at the training range's edges, BAM friction / rotor inertia / gain / torque
constant / resistance scaled, longer actuator and joint-velocity delays than training used, and the
backlash body. Each is judged with the iteration-1 gate table (`eval_walk.gate_rows`, against the v1
baseline by default) and summarised as

  robust    no fall, no dead zone, every non-torque gated row passes, speeds within 0.03 m/s of nominal,
            standing torque within 15 % of nominal, substep peak below the 0.9 N.m fail line
  degraded  no fall and no dead zone, but one of the above slips
  fails     a gated fall, or a 0.10 m/s command (fwd, back or left) gives < 0.03 m/s

Standing torque is judged against the nominal condition, not the gate's absolute threshold: the
delivered policy sits on that threshold (0.317 vs 0.317 N.m), so any noise would flip it.

The point is to learn which servo properties the policy cares about before a real servo is measured:
an axis that "fails" at a factor the bench cannot rule out is one the bench must measure carefully.

usage:
  uv run scripts/sensitivity.py --walking runs/hd1910_v2b_5500.onnx --tag v2b_5500 [--jobs 16] [--replicas 3]
  uv run scripts/sensitivity.py --table-only --tag v2b_5500          # re-render table.md from the JSONs
Output: runs/sens/<tag>/{index.json, <cond>.json, <cond>.log, table.md}
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = REPO / "runs" / "hd1910_v1_final_eval_h2.json"

# (name, eval_walk flags). Training covers: vin 6.5-8.2 V, sag 0-0.2, delay 3-6 substeps, joint_vel lag 1,
# friction x0.9-1.1, armature x0.9-1.1; kp, kt and R are fixed. Factors beyond that are on purpose.
CONDITIONS = [
    ("nominal", []),
    ("vin6.5", ["--vin", "6.5"]),
    ("vin8.2", ["--vin", "8.2"]),
    ("vin6.5_sag0.2", ["--vin", "6.5", "--vin-drop-gain", "0.2"]),
    ("friction_x0.5", ["--bam", "friction*=0.5"]),
    ("friction_x1.5", ["--bam", "friction*=1.5"]),
    ("friction_x2", ["--bam", "friction*=2"]),
    ("armature_x0.5", ["--bam", "armature*=0.5"]),
    ("armature_x1.5", ["--bam", "armature*=1.5"]),
    ("armature_x2", ["--bam", "armature*=2"]),
    ("kp_x0.5", ["--bam", "kp*=0.5"]),
    ("kp_x0.7", ["--bam", "kp*=0.7"]),
    ("kp_x1.3", ["--bam", "kp*=1.3"]),
    ("kp_x1.4", ["--bam", "kp*=1.4"]),
    ("kp_x2", ["--bam", "kp*=2"]),
    ("kp_x4", ["--bam", "kp*=4"]),
    ("kt_x0.8", ["--bam", "kt*=0.8"]),
    ("kt_x1.2", ["--bam", "kt*=1.2"]),
    ("R_x0.8", ["--bam", "R*=0.8"]),
    ("R_x1.2", ["--bam", "R*=1.2"]),
    ("delay5-8", ["--action-delay", "5", "8"]),
    ("delay8-12", ["--action-delay", "8", "12"]),
    ("jvlag2", ["--joint-vel-lag", "2"]),
    ("backlash", ["--backlash"]),
]
# Column -> gate-table row name (eval_walk.gate_rows).
COLUMNS = [
    ("fwd .10", "fwd 0.10 vx [m/s]"),
    ("fwd .20", "fwd 0.20 vx [m/s]"),
    ("fwd .30", "fwd 0.30 vx [m/s]"),
    ("back .10", "back 0.10 vx [m/s]"),
    ("left .10", "left 0.10 vy [m/s]"),
    ("turn", "turn |wz| mean [rad/s]"),
    ("turn rest", "turn from rest |wz| min @1.0 [rad/s]"),
    ("falls", "falls (sched/steps/stand/kick<=0.3)"),
    ("0.4 kick", "falls at the 0.4 kick (report)"),
    ("stand max", "stand torque max joint [N.m]"),
    ("stand legs", "stand torque leg sum [N.m]"),
    ("peak", "substep peak worst [N.m]"),
]
SPEEDS = ("fwd 0.10 vx [m/s]", "fwd 0.20 vx [m/s]", "fwd 0.30 vx [m/s]")
DEAD = ("fwd 0.10 vx [m/s]", "back 0.10 vx [m/s]", "left 0.10 vy [m/s]")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_condition(name, flags, onnx, outdir, replicas, resume):
    out = outdir / f"{name}.json"
    if resume and out.exists():
        return name, 0, 0.0
    cmd = [sys.executable, str(REPO / "scripts" / "eval_walk.py"), "--walking", str(onnx), "--hd1910",
           "--replicas", str(replicas), "--json", str(out), *flags]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    t0 = time.time()
    with open(outdir / f"{name}.log", "w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=REPO).returncode
    return name, rc, time.time() - t0


TORQUE = ("stand torque max joint [N.m]", "stand torque leg sum [N.m]")


def verdict(rows, nominal_rows):
    """robust / degraded / fails for one condition (see the module docstring)."""
    by = {name: (y, v) for name, _, y, v in rows}
    nom = {name: y for name, _, y, _ in (nominal_rows or [])}
    if (by["falls (sched/steps/stand/kick<=0.3)"][0] or 0) > 0:
        return "fails"
    if any(by[k][0] is None or abs(by[k][0]) < 0.03 for k in DEAD):
        return "fails"
    if any(v == "FAIL" for n, (_, v) in by.items() if n not in TORQUE):
        return "degraded"
    if any(nom.get(k) is not None and by[k][0] is not None and by[k][0] < nom[k] - 0.03 for k in SPEEDS):
        return "degraded"
    if any(nom.get(k) and by[k][0] is not None and by[k][0] > 1.15 * nom[k] for k in TORQUE):
        return "degraded"
    return "robust"


def fall_sites(doc):
    """'push x3, steps x1': which rollouts fell, counted over replicas (the 0.4 kick excluded)."""
    ev_limit = 9.05  # eval_walk._falls: a push-rollout fall after 3 kicks x 3 s is the 0.4 kick
    counts = {}
    for key, reps in (doc.get("replicas") or {}).items():
        if key == "steps_metrics":
            continue
        for rep in reps:
            for r in rep:
                t = r.get("fell_at")
                if t is not None and not (key == "push" and t > ev_limit):
                    counts[key] = counts.get(key, 0) + 1
    return ", ".join(f"{k} x{n}" for k, n in counts.items()) or "-"


def short(row_name):
    return row_name.split(" [")[0].replace(" (sched/steps/stand/kick<=0.3)", "")


def table(outdir, baseline_path, conditions=CONDITIONS):
    ev = _load("eval_walk")
    base = json.loads(Path(baseline_path).read_text())
    docs = {}
    for name, _ in conditions:
        p = outdir / f"{name}.json"
        if p.exists():
            docs[name] = json.loads(p.read_text())
    rows = {name: ev.gate_rows(base, doc) for name, doc in docs.items()}
    nominal = rows.get("nominal")
    nom = {n: y for n, _, y, _ in (nominal or [])}
    fmt = lambda v: "n/a" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
    head = ["condition", *[c for c, _ in COLUMNS], "Δfwd vs nominal", "verdict", "falls in", "failing gate rows"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    verdicts = {}
    for name, flags in conditions:
        if name not in rows:
            lines.append(f"| {name} | " + " | ".join(["missing"] * (len(head) - 1)) + " |")
            continue
        by = {n: (y, v) for n, _, y, v in rows[name]}
        cells = []
        for _, key in COLUMNS:
            y, v = by[key]
            cells.append(fmt(y) + (" ✗" if v == "FAIL" else " ⚠" if v == "warn" else ""))
        deltas = [by[k][0] - nom[k] if by[k][0] is not None and nom.get(k) is not None else None for k in SPEEDS]
        dtext = " / ".join("n/a" if d is None else f"{d:+.3f}" for d in deltas)
        verdicts[name] = verdict(rows[name], nominal)
        failing = [short(n) for n, (_, v) in by.items() if v == "FAIL"]
        lines.append(f"| {name} | " + " | ".join(cells)
                     + f" | {dtext} | {verdicts[name]} | {fall_sites(docs[name])} | {', '.join(failing) or '-'} |")
    summary = {v: [n for n, x in verdicts.items() if x == v] for v in ("robust", "degraded", "fails")}
    lines += ["", f"Gate rows judged against `{Path(baseline_path).name}` (✗ = gated row fails, ⚠ = warning). "
              "Verdicts judge standing torque against the nominal row (see the script docstring).",
              "Δfwd = fwd 0.10 / 0.20 / 0.30 vx minus the nominal condition's. Push-rollout falls are at the "
              "0.2 or 0.3 m/s kick while walking at 0.10 m/s."]
    for v, names in summary.items():
        lines.append(f"- **{v}** ({len(names)}): {', '.join(names) or '-'}")
    text = "\n".join(lines) + "\n"
    (outdir / "table.md").write_text(text)
    return text, verdicts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--walking", help="policy ONNX")
    ap.add_argument("--tag", required=True, help="output folder name under runs/sens/")
    ap.add_argument("--replicas", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=min(16, os.cpu_count() or 1))
    ap.add_argument("--baseline", default=str(DEFAULT_BASELINE), help="gate table's A (default: v1 final, harness v2)")
    ap.add_argument("--only", nargs="*", help="run only these condition names")
    ap.add_argument("--resume", action="store_true", help="skip conditions whose JSON exists")
    ap.add_argument("--table-only", action="store_true", help="re-render table.md from existing JSONs")
    args = ap.parse_args()

    outdir = REPO / "runs" / "sens" / args.tag
    outdir.mkdir(parents=True, exist_ok=True)
    conds = [c for c in CONDITIONS if not args.only or c[0] in args.only]
    if not args.table_only:
        if not args.walking:
            ap.error("--walking is required unless --table-only")
        onnx = Path(args.walking).resolve()
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            results = list(pool.map(lambda c: run_condition(*c, onnx, outdir, args.replicas, args.resume), conds))
        failed = [(n, rc) for n, rc, _ in results if rc != 0]
        index = dict(onnx=str(onnx), replicas=args.replicas, baseline=args.baseline, wall_s=round(time.time() - t0, 1),
                     conditions={n: f for n, f in conds}, seconds={n: round(s, 1) for n, _, s in results},
                     failed=failed)
        (outdir / "index.json").write_text(json.dumps(index, indent=1))
        print(f"{len(conds)} conditions in {index['wall_s']} s" + (f"; FAILED to run: {failed}" if failed else ""))
    text, _ = table(outdir, args.baseline, conds)
    print(text)
    print("wrote", outdir / "table.md")


if __name__ == "__main__":
    main()
