"""Evaluation pipeline: evaluate checkpoints as training writes them, with nobody watching.

Meant to run detached next to scripts/train_queue.py (same reason: background shells of an interactive
session can be stopped). Spec (JSON):

  {"baseline": "runs/hd1910_v1_final_eval_h2.json", "head_ref": "runs/hd1910_v3C_s42_4000.onnx",
   "runs": [{"label": "v3CQD_s7", "run_name": "hd1910_v3CQD_s7",
             "iters": [3000, 4000, 4500, 5000, 5500, 5750, 6000], "final": [5000, 5500, 5750, 6000]}, ...]}

For every run and iteration, once a logs/rsl_rl/*/*_<run_name>/model_<it>.pt exists (newest directory that has
it): export runs/hd1910_<label>_<it>.onnx, run the harness (3 replicas, *_eval_h3.json) and the six-direction
start probe (runs/start_probe/<label>_<it>.json). Once all of a run's "final" iterations are in, compare head
tracking against "head_ref" for those that pass the gates and the probe, choose the candidate (every gated
row, probe <= 10 % in every direction, fwd 0.20 / 0.30 >= 0.17 / 0.25, stand torque <= 0.07, head tracking
>= 95 % of the reference, then the smallest step error sum), and run the contract test and the sensitivity
matrix on it. The final bar per run is judged and written, with every number, to <spec>.md after each step.

Existing outputs are reused, so a restart picks up where it stopped. Create <spec>.STOP to make it exit.

  setsid nohup .venv/bin/python scripts/eval_pipeline.py runs/pipeline/iter3d.json > runs/pipeline/iter3d.out 2>&1 &
"""
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = str(REPO / ".venv" / "bin" / "python")
TASK = "Mjlab-Velocity-Flat-MicroDuck-HD1910"
PROBE_MAX = 0.10
HEAD_MIN = 0.95


def _ev():
    spec = importlib.util.spec_from_file_location("eval_walk", REPO / "scripts" / "eval_walk.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_checkpoint(run_name, it, log_root=REPO / "logs" / "rsl_rl"):
    hits = sorted(Path(log_root).glob(f"*/*_{run_name}/model_{it}.pt"))
    return hits[-1] if hits else None


def gate_summary(rows):
    """{name: (value, verdict)} from eval_walk.gate_rows output, plus the failing names."""
    by = {name: (y, v) for name, _, y, v in rows}
    return by, [name for name, (_, v) in by.items() if v == "FAIL"]


def probe_worst(probe, cond="quiet"):
    """(direction, share) of the worst direction in a start_probe result for one policy."""
    res = next(iter(probe["results"].values()))[cond]
    d = max(res, key=lambda c: res[c]["still"])
    return d, res[d]["still"]


def qualifies(item):
    """Every final-bar item that is measured per checkpoint, except head tracking (missing numbers fail)."""
    v = lambda key, missing: item["gates"].get(key, (None,))[0] if item["gates"].get(key, (None,))[0] is not None else missing
    return (not item["fails"] and item["probe_worst"][1] <= PROBE_MAX
            and v("fwd 0.20 vx [m/s]", -1.0) >= 0.17 and v("fwd 0.30 vx [m/s]", -1.0) >= 0.25
            and v("stand torque max joint [N.m]", 9.0) <= 0.07)


def choose_candidate(items, head):
    """items {it: item}; head {it: ratio to the reference}. Qualifying + head >= 95 %, then smallest step error."""
    ok = [it for it, x in items.items() if qualifies(x) and head.get(it, 0.0) >= HEAD_MIN]
    return min(ok, key=lambda it: items[it]["ss_sum"]) if ok else None


def final_verdict(items, final, cand, extras):
    """Final bar: candidate passes everything; >= 3 of the final iterations pass the gates and the probe."""
    reasons = []
    n_gates = sum(1 for it in final if it in items and not items[it]["fails"])
    n_probe = sum(1 for it in final if it in items and items[it]["probe_worst"][1] <= PROBE_MAX)
    if n_gates < 3:
        reasons.append(f"gated rows pass at {n_gates} of {len(final)}")
    if n_probe < 3:
        reasons.append(f"start probe <= 10 % at {n_probe} of {len(final)}")
    if cand is None:
        reasons.append("no checkpoint passes every per-checkpoint item")
    else:
        if extras.get("sens_fails") is None or extras["sens_fails"] > 4:
            reasons.append(f"sensitivity fails {extras.get('sens_fails')}")
        if not extras.get("contract"):
            reasons.append("contract test")
    return ("PASS" if not reasons else "FAIL"), reasons


class Pipeline:
    def __init__(self, spec_path):
        self.path = Path(spec_path)
        self.spec = json.loads(self.path.read_text())
        self.stop_path = self.path.with_suffix(".STOP")
        self.md_path = self.path.with_suffix(".md")
        self.log_path = self.path.with_suffix(".log")
        self.ev = _ev()
        self.base = json.loads((REPO / self.spec.get("baseline", "runs/hd1910_v1_final_eval_h2.json")).read_text())
        self.items = {r["label"]: {} for r in self.spec["runs"]}
        self.attempts = {}
        self.extras = {r["label"]: {} for r in self.spec["runs"]}

    def log(self, msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")

    def sh(self, cmd, env=None, log=None):
        e = {**os.environ, **(env or {})}
        with open(log or os.devnull, "w") as out:
            return subprocess.run(cmd, cwd=REPO, env=e, stdout=out, stderr=subprocess.STDOUT).returncode

    def evaluate(self, label, it, pt):
        onnx = REPO / "runs" / f"hd1910_{label}_{it}.onnx"
        harness = REPO / "runs" / f"hd1910_{label}_{it}_eval_h3.json"
        probe = REPO / "runs" / "start_probe" / f"{label}_{it}.json"
        if not onnx.exists():
            rc = self.sh([PY, "scripts/export.py", TASK, "--checkpoint-file", str(pt), "--onnx-file", str(onnx), "--num-envs", "1"],
                         log=REPO / "runs" / f"export_{label}_{it}.log")
            if rc or not onnx.exists():
                self.log(f"{label}@{it}: export failed ({rc})")
                return None
        if not harness.exists():
            self.sh([PY, "scripts/eval_walk.py", "--walking", str(onnx), "--hd1910", "--replicas", "3", "--json", str(harness)],
                    log=REPO / "runs" / f"eval_{label}_{it}.log")
        if not probe.exists():
            self.sh([PY, "scripts/start_probe.py", "--policy", f"{label}_{it}={onnx}", "--json", str(probe)],
                    log=REPO / "runs" / "start_probe" / f"{label}_{it}.log")
        if not (harness.exists() and probe.exists()):
            self.log(f"{label}@{it}: harness or probe missing after the run")
            return None
        doc = json.loads(harness.read_text())
        gates, fails = gate_summary(self.ev.gate_rows(self.base, doc))
        ss = sum(abs(s.get("ss_err") or 0.0) for s in (doc["summary"].get("steps_metrics") or []))
        pr = json.loads(probe.read_text())
        quiet = next(iter(pr["results"].values()))["quiet"]
        return dict(onnx=str(onnx.relative_to(REPO)), gates=gates, fails=fails, ss_sum=ss, probe_worst=probe_worst(pr),
                    probe={d: r["still"] for d, r in quiet.items()})

    def head_tracking(self, label, its):
        out = REPO / "runs" / "cf" / f"headtrack_{label}.json"
        if not out.exists():
            pols = [a for it in its for a in ("--policy", f"{label}_{it}=runs/hd1910_{label}_{it}.onnx")]
            self.sh([PY, "scripts/counterfactual_reward.py", "--mode", "natural", "--envs", "1024", "--steps", "1500",
                     "--skip", "150", *pols, "--policy", f"REF={self.spec['head_ref']}", "--json", str(out)],
                    env={"HD1910_ARM": "v2d"}, log=out.with_suffix(".log"))
        if not out.exists():
            return {}
        res = json.loads(out.read_text())["results"]
        term = lambda name: res[name]["variants"]["current"]["terms"].get("head_pose_tracking", 0.0)
        ref = term("REF")
        return {int(n.rsplit("_", 1)[1]): term(n) / ref for n in res if n != "REF"}

    def finish_run(self, run):
        label, final = run["label"], run["final"]
        items, extras = self.items[label], self.extras[label]
        if "verdict" in extras:
            return
        cands = [it for it in final if it in items and qualifies(items[it])]
        extras["head"] = self.head_tracking(label, cands) if cands else {}
        cand = choose_candidate({it: items[it] for it in final if it in items}, extras["head"])
        extras["candidate"] = cand
        if cand is not None:
            onnx = items[cand]["onnx"]
            extras["contract"] = self.sh(["uv", "run", "--with", "pytest", "pytest", "-q", "tests/test_deploy_contract.py"],
                                         env={"MICRODUCK_TEST_ONNX": onnx}) == 0
            tag = f"{label}_{cand}"
            table = REPO / "runs" / "sens" / tag / "table.md"
            if not table.exists():
                self.sh([PY, "scripts/sensitivity.py", "--walking", onnx, "--tag", tag, "--jobs", "8"], log=REPO / "runs" / f"sens_{tag}.out")
            m = re.search(r"\*\*fails\*\* \((\d+)\)", table.read_text()) if table.exists() else None
            extras["sens_fails"] = int(m.group(1)) if m else None
        extras["verdict"], extras["reasons"] = final_verdict(items, final, cand, extras)
        self.log(f"{label}: final bar {extras['verdict']} {extras['reasons']} candidate {cand}")

    def write_md(self):
        lines = ["# Evaluation pipeline: " + self.path.stem, "", f"Updated {time.strftime('%Y-%m-%d %H:%M:%S')}.", ""]
        dirs = ("fwd", "back", "left", "right", "turnL", "turnR")
        for run in self.spec["runs"]:
            label, items, extras = run["label"], self.items[run["label"]], self.extras[run["label"]]
            lines += [f"## {label}", "", "| iter | failing gated rows | fwd .10 / .20 / .30 | turn rest @0.5 | quiet start: "
                      + " / ".join(dirs) + " | Σ step error |", "|---|---|---|---|---|---|"]
            for it in run["iters"]:
                x = items.get(it)
                if x is None:
                    lines.append(f"| {it} | pending | | | | |")
                    continue
                g = x["gates"]
                fv = lambda k: "n/a" if g[k][0] is None else f"{g[k][0]:.3f}"
                lines.append(f"| {it} | {', '.join(f.split(' [')[0].replace('|', '') for f in x['fails']) or '-'} | "
                             f"{fv('fwd 0.10 vx [m/s]')} / {fv('fwd 0.20 vx [m/s]')} / {fv('fwd 0.30 vx [m/s]')} | "
                             f"{fv('turn from rest |wz| min @0.5 [rad/s]')} | "
                             + " / ".join(f"{x['probe'].get(d, float('nan')):.0%}" for d in dirs) + f" | {x['ss_sum']:.3f} |")
            if "verdict" in extras:
                head = ", ".join(f"{it}: {r:.1%}" for it, r in sorted(extras.get("head", {}).items())) or "-"
                lines += ["", f"- head tracking vs reference: {head}",
                          f"- candidate: {extras['candidate']}; contract {extras.get('contract')}; sensitivity fails {extras.get('sens_fails')}",
                          f"- **final bar: {extras['verdict']}** {'; '.join(extras['reasons'])}"]
            lines.append("")
        self.md_path.write_text("\n".join(lines))

    def run(self):
        self.log(f"pipeline pid {os.getpid()} started on {self.path}")
        while not self.stop_path.exists():
            progressed = False
            for run in self.spec["runs"]:
                for it in run["iters"]:
                    if it in self.items[run["label"]] or self.stop_path.exists():
                        continue
                    if self.attempts.get((run["label"], it), 0) >= 2:
                        continue  # failed twice: logged, skipped
                    pt = find_checkpoint(run["run_name"], it)
                    if pt is None:
                        continue
                    self.log(f"{run['label']}@{it}: evaluating {pt.parent.name}/{pt.name}")
                    self.attempts[(run["label"], it)] = self.attempts.get((run["label"], it), 0) + 1
                    item = self.evaluate(run["label"], it, pt)
                    if item is not None:
                        self.items[run["label"]][it] = item
                        self.log(f"{run['label']}@{it}: fails {item['fails']} quiet worst {item['probe_worst']}")
                    progressed = True
                    self.write_md()
                done = lambda it: it in self.items[run["label"]] or self.attempts.get((run["label"], it), 0) >= 2
                if all(done(it) for it in run["final"]):
                    self.finish_run(run)
                    self.write_md()
            if all("verdict" in self.extras[r["label"]] and all(it in self.items[r["label"]] or
                   self.attempts.get((r["label"], it), 0) >= 2 for it in r["iters"]) for r in self.spec["runs"]):
                self.log("all runs evaluated")
                break
            if not progressed:
                time.sleep(120)
        self.write_md()


if __name__ == "__main__":
    Pipeline(sys.argv[1]).run()
