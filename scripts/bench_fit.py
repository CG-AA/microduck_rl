"""Fit the HD-1910 BAM model to pendulum logs and convert it into the simulator's parameter file.

Steps: resample the raw logs (bench_record.py) onto a 5 ms grid -> fit kt, R, rotor inertia, the m6
friction terms (and the duty/rad ratio unless the gain map fixed it) by differential evolution on the
mean position error of the simulated pendulum -> one overlay plot per log -> convert to the XL330-law
JSON the sim's actuator reads, with the kp_fw that reproduces the measured stiffness, and the P register
that reproduces the stiffness the current policies trained at.

Pass the gain map (bench_static.py gainmap) so the duty/rad factor is measured, not fitted: position
logs alone trade it against kt/R. With --truth (written by bench_record.py --fake) the script checks
the identifiable combinations against the planted servo and exits 1 if any is off by more than its
tolerance (TOLERANCES, set from the sensitivity matrix).

usage:
  uv run scripts/bench_fit.py --raw runs/bench/raw --gainmap runs/bench/gainmap.json --out runs/bench/fit_v1
  uv run scripts/bench_fit.py --raw runs/bench_sim/raw --truth runs/bench_sim/raw/truth.json.planted --out runs/bench_sim/fit
Outputs in --out: processed/, fit.json, plots/*.png, hd1910_m6_sim.json, report.json.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from mjlab_microduck.bench import bam_hd1910 as bh


# What a pendulum fit has to get right, with tolerances set by the servo-model sensitivity matrix
# (docs/hd1910.md): the policy barely notices rotor inertia (x0.5-2) or kt/R within 20 %, fails on friction
# x2 and on firmware gain x0.7.
TOLERANCES = dict(kt_over_R=0.10, kt2_over_R=0.10, duty_per_rad_per_p=0.05, holding_friction=0.30, armature=0.25)


def holding_friction(params: dict) -> float:
    """BAM's friction budget at rest while holding a 0.1 N.m load (motor and load torques opposed), the
    standing case: one number that the m6 friction terms add up to."""
    m = bh.model_from_json(params)
    fl, _ = m.compute_frictions(0.1, -0.1, 0.0)
    return float(fl)


def identifiable(params: dict) -> dict:
    """What pendulum logs pin down: static torque per volt, back-EMF damping, inertia, duty per rad/P,
    and the friction budget while holding a load."""
    return dict(kt_over_R=params["kt"] / params["R"], kt2_over_R=params["kt"] ** 2 / params["R"],
                armature=params["armature"], duty_per_rad_per_p=params["error_gain"] * params.get("error_gain_ratio", 1.0),
                holding_friction=holding_friction(params))


def compare_truth(fit_params: dict, truth: dict, tol: dict | float) -> tuple[bool, dict]:
    got, want = identifiable(fit_params), identifiable(truth)
    tols = tol if isinstance(tol, dict) else {k: tol for k in want}
    rel = {k: abs(got[k] - want[k]) / abs(want[k]) for k in want}
    return all(rel[k] <= tols[k] for k in rel), dict(fit=got, truth=want, rel_err=rel, tol=tols)


def plot(m, processed_dir, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    errors = {}
    for path in sorted(Path(processed_dir).glob("*.json")):
        log = json.loads(path.read_text())
        sim = bh.rollout(m, log)
        t = np.arange(len(log["entries"])) * log["dt"]
        q = np.array([e["position"] for e in log["entries"]])
        goal = np.array([e["goal_position"] for e in log["entries"]])
        te = np.array([e["torque_enable"] for e in log["entries"]])
        errors[path.name] = float(np.mean(np.abs(sim - q)))
        fig, ax = plt.subplots(figsize=(9, 3.2))
        ax.plot(t, goal, color="0.6", lw=1, label="goal")
        ax.plot(t, q, color="C0", lw=1.4, label="servo (log)")
        ax.plot(t, sim, color="C1", lw=1.2, ls="--", label="model")
        if (~te).any():
            ax.fill_between(t, *ax.get_ylim(), where=~te, color="0.9", zorder=0, label="torque off")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("angle from hanging [rad]")
        ax.set_title(f"{path.stem}  (mean |error| {np.degrees(errors[path.name]):.2f} deg)", fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout()
        fig.savefig(out_dir / f"{path.stem}.png", dpi=110)
        plt.close(fig)
    return errors


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="folder of raw logs from bench_record.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--model", default="m6")
    ap.add_argument("--gainmap", help="bench_static.py gainmap JSON: fixes error_gain, ratio 1")
    ap.add_argument("--error-gain", type=float, help="duty/rad per P unit (instead of --gainmap)")
    ap.add_argument("--evals", type=int, default=3000, help="objective evaluations (real data: 20000+)")
    ap.add_argument("--workers", type=int, default=1, help="parallel processes (-1 = all cores)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--validation-kp", type=int, help="hold out the logs recorded at this P")
    ap.add_argument("--p-register", type=float, help="P used on the robot (default: the P matching the trained stiffness)")
    ap.add_argument("--truth", help="planted parameters (bench_record.py --fake) to check the fit against")
    ap.add_argument("--tol", type=float, help="--truth: one max relative error for every quantity "
                    "(default: TOLERANCES, set from the sensitivity matrix)")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    processed = out / "processed"
    written = bh.process(args.raw, processed, args.dt)
    print(f"processed {len(written)} logs -> {processed}")

    fixed, error_gain = {}, bh.ERROR_GAIN_PLACEHOLDER
    if args.gainmap:
        error_gain = float(json.loads(Path(args.gainmap).read_text())["error_gain"])
    if args.error_gain is not None:
        error_gain = args.error_gain
    if args.gainmap or args.error_gain is not None:
        fixed["error_gain_ratio"] = 1.0
    else:
        print(f"warning: no gain map; error_gain {error_gain} is a placeholder and the ratio is fitted "
              "(position logs trade it against kt/R)")
    fixed["q_offset"] = 0.0  # the recorder measures the hanging zero

    params = bh.fit(processed, model=args.model, error_gain=error_gain, fixed=fixed, evals=args.evals,
                    seed=args.seed, workers=args.workers, validation_kp=args.validation_kp)
    (out / "fit.json").write_text(json.dumps(params, indent=1))
    m = bh.model_from_json(params)
    report = dict(fit=params["_fit"], identifiable=identifiable(params))
    if not args.no_plots:
        report["per_log_error_rad"] = plot(m, processed, out / "plots")
        worst = max(report["per_log_error_rad"].items(), key=lambda kv: kv[1])
        print(f"plots in {out / 'plots'}; worst log {worst[0]} ({np.degrees(worst[1]):.2f} deg mean error)")

    p_exact, p_int = bh.p_register_for_sim(params)
    p_reg = args.p_register if args.p_register is not None else p_int
    sim, kp_fw = bh.to_xl330_law(params, p_reg)
    (out / "hd1910_m6_sim.json").write_text(json.dumps(sim, indent=1))
    report.update(p_for_trained_stiffness=p_exact, p_register=p_reg, kp_fw=kp_fw)
    print(f"kt {params['kt']:.3f}  R {params['R']:.2f}  armature {params['armature']:.2e}  "
          f"duty/rad per P {identifiable(params)['duty_per_rad_per_p']:.4f}")
    print(f"trained stiffness (kp_fw {bh.SIM_KP_FW:g}) needs P = {p_exact:.2f}; at P = {p_reg:g} the sim's "
          f"kp_fw is {kp_fw:.1f} -> {out / 'hd1910_m6_sim.json'}")

    code = 0
    if args.truth:
        ok, cmp = compare_truth(params, json.loads(Path(args.truth).read_text()),
                                TOLERANCES if args.tol is None else args.tol)
        report["truth_check"] = dict(ok=ok, **cmp)
        for k, v in cmp["rel_err"].items():
            print(f"  {k:20s} fit {cmp['fit'][k]:.5g}  truth {cmp['truth'][k]:.5g}  "
                  f"({100 * v:.1f} %, tolerance {100 * cmp['tol'][k]:.0f} %){'' if v <= cmp['tol'][k] else '  <- FAIL'}")
        print("truth check:", "PASS" if ok else "FAIL")
        code = 0 if ok else 1
    (out / "report.json").write_text(json.dumps(report, indent=1))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
