"""Counterfactual reward scoring: what does the training reward actually pay a policy for a behaviour?

Runs exported ONNX policies in the training environment (GPU, final curriculum stage: wide servo
randomization, pushes, head commands, noise, delays) and reports the reward per second each term pays,
under the current recipe and under candidate variants computed from the same rollout:

  C   track_yaw_rate_fine faded out with the linear command (gate (0.10, 0.20) m/s)
  D   stand_torque charged only after the command has been zero for 1.0 s
  CD  both

Commands either follow the environment's own resampling (``--mode natural``, the true training mix) or
are forced (``--mode forced --cmd vx,vy,wz``). The command the policy SEES can differ from the one it is
SCORED on: ``--feed-zero`` (the policy stands while scored on the command) or ``--feed-lin-scale K``
(linear commands at or above ``--feed-lin-above`` are fed K times larger, so a slow policy actually walks
faster). That measures, inside one policy, what the reward pays for standing still, walking faster or
turning, without comparing checkpoints that also differ in everything else they learned.

Falls end an episode; PPO with gamma 0.99 at 50 Hz loses about 2 s of reward per fall, so the
"fall-adjusted" total subtracts 2 s of the policy's own mean reward rate per fall. Intervals are
bootstrap 95 % over environments.

usage:
  uv run scripts/counterfactual_reward.py --policy fast=runs/a.onnx --policy slow=runs/b.onnx --mode natural
  uv run scripts/counterfactual_reward.py --policy p=runs/a.onnx --mode forced --cmd 0.3,0,0 --feed-lin-scale 1.8
  add --pt logs/.../model_N.pt (one per policy, same order) and --sample to add the policy's action noise.
"""
import argparse
import json
from pathlib import Path

import numpy as np

GATE = (0.10, 0.20)
SETTLE_S = 1.0
FALL_COST_S = 2.0
VARIANTS = ("current", "C", "D", "CD")


def gate_factor(cmd_xy_norm, lo=GATE[0], hi=GATE[1]):
    return np.clip((hi - cmd_xy_norm) / (hi - lo), 0.0, 1.0)


def variant_rates(step_reward, names, cmd_xy_norm, settled):
    """Per-env reward per second under each variant, from per-step weighted term values.

    step_reward [T, N, K] (raw x weight, no dt: reward per second), cmd_xy_norm [T, N], settled [T, N] bool
    (zero command for >= SETTLE_S). Returns {variant: [N]} means over T, and {variant: [K]} per-term means."""
    k_yaw, k_st = names.index("track_yaw_rate_fine"), names.index("stand_torque")
    out, terms = {}, {}
    for v in VARIANTS:
        r = step_reward.copy()
        if v in ("C", "CD"):
            r[:, :, k_yaw] *= gate_factor(cmd_xy_norm)
        if v in ("D", "CD"):
            r[:, :, k_st] *= settled
        out[v] = r.sum(axis=2).mean(axis=0)
        terms[v] = r.mean(axis=(0, 1))
    return out, terms


def bootstrap(x, n=1000, rng=None):
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def batched_session(path, threads=8):
    import onnx
    import onnxruntime as ort
    m = onnx.load(path)
    for v in (m.graph.input[0], m.graph.output[0]):
        v.type.tensor_type.shape.dim[0].dim_param = "N"
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    return ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])


def action_std(pt):
    import torch
    d = torch.load(pt, map_location="cpu", weights_only=False)
    return d["actor_state_dict"]["distribution.std_param"].numpy().astype(np.float32)


def run(policies, args):
    import torch
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg("Mjlab-Velocity-Flat-MicroDuck-HD1910")
    cfg.scene.num_envs = args.envs
    if args.mode == "forced":
        cfg.commands["twist"].resampling_time_range = (1e6, 1e6)
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    env.common_step_counter = args.stage_iter * 24
    env.curriculum_manager.compute(env_ids=torch.arange(args.envs, device=env.device))
    rm, tw = env.reward_manager, env.command_manager.get_term("twist")
    names = list(rm.active_terms)
    dt = float(env.step_dt)
    forced = torch.tensor(args.cmd, device=env.device, dtype=torch.float32) if args.mode == "forced" else None
    results = {}
    for name, (onnx_path, pt) in policies.items():
        sess = batched_session(onnx_path)
        inp = sess.get_inputs()[0].name
        std = action_std(pt) if (args.sample and pt) else None
        rng = np.random.default_rng(args.seed)
        torch.manual_seed(args.seed)
        obs, _ = env.reset()
        zero_age = np.zeros(args.envs, dtype=np.float32)
        rew, cmdn, settled, falls = [], [], [], np.zeros(args.envs, dtype=np.int64)
        st_bucket = {"after_stop_<1s": 0.0, "after_reset_<1s": 0.0, "settled_>=1s": 0.0}
        speed_pairs = []
        for k in range(args.steps):
            if forced is not None:
                tw.vel_command_b[:] = forced
                tw.is_standing_env[:] = bool(forced.abs().sum() == 0)
            cmd = tw.vel_command_b.detach().cpu().numpy().copy()
            o = obs["actor"].detach().cpu().numpy().astype(np.float32)
            fed = cmd.copy()
            if args.feed_zero:
                fed[:] = 0.0
            elif args.feed_lin_scale != 1.0:
                fast = np.linalg.norm(cmd[:, :2], axis=1) >= args.feed_lin_above
                fed[fast, :2] *= args.feed_lin_scale
            o[:, 48:51] = fed
            a = sess.run(None, {inp: o})[0]
            if std is not None:
                a = a + rng.standard_normal(a.shape).astype(np.float32) * std
            ep_len_before = env.episode_length_buf.detach().cpu().numpy().copy()
            obs, _, term, trunc, _ = env.step(torch.as_tensor(a, device=env.device))
            sr = rm._step_reward.detach().cpu().numpy().copy()
            cn = np.linalg.norm(cmd[:, :2], axis=1)
            is_zero = (cn + np.abs(cmd[:, 2])) < 0.01
            fresh = ep_len_before <= 0  # this step started an episode
            zero_age[~is_zero | fresh] = 0.0
            zero_age += dt * is_zero
            if k >= args.skip:
                rew.append(sr)
                cmdn.append(cn)
                settled.append(zero_age >= SETTLE_S - 0.5 * dt)
                falls += term.detach().cpu().numpy() & ~trunc.detach().cpu().numpy()
                st = sr[:, names.index("stand_torque")]
                since_reset = ep_len_before * dt
                young = zero_age < SETTLE_S - 0.5 * dt
                st_bucket["after_reset_<1s"] += float(st[young & (since_reset < SETTLE_S)].sum())
                st_bucket["after_stop_<1s"] += float(st[young & (since_reset >= SETTLE_S)].sum())
                st_bucket["settled_>=1s"] += float(st[~young].sum())
                v = env.scene["robot"].data.root_link_lin_vel_b[:, :2].detach().cpu().numpy()
                speed_pairs.append(np.stack([cn, np.einsum("ij,ij->i", v, cmd[:, :2]) / np.maximum(cn, 1e-6)], 1))
        R, C, S = np.stack(rew), np.stack(cmdn), np.stack(settled).astype(np.float32)
        rates, terms = variant_rates(R, names, C, S)
        window_s = R.shape[0] * dt
        sp = np.concatenate(speed_pairs)
        bins = [(0.05, 0.15), (0.15, 0.25), (0.25, 0.45)]
        speed = {f"{lo:.2f}-{hi:.2f}": float(np.mean(sp[(sp[:, 0] >= lo) & (sp[:, 0] < hi), 1])) if np.any((sp[:, 0] >= lo) & (sp[:, 0] < hi)) else None
                 for lo, hi in bins}
        tot_st = sum(st_bucket.values())
        res = dict(falls=int(falls.sum()), window_s=window_s, speed_along_cmd=speed,
                   stand_torque_share={b: (v / tot_st if tot_st else None) for b, v in st_bucket.items()}, variants={})
        for v in VARIANTS:
            x = rates[v]
            fall_cost = falls.sum() * FALL_COST_S * float(x.mean()) / (args.envs * window_s)
            res["variants"][v] = dict(mean=float(x.mean()), ci=bootstrap(x), fall_adjusted=float(x.mean() - fall_cost),
                                      terms={n: float(t) for n, t in zip(names, terms[v]) if abs(t) > 1e-4}, per_env=x.tolist())
        results[name] = res
        print(f"done {name}: current {res['variants']['current']['mean']:.3f} /s, falls {res['falls']}")
    env.close()
    return results, names


def report(results):
    pols = list(results)
    print("\nreward per second, fall-adjusted [95 % CI of the raw mean]")
    print(f"{'variant':10s}" + "".join(f"{p:>30s}" for p in pols))
    for v in VARIANTS:
        print(f"{v:10s}" + "".join(f"{results[p]['variants'][v]['fall_adjusted']:12.3f} [{results[p]['variants'][v]['ci'][0]:6.3f},{results[p]['variants'][v]['ci'][1]:6.3f}]" for p in pols))
    if len(pols) >= 2:
        a, b = pols[0], pols[1]
        print(f"\nmargin {a} - {b} (fall-adjusted):", "  ".join(f"{v} {results[a]['variants'][v]['fall_adjusted'] - results[b]['variants'][v]['fall_adjusted']:+.3f}" for v in VARIANTS))
    print("\nstand_torque cost share by time since the zero command began:")
    for p in pols:
        print(f"  {p:28s}", {k: (round(v, 3) if v is not None else None) for k, v in results[p]["stand_torque_share"].items()})
    print("\nmean speed along the command, by command speed [m/s]:")
    for p in pols:
        print(f"  {p:28s}", {k: (round(v, 3) if v is not None else None) for k, v in results[p]["speed_along_cmd"].items()}, "falls", results[p]["falls"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", action="append", required=True, metavar="NAME=ONNX")
    ap.add_argument("--pt", action="append", default=[], help="checkpoint per policy (same order), for --sample")
    ap.add_argument("--sample", action="store_true", help="add the policy's Gaussian action noise (what PPO optimizes)")
    ap.add_argument("--mode", choices=("natural", "forced"), default="natural")
    ap.add_argument("--cmd", type=lambda s: [float(x) for x in s.split(",")], default=[0.0, 0.0, 0.0])
    ap.add_argument("--feed-zero", action="store_true")
    ap.add_argument("--feed-lin-scale", type=float, default=1.0)
    ap.add_argument("--feed-lin-above", type=float, default=0.2)
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--skip", type=int, default=150)
    ap.add_argument("--stage-iter", type=int, default=6000, help="curriculum stage to score at (iteration)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--json", help="write results here (per-env rates included)")
    args = ap.parse_args()
    pols = {}
    for i, spec in enumerate(args.policy):
        name, path = spec.split("=", 1)
        pols[name] = (path, args.pt[i] if i < len(args.pt) else None)
    results, names = run(pols, args)
    report(results)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(dict(args=vars(args), terms=names, results=results)))
        print("wrote", args.json)


if __name__ == "__main__":
    main()
