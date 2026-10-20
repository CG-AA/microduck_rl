"""Start probe: does a policy start a small command from a settled stand, across the training env's robots?

The harness (scripts/eval_walk.py) tests one nominal robot. This runs the training env (GPU, final
curriculum stage: servo/body randomization) with every env standing for --stand-s at zero command,
then commanding fwd 0.10 m/s or turn 0.5 rad/s for --cmd-s, and reports the mean response over the
last 4 s and the share of envs that stayed still (|response| < 20 % of the command). Two conditions:

  trained  the env as trained: head/body pose commands re-drawn every few seconds, pushes on
  quiet    head/body pose commands zero and no pushes: the robot's idle state on hardware

Found the HD-1910 iteration-3b start threshold (2026-10-19): v3C s7@4000 stays still in 2 % / 16 %
(fwd / turn) as trained but 29 % / 83 % quiet. ``--noise`` adds training-like action noise: at the
learned 0.2-0.3 rad it rescues every start, which is why training never saw the failure (3c). The env is built with HD1910_ARM=v2d whatever the
caller's environment, so quiet episodes of the CQ arm do not leak into the "trained" condition.

usage:
  uv run scripts/start_probe.py --policy s7=runs/hd1910_v3C_s7_4000.onnx --policy s42=runs/hd1910_v3C_s42_4000.onnx
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

COMMANDS = (("fwd 0.10", (0.10, 0.0, 0.0)), ("turn 0.5", (0.0, 0.0, 0.5)))
STILL = 0.2  # stayed still: |mean response| below this fraction of the command
MEASURE_S = 4.0


def stays_still(response, command):
    return np.abs(response) < STILL * command


def run(policies, args):
    os.environ["HD1910_ARM"] = "v2d"
    import torch
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    from counterfactual_reward import batched_session

    cfg = load_env_cfg("Mjlab-Velocity-Flat-MicroDuck-HD1910")
    cfg.scene.num_envs = args.envs
    cfg.commands["twist"].resampling_time_range = (1e6, 1e6)
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    env.common_step_counter = args.stage_iter * 24
    env.curriculum_manager.compute(env_ids=torch.arange(args.envs, device=env.device))
    tw, robot = env.command_manager.get_term("twist"), env.scene["robot"]
    push = env.event_manager.get_term_cfg("push_robot").params
    push_range = dict(push["velocity_range"])
    dt = float(env.step_dt)
    n_stand, n_cmd, n_measure = (int(round(s / dt)) for s in (args.stand_s, args.cmd_s, MEASURE_S))
    out = {}
    for name, path in policies.items():
        sess = batched_session(path)
        inp = sess.get_inputs()[0].name
        out[name] = {}
        for cond in ("trained", "quiet"):
            push["velocity_range"] = {k: (0.0, 0.0) for k in push_range} if cond == "quiet" else push_range
            out[name][cond] = {}
            for cname, cmd in COMMANDS:
                torch.manual_seed(args.seed)
                rng = np.random.default_rng(args.seed)
                obs, _ = env.reset()
                vals, alive = [], np.ones(args.envs, bool)
                for k in range(n_stand + n_cmd):
                    c = (0.0, 0.0, 0.0) if k < n_stand else cmd
                    tw.vel_command_b[:] = torch.tensor(c, device=env.device)
                    tw.is_standing_env[:] = k < n_stand
                    o = obs["actor"].detach().cpu().numpy().astype(np.float32)
                    o[:, 48:51] = c
                    if cond == "quiet":
                        o[:, 51:61] = 0.0  # head + body pose commands
                    a = sess.run(None, {inp: o})[0]
                    if args.noise > 0:
                        a = a + (rng.standard_normal(a.shape) * args.noise).astype(np.float32)
                    obs, _, term, trunc, _ = env.step(torch.as_tensor(a, device=env.device))
                    alive &= ~(term | trunc).cpu().numpy()
                    if k >= n_stand + n_cmd - n_measure:
                        v = robot.data.root_link_lin_vel_b[:, 0] if cmd[0] else robot.data.root_link_ang_vel_b[:, 2]
                        vals.append(v.cpu().numpy())
                resp = np.mean(vals, 0)[alive]
                mag = max(cmd)
                out[name][cond][cname] = dict(mean=float(resp.mean()), still=float(stays_still(resp, mag).mean()),
                                              fell=int((~alive).sum()))
            print(f"{name:14s} {cond:8s} " + " | ".join(
                f"{c}: {r['mean']:+.3f}, still {r['still']:4.0%}, fell {r['fell']}" for c, r in out[name][cond].items()), flush=True)
    push["velocity_range"] = push_range
    env.close()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", action="append", required=True, metavar="NAME=ONNX")
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--stand-s", type=float, default=3.0)
    ap.add_argument("--cmd-s", type=float, default=6.0)
    ap.add_argument("--stage-iter", type=int, default=6000, help="curriculum stage to probe at (iteration)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.0,
                    help="Gaussian action noise [rad] on every joint: what training sees at that std (0 = deployed)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--json", help="write results here")
    args = ap.parse_args()
    if args.cmd_s <= MEASURE_S:
        ap.error(f"--cmd-s must exceed the {MEASURE_S} s measuring window")
    pols = dict(spec.split("=", 1) for spec in args.policy)
    out = run(pols, args)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(dict(args=vars(args), results=out), indent=1))
        print("wrote", args.json)


if __name__ == "__main__":
    main()
