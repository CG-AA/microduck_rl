#!/usr/bin/env python3
"""Render a walking policy to MP4, headless (EGL), on the same CPU MuJoCo + BAM loop as scripts/eval_walk.py.

A tracking camera follows the trunk; the command and the measured body-frame speed are drawn on each frame.

usage:
  MUJOCO_GL=egl uv run scripts/render_walk.py --walking policy.onnx --hd1910 --out walk.mp4
"""
import argparse
import importlib.util
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("eval_walk", REPO / "scripts" / "eval_walk.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

# (label, seconds, vx, vy, wz): the speeds this policy actually walks at (see NOTES.md section 9)
SCRIPT = [
    ("stand", 1.5, 0.0, 0.0, 0.0),
    ("forward 0.3 m/s", 7.0, 0.3, 0.0, 0.0),
    ("turn -1.0 rad/s", 4.0, 0.0, 0.0, -1.0),
    ("forward 0.3 m/s", 4.0, 0.3, 0.0, 0.0),
    ("stand", 2.0, 0.0, 0.0, 0.0),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--walking", required=True)
    ap.add_argument("--hd1910", action="store_true")
    ap.add_argument("--vin", type=float, default=7.4)
    ap.add_argument("--vin-drop-gain", type=float, default=0.1)
    ap.add_argument("--kp-fw", type=float, default=None)
    ap.add_argument("--out", default="walk.mp4")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--size", type=int, nargs=2, default=(960, 540))
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    bam_model, model, data, bam_ctrl, policy = ev.build(args)
    ev.reset(model, data, bam_ctrl, policy)
    w, h = args.size
    model.vis.global_.offwidth, model.vis.global_.offheight = max(w, 640), max(h, 480)
    renderer = mujoco.Renderer(model, h, w)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = model.body("trunk_base").id
    cam.distance, cam.azimuth, cam.elevation = 0.75, 135, -15
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]  # visual meshes, no collision/sites

    trunk = model.body("trunk_base").id
    dof = model.jnt_dofadr[model.joint("trunk_base_freejoint").id]
    dt = ev.DECIMATION * model.opt.timestep
    every = max(1, round(1.0 / (args.fps * dt)))  # control steps per frame (50 Hz control)
    title = args.title or ("Microduck, Feetech HD-1910 (provisional BAM model)" if args.hd1910 else "Microduck, XL330")
    writer = imageio.get_writer(args.out, fps=1.0 / (every * dt), codec="libx264", quality=8, macro_block_size=8)
    speed, t, k = np.zeros(3), 0.0, 0
    for label, seconds, vx, vy, wz in SCRIPT:
        policy.set_vel_cmd(vx, vy, wz)
        for _ in range(int(round(seconds / dt))):
            policy.apply_action(policy.infer())
            for _ in range(ev.DECIMATION):
                bam_ctrl.update()
                mujoco.mj_step(model, data)
            t += dt
            R = data.xmat[trunk].reshape(3, 3)
            v = np.array([*(R.T @ data.qvel[dof:dof + 3])[:2], data.qvel[dof + 5]])
            speed = 0.9 * speed + 0.1 * v  # ~0.2 s smoothing for display
            if k % every == 0:
                renderer.update_scene(data, cam, opt)
                img = Image.fromarray(renderer.render())
                d = ImageDraw.Draw(img)
                d.rectangle([0, 0, w, 58], fill=(0, 0, 0))
                d.text((12, 8), title, fill=(255, 255, 255))
                d.text((12, 30), f"t {t:5.1f} s   cmd: {label:16s}   measured vx {speed[0]:+.2f} m/s  "
                                 f"vy {speed[1]:+.2f} m/s  yaw {speed[2]:+.2f} rad/s", fill=(255, 220, 120))
                writer.append_data(np.asarray(img))
            k += 1
    writer.close()
    upright = data.xmat[trunk][8] > 0.7
    print(f"wrote {args.out} ({t:.1f} s); travelled {np.linalg.norm(data.xpos[trunk][:2]):.2f} m; "
          f"{'upright' if upright else 'FELL'} at the end")


if __name__ == "__main__":
    main()
