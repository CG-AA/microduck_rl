"""Memory-aware training queue: run training jobs when the machine has room, stop them first when it doesn't.

This machine is shared with other heavy work (14 GB RAM; a training run needs about 4 GB). The supervisor
runs a list of jobs one at a time and, for each:

  * waits until MemAvailable >= --start-mb has held for --hold-s seconds before launching (after a memory
    stop: --restart-hold-s, doubling while stops keep coming, up to 30 min). Training itself takes ~4 GB,
    so --start-mb must exceed --stop-mb + 4000 + a margin or a launch would trip its own stop;
  * launches `uv run train ... --agent.resume` from the job's newest checkpoint (the starting checkpoint on
    the first launch), saving every --save-interval iterations so a stop loses little;
  * marks the whole training process tree first in line for the kernel's out-of-memory killer
    (oom_score_adj 1000), so if memory runs out anyway, training dies before anything else;
  * polls MemAvailable every --poll-s seconds and, below --stop-mb, stops training itself
    (SIGINT to the process group, SIGTERM after 20 s, SIGKILL after 10 more), then waits for memory again
    and resumes from the newest checkpoint;
  * marks a job done once a checkpoint at or past its target iteration exists, and gives up on a job after
    --max-failures crashes that were not its own stops.

Jobs are a JSON list of {"name", "task", "run_name", "seed", "target_iter", "init": {"load_run",
"checkpoint"}, "env": {...}, "args": [...]}. State and logs go next to the job file (<jobs>.state.json,
<jobs>.<name>.log, <jobs>.supervisor.log). Create the file <jobs>.STOP to make the supervisor stop the
current job and exit. Pure standard library; meant to be started detached:

  setsid nohup .venv/bin/python scripts/train_queue.py runs/queue/iter3.json > runs/queue/iter3.out 2>&1 &
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOG_ROOT = REPO / "logs" / "rsl_rl"


def mem_available_mb(meminfo="/proc/meminfo"):
    with open(meminfo) as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    raise RuntimeError("MemAvailable not found")


def latest_checkpoint(run_name, log_root=LOG_ROOT, init=None):
    """(run_dir_name, iteration) of the newest model_N.pt across every directory of this run name
    (each relaunch writes a new timestamped directory), else the job's starting checkpoint."""
    best = None
    for d in Path(log_root).glob(f"*/*_{run_name}"):
        for f in d.glob("model_*.pt"):
            m = re.fullmatch(r"model_(\d+)\.pt", f.name)
            if m and (best is None or int(m.group(1)) > best[1]):
                best = (d.name, int(m.group(1)))
    if best is None and init:
        m = re.fullmatch(r"model_(\d+)\.pt", init["checkpoint"])
        best = (init["load_run"], int(m.group(1)))
    return best


def train_command(job, load_run, it, save_interval):
    """rsl_rl trains `max_iterations` iterations from the loaded one and saves the last as
    model_{start + max_iterations - 1}, so this lands exactly on target_iter."""
    n = job["target_iter"] - it + 1
    return ["uv", "run", "train", job["task"], "--env.scene.num-envs", str(job.get("num_envs", 4096)),
            "--agent.logger", "tensorboard", "--agent.seed", str(job["seed"]), "--agent.run-name", job["run_name"],
            "--agent.resume", "True", "--agent.load-run", load_run, "--agent.load-checkpoint", f"model_{it}.pt",
            "--agent.max-iterations", str(n), "--agent.save-interval", str(save_interval), *job.get("args", [])]


def descendants(pid):
    """pid and every process below it (via /proc/<pid>/task/*/children)."""
    out, todo = [], [pid]
    while todo:
        p = todo.pop()
        out.append(p)
        for t in Path(f"/proc/{p}/task").glob("*"):
            try:
                todo += [int(c) for c in (t / "children").read_text().split()]
            except OSError:
                pass
    return out


def mark_oom_first(pid):
    for p in descendants(pid):
        try:
            Path(f"/proc/{p}/oom_score_adj").write_text("1000")
        except OSError:
            pass


class Supervisor:
    def __init__(self, jobs_path, a):
        self.path = Path(jobs_path)
        self.jobs = json.loads(self.path.read_text())
        self.a = a
        self.state_path = self.path.with_suffix(".state.json")
        self.stop_path = self.path.with_suffix(".STOP")
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        self.log_path = self.path.with_suffix(".supervisor.log")

    def log(self, msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")

    def save(self, name, **kw):
        self.state.setdefault(name, {}).update(kw, updated=time.strftime("%Y-%m-%d %H:%M:%S"))
        self.state_path.write_text(json.dumps(self.state, indent=1))

    def stopping(self):
        return self.stop_path.exists()

    def wait_for_memory(self, name, hold_s):
        since, last_note = None, 0.0
        while not self.stopping():
            mb = mem_available_mb()
            if mb >= self.a.start_mb:
                since = since or time.time()
                if time.time() - since >= hold_s:
                    return True
            else:
                since = None
                if time.time() - last_note > 300:
                    self.log(f"{name}: waiting for memory ({mb} MB available, need {self.a.start_mb} for {hold_s:.0f} s)")
                    self.save(name, status="waiting for memory", mem_mb=mb)
                    last_note = time.time()
            time.sleep(self.a.poll_s)
        return False

    def stop(self, proc, why):
        self.log(f"stopping training pid {proc.pid}: {why}")
        for sig, wait in ((signal.SIGINT, 20), (signal.SIGTERM, 10), (signal.SIGKILL, 5)):
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                break
            try:
                proc.wait(timeout=wait)
                break
            except subprocess.TimeoutExpired:
                continue

    def run_job(self, job):
        name, failures, hold, mem_stops = job["name"], 0, self.a.hold_s, []
        while not self.stopping():
            ckpt = latest_checkpoint(job["run_name"], init=job.get("init"))
            if ckpt is None:
                self.log(f"{name}: no checkpoint to start from")
                self.save(name, status="failed: no starting checkpoint")
                return
            run_dir, it = ckpt
            if it >= job["target_iter"]:
                self.log(f"{name}: done (model_{it}.pt in {run_dir})")
                self.save(name, status="done", last_ckpt=f"{run_dir}/model_{it}.pt")
                return
            if not self.wait_for_memory(name, hold):
                return
            cmd = train_command(job, run_dir, it, self.a.save_interval)
            env = {**os.environ, **{k: str(v) for k, v in job.get("env", {}).items()}}
            joblog = self.path.with_suffix(f".{name}.log")
            self.log(f"{name}: launching from {run_dir}/model_{it}.pt to {job['target_iter']} ({mem_available_mb()} MB available)")
            launches = self.state.get(name, {}).get("launches", 0) + 1
            self.save(name, status="running", from_ckpt=f"{run_dir}/model_{it}.pt", launches=launches)
            with open(joblog, "a") as lf:
                lf.write(f"\n===== launch {launches} {time.strftime('%Y-%m-%d %H:%M:%S')}: {' '.join(cmd)}\n")
                lf.flush()
                proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
            stopped_by_us = False
            while proc.poll() is None:
                mark_oom_first(proc.pid)
                if self.stopping():
                    self.stop(proc, "stop file present")
                    stopped_by_us = True
                    break
                mb = mem_available_mb()
                if mb < self.a.stop_mb:
                    self.stop(proc, f"memory low ({mb} MB available < {self.a.stop_mb})")
                    stopped_by_us = True
                    self.save(name, status="paused (memory)", mem_mb=mb)
                    break
                time.sleep(self.a.poll_s)
            rc = proc.wait()
            if stopped_by_us:
                if self.stopping():
                    return
                now = time.time()
                mem_stops = [t for t in mem_stops if now - t < 3600] + [now]
                hold = min(1800.0, self.a.restart_hold_s * 2 ** (len(mem_stops) - 1))
                self.save(name, memory_stops=self.state.get(name, {}).get("memory_stops", 0) + 1, next_hold_s=hold)
                continue
            if rc != 0:
                failures += 1
                self.log(f"{name}: training exited with {rc} (failure {failures}/{self.a.max_failures})")
                self.save(name, status=f"crashed ({rc})", failures=failures)
                if failures >= self.a.max_failures:
                    self.save(name, status=f"failed after {failures} crashes")
                    return
                time.sleep(30)

    def run(self):
        self.log(f"supervisor pid {os.getpid()} started: {len(self.jobs)} jobs, start >= {self.a.start_mb} MB for {self.a.hold_s} s, stop < {self.a.stop_mb} MB")
        self.state["_supervisor"] = {"status": "running", "pid": os.getpid(), "start_mb": self.a.start_mb, "stop_mb": self.a.stop_mb,
                                     "at": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.state_path.write_text(json.dumps(self.state, indent=1))
        for job in self.jobs:
            if self.stopping():
                break
            if self.state.get(job["name"], {}).get("status") == "done":
                continue
            self.run_job(job)
        self.log("supervisor exiting" + (" (stop file)" if self.stopping() else ": queue finished"))
        self.state["_supervisor"] = {"status": "exited", "at": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.state_path.write_text(json.dumps(self.state, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jobs")
    ap.add_argument("--start-mb", type=int, default=8000)
    ap.add_argument("--stop-mb", type=int, default=2500)
    ap.add_argument("--hold-s", type=float, default=60.0)
    ap.add_argument("--restart-hold-s", type=float, default=300.0)
    ap.add_argument("--poll-s", type=float, default=1.0)
    ap.add_argument("--save-interval", type=int, default=50)
    ap.add_argument("--max-failures", type=int, default=3)
    args = ap.parse_args()
    if args.start_mb < args.stop_mb + 4000:
        ap.error(f"--start-mb {args.start_mb} must be at least --stop-mb + 4000 ({args.stop_mb + 4000}): training takes ~4 GB")
    Supervisor(args.jobs, args).run()


if __name__ == "__main__":
    sys.exit(main())
