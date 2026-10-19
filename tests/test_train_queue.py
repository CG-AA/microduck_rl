"""scripts/train_queue.py: checkpoint discovery across relaunch directories and the resume command."""
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("train_queue", REPO / "scripts" / "train_queue.py")
tq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tq)


def test_latest_checkpoint_spans_relaunch_directories(tmp_path):
    exp = tmp_path / "velocity_hd1910"
    for d, its in (("2026-10-19_06-30-10_hd1910_v3C_s42", (2500, 3000, 3250)), ("2026-10-19_08-00-00_hd1910_v3C_s42", (3250, 3300, 3350)),
                   ("2026-10-19_08-00-00_hd1910_v3C_s7", (9999,))):
        (exp / d).mkdir(parents=True)
        for it in its:
            (exp / d / f"model_{it}.pt").write_bytes(b"")
    assert tq.latest_checkpoint("hd1910_v3C_s42", log_root=tmp_path) == ("2026-10-19_08-00-00_hd1910_v3C_s42", 3350)
    init = {"load_run": "2026-10-18_10-57-43_hd1910_v2d_s7", "checkpoint": "model_2500.pt"}
    assert tq.latest_checkpoint("hd1910_v3C_s99", log_root=tmp_path, init=init) == ("2026-10-18_10-57-43_hd1910_v2d_s7", 2500)
    assert tq.latest_checkpoint("hd1910_v3C_s99", log_root=tmp_path) is None


def test_train_command_lands_on_the_target_iteration():
    job = {"task": "Mjlab-Velocity-Flat-MicroDuck-HD1910", "seed": 7, "run_name": "hd1910_v3C_s7", "target_iter": 4000}
    cmd = tq.train_command(job, "2026-10-18_10-57-43_hd1910_v2d_s7", 2500, 50)
    i = cmd.index("--agent.max-iterations")
    assert cmd[i + 1] == "1501"  # rsl_rl saves the last as model_{2500 + 1501 - 1} = model_4000
    assert cmd[cmd.index("--agent.load-checkpoint") + 1] == "model_2500.pt"
    assert cmd[cmd.index("--agent.save-interval") + 1] == "50" and cmd[cmd.index("--agent.resume") + 1] == "True"


def test_mem_available_reads_meminfo(tmp_path):
    f = tmp_path / "meminfo"
    f.write_text("MemTotal:       14680064 kB\nMemFree:  1000 kB\nMemAvailable:    7340032 kB\n")
    assert tq.mem_available_mb(str(f)) == 7168
