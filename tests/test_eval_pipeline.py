"""scripts/eval_pipeline.py: candidate choice and the final-bar verdict from per-checkpoint results."""
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("eval_pipeline", REPO / "scripts" / "eval_pipeline.py")
ep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ep)


def _item(fails=(), worst=0.0, fwd20=0.20, fwd30=0.28, torque=0.05, ss=0.15):
    gates = {"fwd 0.20 vx [m/s]": (fwd20, "pass"), "fwd 0.30 vx [m/s]": (fwd30, "pass"),
             "stand torque max joint [N.m]": (torque, "pass")}
    return dict(gates=gates, fails=list(fails), probe_worst=("back", worst), ss_sum=ss)


def test_candidate_needs_every_item_then_the_smallest_step_error():
    items = {5000: _item(ss=0.10, worst=0.30),          # probe fails
             5500: _item(ss=0.16), 5750: _item(ss=0.12),
             6000: _item(ss=0.11, fwd30=0.24)}          # too slow
    head = {5500: 0.97, 5750: 0.94, 6000: 0.99}         # 5750 misses head tracking
    assert ep.choose_candidate(items, head) == 5500
    assert ep.choose_candidate(items, {}) is None
    assert not ep.qualifies(_item(fwd20=None))          # a missing number fails


def test_final_verdict_counts_neighbours_and_needs_the_extras():
    final = [5000, 5500, 5750, 6000]
    items = {5000: _item(), 5500: _item(), 5750: _item(fails=["falls"]), 6000: _item(worst=0.2)}
    assert ep.final_verdict(items, final, 5500, {"sens_fails": 2, "contract": True}) == ("PASS", [])
    v, why = ep.final_verdict(items, final, 5500, {"sens_fails": 5, "contract": True})
    assert v == "FAIL" and why == ["sensitivity fails 5"]
    items[5000] = _item(fails=["fwd 0.10 vx"])
    v, why = ep.final_verdict(items, final, 5500, {"sens_fails": 2, "contract": True})
    assert v == "FAIL" and why == ["gated rows pass at 2 of 4"]


def test_candidates_are_ranked_by_step_error():
    items = {5000: _item(ss=0.10), 5500: _item(ss=0.12), 5750: _item(ss=0.08), 6000: _item(ss=0.06, worst=0.5)}
    assert ep.rank_candidates(items, {5000: 0.97, 5500: 0.97, 5750: 0.97, 6000: 0.99}) == [5750, 5000, 5500]
