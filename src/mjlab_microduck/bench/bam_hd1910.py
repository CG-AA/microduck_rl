"""The HD-1910 as a BAM actuator, its fit on pendulum logs, and the conversion to the sim's parameter file.

Control law (the one the simulator runs, so a fit maps one-to-one onto training):

    duty = clip((q_target - q) * P * error_gain * error_gain_ratio, -max_pwm, max_pwm),   V = vin * duty
    tau  = kt * V / R - kt^2 * dq / R

P is the RAM P-gain register (50), `error_gain` the duty per radian per register unit. The blocked-rotor
test (`scripts/bench_static.py gainmap`) measures `error_gain`; until then it is a placeholder and
`error_gain_ratio` is fitted (position-only logs cannot separate it from kt/R well, so fix it once
measured). No speed slew: bam's STS3215 model has one (max_velocity), the XL330 law the sim uses does
not; the bench profile sets acceleration 0 and goal speed 0 (no firmware profile). If a fit leaves a
large error only on the fast trajectories, the firmware is slewing anyway: see docs/bench.md.

`register()` adds "hd1910" to bam's actuator registry, so `bam.model.load_model` can read a fit JSON.
The fit runs in-process with scipy's differential evolution (bam.fit needs optuna and resolves
actuators in its own process).
"""
from __future__ import annotations

import glob
import json
import math
import os
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

from bam.actuator import VoltageControlledActuator
from bam.parameter import Parameter
from bam.testbench import Pendulum

# The XL330 law the sim runs: duty/rad = kp_fw * XL330_ERROR_GAIN (bam/dynamixel/actuator.py).
XL330_ERROR_GAIN = (4096 / (2 * math.pi)) / (256 * 885)
# Duty per radian per P-register unit. UNMEASURED: the STS3215's oscilloscope value (bam/feetech) as a
# placeholder. If it holds, factory P = 32 is 5.3 duty/rad, 12x the sim's 0.449 (kp_fw 156).
ERROR_GAIN_PLACEHOLDER = 0.166
MAX_PWM = 0.98  # max_torque register 980 x 0.1 %
SIM_KP_FW = 156.0  # microduck_hd1910_constants.HD1910_KP_FW: the stiffness every HD-1910 policy trained at
FRICTION_PARAMS = ("friction_base", "friction_stribeck", "load_friction_motor", "load_friction_external",
                   "load_friction_motor_stribeck", "load_friction_external_stribeck", "load_friction_motor_quad",
                   "load_friction_external_quad", "dtheta_stribeck", "alpha", "friction_viscous")


class HD1910Actuator(VoltageControlledActuator):
    """Feetech HD-1910-C001 (HLS firmware) in position mode: proportional duty on the position error."""

    def __init__(self, testbench_class=Pendulum, error_gain: float = ERROR_GAIN_PLACEHOLDER):
        super().__init__(testbench_class, vin=7.4, kp=32, error_gain=error_gain, max_pwm=MAX_PWM, max_current=None)

    def initialize(self):
        self.model.kt = Parameter(0.62, 0.2, 1.5)  # datasheet-derived start (docs/hd1910_c001_spec.md)
        self.model.R = Parameter(3.75, 1.0, 8.0)
        self.model.armature = Parameter(0.0018, 1e-5, 0.01)
        self.model.error_gain_ratio = Parameter(1.0, 0.3, 3.0)

    def get_extra_inertia(self) -> float:
        return self.model.armature.value

    def compute_control(self, q_target, q, dq, dt):
        duty = (q_target - q) * self.kp * self.error_gain * self.model.error_gain_ratio.value
        return self.vin * self.backend.clamp(duty, -self.max_pwm, self.max_pwm)

    def duty_per_rad(self, p_register: float) -> float:
        return p_register * self.error_gain * self.model.error_gain_ratio.value


def register(error_gain: float = ERROR_GAIN_PLACEHOLDER) -> None:
    """Make "hd1910" loadable by bam.model.load_model / load_model_from_dict (same dict object bam uses)."""
    from bam.actuators import actuators
    actuators["hd1910"] = lambda: HD1910Actuator(Pendulum, error_gain)


def make_model(model: str = "m6", error_gain: float = ERROR_GAIN_PLACEHOLDER, values: dict | None = None,
               fixed: dict | None = None):
    """A BAM model with the HD-1910 actuator. `values` set starting values, `fixed` set and freeze."""
    from bam.model import models
    m = models[model]()
    m.set_actuator(HD1910Actuator(Pendulum, error_gain))
    m.actuator_name = "hd1910"
    params = m.get_parameters()
    for name, v in (values or {}).items():
        if name in params:
            params[name].value = float(v)
    for name, v in (fixed or {}).items():
        params[name].value = float(v)
        params[name].optimize = False
    return m


def model_from_json(data: dict, error_gain: float | None = None):
    """Rebuild a model from a fit dict (as written by `fit` / `params_dict`)."""
    eg = float(data.get("error_gain", ERROR_GAIN_PLACEHOLDER) if error_gain is None else error_gain)
    m = make_model(data.get("model", "m6"), eg)
    m.load_parameters_from_dict(data)
    return m


def params_dict(m, **extra) -> dict:
    out = {name: float(p.value) for name, p in m.get_parameters().items()}
    out.update(model=m.name, actuator="hd1910", error_gain=float(m.actuator.error_gain),
               max_pwm=float(m.actuator.max_pwm), **extra)
    return out


# ---- logs ------------------------------------------------------------------------------------------

def process(raw_dir, out_dir, dt: float = 0.005) -> list[str]:
    """bam.process as a function: resample every raw log onto a fixed dt grid (linear), torque_enable
    re-thresholded. Returns the written paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for path in sorted(glob.glob(f"{raw_dir}/*.json")):
        data = json.loads(Path(path).read_text())
        entries = data["entries"]
        ts = np.array([e["timestamp"] for e in entries], dtype=float)
        ts -= ts[0]
        grid = np.arange(0.0, ts[-1], dt)
        out = deepcopy(data)
        out["dt"] = dt
        keys = [k for k in entries[0] if k != "timestamp"]
        cols = {k: np.interp(grid, ts, np.array([float(e[k]) for e in entries])) for k in keys}
        out["entries"] = [
            {**{k: float(cols[k][i]) for k in keys}, "torque_enable": bool(cols["torque_enable"][i] > 0.5),
             "timestamp": float(t)}
            for i, t in enumerate(grid)
        ]
        dest = out_dir / Path(path).name
        dest.write_text(json.dumps(out))
        written.append(str(dest))
    return written


def load_logs(processed_dir):
    from bam.logs import Logs
    logs = Logs(str(processed_dir))
    if not logs.logs:
        raise FileNotFoundError(f"no processed logs in {processed_dir}")
    return logs


def rollout(m, log) -> np.ndarray:
    """Simulated positions for one log or a batch (bam.simulate, firmware law simulated)."""
    from bam.simulate import Simulator
    pos, _, _ = Simulator(m).rollout_log(log, simulate_control=True)
    return np.array(pos)


def score(m, batch) -> float:
    """bam.fit's loss: mean |simulated - logged position| [rad]."""
    log_pos = np.array([e["position"] for e in batch["entries"]])
    return float(np.mean(np.abs(rollout(m, batch) - log_pos)))


class _Objective:
    """Picklable objective for scipy workers: rebuilds its model once per process."""

    def __init__(self, model_name, error_gain, start, fixed, names, batch):
        self.args = (model_name, error_gain, start, fixed)
        self.names, self.batch, self._m = names, batch, None

    def __call__(self, x):
        if self._m is None:
            name, eg, start, fixed = self.args
            self._m = make_model(name, eg, values=start, fixed=fixed)
        params = self._m.get_parameters()
        for n, v in zip(self.names, x):
            params[n].value = float(v)
        loss = score(self._m, self.batch)
        return loss if math.isfinite(loss) else 1e3


def fit(processed_dir, *, model: str = "m6", error_gain: float = ERROR_GAIN_PLACEHOLDER, fixed: dict | None = None,
        start: dict | None = None, evals: int = 3000, seed: int = 0, workers: int = 1, validation_kp=None,
        log=print) -> dict:
    """Fit the free parameters on every processed log (minus the `validation_kp` ones) and return a
    params dict (bam's JSON layout + error_gain + a `_fit` report). `evals` bounds the objective calls."""
    from scipy.optimize import differential_evolution

    logs = load_logs(processed_dir)
    val = logs.split(validation_kp) if validation_kp is not None else None
    if not logs.logs:
        raise ValueError("every log was held out for validation")
    batch = logs.make_batch()
    m = make_model(model, error_gain, values=start, fixed=fixed)
    params = m.get_parameters()
    names = [n for n, p in params.items() if p.optimize]
    bounds = [(params[n].min, params[n].max) for n in names]
    x0 = np.clip([params[n].value for n in names], [b[0] for b in bounds], [b[1] for b in bounds])
    popsize = 10
    maxiter = max(1, evals // (popsize * len(names)) - 1)
    obj = _Objective(model, error_gain, start, fixed, names, batch)
    t0 = time.time()
    loss0 = obj(x0)
    log(f"fitting {len(names)} parameters on {len(logs.logs)} logs: start loss {loss0:.5f} rad, "
        f"{maxiter} generations x {popsize * len(names)}")
    res = differential_evolution(obj, bounds, x0=x0, seed=seed, popsize=popsize, maxiter=maxiter, tol=1e-8,
                                 polish=False, init="latinhypercube", workers=workers,
                                 updating="deferred" if workers != 1 else "immediate")
    for n, v in zip(names, res.x):
        params[n].value = float(v)
    report = dict(loss=float(res.fun), loss_start=float(loss0), evals=int(res.nfev), seconds=round(time.time() - t0, 1),
                  logs=[os.path.basename(f) for f in logs.json_files], free=names, fixed=dict(fixed or {}), seed=seed)
    if val is not None and val.logs:
        report["validation_kp"] = validation_kp
        report["validation_loss"] = score(m, val.make_batch())
    log(f"fit: loss {report['loss']:.5f} rad after {report['evals']} evals in {report['seconds']} s"
        + (f", validation {report['validation_loss']:.5f}" if "validation_loss" in report else ""))
    return params_dict(m, _fit=report)


# ---- to the simulator ------------------------------------------------------------------------------

def to_xl330_law(fit_params: dict, p_register: float) -> tuple[dict, float]:
    """The sim's params JSON (bam "xl330" actuator + m6 friction) and the kp_fw that reproduces the fitted
    duty/rad at `p_register`: kp_fw * XL330_ERROR_GAIN == P * error_gain * error_gain_ratio. The sim
    applies the HD-1910 firmware limits itself (max_pwm 0.98, no current clamp)."""
    duty_per_rad = p_register * fit_params["error_gain"] * fit_params.get("error_gain_ratio", 1.0)
    kp_fw = duty_per_rad / XL330_ERROR_GAIN
    sim = {k: float(fit_params[k]) for k in ("kt", "R", "armature")}
    sim["q_offset"] = 0.0  # a per-servo mounting offset, not a property of the servo model
    sim.update({k: float(fit_params[k]) for k in FRICTION_PARAMS if k in fit_params})
    sim.update(model="m6", actuator="xl330",
               _note=(f"HD-1910 bench fit converted to the XL330 law: use kp_fw {kp_fw:.1f} with P register "
                      f"{p_register:g} ({duty_per_rad:.3f} duty/rad); the sim adds max_pwm 0.98, no current clamp."),
               kp_fw_equivalent=kp_fw, p_register=p_register, _hd1910_fit=fit_params)
    return sim, kp_fw


def p_register_for_sim(fit_params: dict, kp_fw: float = SIM_KP_FW) -> tuple[float, int]:
    """The P register that reproduces a trained stiffness: (exact, nearest integer >= 1)."""
    per_unit = fit_params["error_gain"] * fit_params.get("error_gain_ratio", 1.0)
    exact = kp_fw * XL330_ERROR_GAIN / per_unit
    return exact, max(1, int(round(exact)))
