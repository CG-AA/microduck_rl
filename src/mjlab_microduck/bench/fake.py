"""A simulated HD-1910 on a pendulum, behind the HLS register map, for hardware-free bench runs and tests.

`FakeHlsController` answers rustypot's four raw calls from a register memory and a physics model: BAM's
pendulum simulator driving a planted `HD1910Actuator` model (the "truth"), so a recording made through
`HlsBus` on the fake and fitted back must recover the planted servo. What it models: position mode with
the P register (RAM 50) through `error_gain`, torque off / on / damping, the encoder quantised to 4096
counts and the speed register to 0.732 rpm, sign-magnitude registers, a blocked rotor, a first-order
temperature rise, and a fixed latency per bus transaction on a virtual clock (`FakeClock`, so nothing
sleeps). What it does not: the D and I gains, acceleration profiles, overcurrent unloading.
"""
from __future__ import annotations

import numpy as np

from mjlab_microduck.bench import hls
from mjlab_microduck.bench.bam_hd1910 import ERROR_GAIN_PLACEHOLDER, make_model

DEFAULT_TRUTH = dict(kt=0.70, R=4.2, armature=0.0012, error_gain_ratio=1.0, friction_base=0.008,
                     friction_stribeck=0.006, load_friction_motor=0.15, load_friction_external=0.02,
                     load_friction_motor_stribeck=0.02, load_friction_external_stribeck=0.05,
                     load_friction_motor_quad=0.004, load_friction_external_quad=0.002, dtheta_stribeck=1.5,
                     alpha=2.0, friction_viscous=0.01, q_offset=0.0)


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t

    def sleep(self, s):
        if s > 0:
            self.t += s


class FakeHlsController:
    def __init__(self, clock: FakeClock, ids=(1,), truth: dict | None = None, error_gain=ERROR_GAIN_PLACEHOLDER,
                 mass=0.05, length=0.08, arm_mass=0.01, vin=7.4, theta0=0.3, blocked=False,
                 bus_latency=0.0003, substep=0.001, heat_k=0.7, heat_tau=300.0):
        from bam.simulate import Simulator
        from bam.testbench import Pendulum
        self.clock, self.ids = clock, list(ids)
        self.truth = {**DEFAULT_TRUTH, **(truth or {})}
        self.model = make_model("m6", error_gain, values=self.truth)
        self.model.actuator.testbench = Pendulum(dict(mass=mass, length=length, arm_mass=arm_mass))
        self.model.actuator.vin = vin
        self.sim = Simulator(self.model)
        self.vin, self.theta0, self.blocked = vin, theta0, blocked
        self.bus_latency, self.substep = bus_latency, substep
        self.heat_k, self.heat_tau, self.temp = heat_k, heat_tau, 25.0
        self.sim_t, self.duty, self.current = clock.monotonic(), 0.0, 0.0
        self.writes = []  # (sid, addr, data), every write in order
        self.mem = {sid: self._factory(sid) for sid in self.ids}
        self._refresh(self.ids[0])

    @staticmethod
    def _factory(sid):
        m = bytearray(128)
        defaults = dict(model=hls.MODEL_NUMBER, id=sid, baud=0, min_angle=0, max_angle=4095, max_temperature=70,
                        min_voltage=40, max_torque=980, p_gain_eeprom=32, d_gain_eeprom=40, i_gain_eeprom=0,
                        cw_dead=1, ccw_dead=1, protection_current=310, mode=0, overcurrent_time=200,
                        torque_limit=980, p_gain=32, d_gain=40, i_gain=0, lock=1, goal_position=hls.CENTER)
        for name, v in defaults.items():
            r = hls.REG[name]
            m[r.addr:r.addr + r.size] = bytes(hls.le_bytes(hls.from_signed(v, r.sign_bit), r.size))
        return m

    def _reg(self, sid, name):
        r = hls.REG[name]
        return hls.to_signed(hls.le_int(self.mem[sid][r.addr:r.addr + r.size]), r.sign_bit)

    def _put(self, sid, name, value):
        r = hls.REG[name]
        self.mem[sid][r.addr:r.addr + r.size] = bytes(hls.le_bytes(hls.from_signed(int(value), r.sign_bit), r.size))

    # -- physics (the first id is the servo on the pendulum) -------------------------------------
    def _advance(self):
        sid = self.ids[0]
        act = self.model.actuator
        while self.sim_t + self.substep <= self.clock.monotonic() + 1e-12:
            te = self._reg(sid, "torque_enable")
            act.kp = self._reg(sid, "p_gain")
            goal = hls.count_to_rad(self._reg(sid, "goal_position")) - self.theta0
            q, dq = float(self.sim.q), float(self.sim.dq)
            control = act.compute_control(goal, q, dq, self.substep) if te == hls.TORQUE_ON else 0.0
            powered = te in (hls.TORQUE_ON, hls.TORQUE_DAMPING)
            if self.blocked:
                self.sim.dq = 0.0
            else:
                self.sim.step(control, powered, self.substep)
            dq_now = 0.0 if self.blocked else float(self.sim.dq)
            self.duty = control / self.vin
            self.current = (control - self.model.kt.value * dq_now) / self.model.R.value if powered else 0.0
            self.temp += self.substep * (self.heat_k * self.current ** 2 - (self.temp - 25.0) / self.heat_tau)
            self.sim_t += self.substep
        self._refresh(sid)

    def _refresh(self, sid):
        q, dq = float(self.sim.q), 0.0 if self.blocked else float(self.sim.dq)
        self._put(sid, "present_position", hls.rad_to_count(q + self.theta0))
        self._put(sid, "present_speed", int(round(dq / hls.RAD_S_PER_SPEED_COUNT)))
        self._put(sid, "present_load", int(round(np.clip(self.duty, -1, 1) / hls.DUTY_PER_LOAD_COUNT)))
        self._put(sid, "present_voltage", int(round(self.vin / hls.VOLT_PER_COUNT)))
        self._put(sid, "present_temperature", int(round(self.temp)))
        self._put(sid, "present_current", int(round(self.current / hls.AMP_PER_CURRENT_COUNT)))
        self._put(sid, "moving", int(abs(dq) > 0.05))

    def _transaction(self, sid):
        if sid not in self.mem:
            raise RuntimeError(f"timeout: no servo with id {sid}")
        self.clock.sleep(self.bus_latency)
        self._advance()

    # -- rustypot raw API ------------------------------------------------------------------------
    def read_raw_data(self, sid, addr, length):
        self._transaction(sid)
        return list(self.mem[sid][addr:addr + length])

    def write_raw_data(self, sid, addr, data):
        self._transaction(sid)
        self.writes.append((sid, addr, list(data)))
        self.mem[sid][addr:addr + len(data)] = bytes(data)

    def sync_read_raw_data(self, ids, addr, length):
        out = []
        for sid in ids:
            self._transaction(sid)
            out.append(list(self.mem[sid][addr:addr + length]))
        return out

    def sync_write_raw_data(self, ids, addr, data):
        self.clock.sleep(self.bus_latency)
        self._advance()
        for sid, d in zip(ids, data):
            if sid in self.mem:
                self.writes.append((sid, addr, list(d)))
                self.mem[sid][addr:addr + len(d)] = bytes(d)

    # -- test helpers ----------------------------------------------------------------------------
    def eeprom_writes_while_locked(self):
        """Writes below address 40 made while the lock register was 1 (a real servo would not keep them)."""
        bad, locked = [], {sid: True for sid in self.mem}
        for sid, addr, data in self.writes:
            if addr == hls.REG["lock"].addr:
                locked[sid] = bool(data[0])
            elif addr < 40 and locked[sid]:
                bad.append((sid, addr, data))
        return bad
