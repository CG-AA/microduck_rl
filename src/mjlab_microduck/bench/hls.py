"""Feetech HLS-protocol bus access for the HD-1910-C001, over any controller with rustypot's raw calls.

The HLS frame (FF FF ID LEN INST ..., checksum ~sum) is the one rustypot's `Sts3215PyController` speaks,
and most register addresses match the STS3215's. The differences are why this module never uses the
typed STS methods (docs/hd1910_c001_spec.md):

- address 44 is GOAL CURRENT on HLS (goal time on STS): nothing here writes it unless asked by name;
- the live position gains are RAM 50/51/52 (P/D/I); EEPROM 21/22/23 are only copied there at power-up;
- the present-load sign is BIT10 (position, speed and current use BIT15), sign-magnitude, not two's
  complement;
- torque_enable 2 = damping (motor shorted), mode 0/1/2/3 = position / speed / current / PWM.

`HlsBus` needs only `read_raw_data(id, addr, n) -> list[int]`, `write_raw_data(id, addr, list[int])`,
`sync_read_raw_data(ids, addr, n) -> list[list[int]]` and `sync_write_raw_data(ids, addr, list[list[int]])`,
which rustypot 1.5's controllers provide (failures raise RuntimeError) and `bench.fake.FakeHlsController`
mimics. Open a real bus with `open_bus(port)`.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

COUNTS_PER_REV = 4096
CENTER = 2048
RAD_PER_COUNT = 2 * math.pi / COUNTS_PER_REV
RAD_S_PER_SPEED_COUNT = 0.732 * 2 * math.pi / 60  # 0.732 rpm per count
AMP_PER_CURRENT_COUNT = 0.0065
DUTY_PER_LOAD_COUNT = 0.001  # 0.1 % of full duty
VOLT_PER_COUNT = 0.1
BAUD_1M = 1_000_000
FEEDBACK_ADDR, FEEDBACK_LEN = 56, 15
MODEL_NUMBER = 0x1F0A  # read back from real HD-1910-C001 units (fw 3.46)


@dataclass(frozen=True)
class Reg:
    addr: int
    size: int
    area: str  # "eeprom" (unlock 55 to write), "ram", "ro"
    sign_bit: int | None = None  # sign-magnitude sign bit, None = unsigned


REG = {
    "model": Reg(3, 2, "ro"),
    "id": Reg(5, 1, "eeprom"),
    "baud": Reg(6, 1, "eeprom"),
    "min_angle": Reg(9, 2, "eeprom"),
    "max_angle": Reg(11, 2, "eeprom"),
    "max_temperature": Reg(13, 1, "eeprom"),
    "min_voltage": Reg(15, 1, "eeprom"),
    "max_torque": Reg(16, 2, "eeprom"),
    "p_gain_eeprom": Reg(21, 1, "eeprom"),
    "d_gain_eeprom": Reg(22, 1, "eeprom"),
    "i_gain_eeprom": Reg(23, 1, "eeprom"),
    "cw_dead": Reg(26, 1, "eeprom"),
    "ccw_dead": Reg(27, 1, "eeprom"),
    "protection_current": Reg(28, 2, "eeprom"),
    "offset": Reg(31, 2, "eeprom", 15),
    "mode": Reg(33, 1, "eeprom"),
    "overcurrent_time": Reg(38, 1, "eeprom"),
    "torque_enable": Reg(40, 1, "ram"),
    "acceleration": Reg(41, 1, "ram"),
    "goal_position": Reg(42, 2, "ram", 15),
    "goal_current": Reg(44, 2, "ram", 15),
    "goal_speed": Reg(46, 2, "ram", 15),
    "torque_limit": Reg(48, 2, "ram"),
    "p_gain": Reg(50, 1, "ram"),
    "d_gain": Reg(51, 1, "ram"),
    "i_gain": Reg(52, 1, "ram"),
    "lock": Reg(55, 1, "ram"),
    "present_position": Reg(56, 2, "ro", 15),
    "present_speed": Reg(58, 2, "ro", 15),
    "present_load": Reg(60, 2, "ro", 10),
    "present_voltage": Reg(62, 1, "ro"),
    "present_temperature": Reg(63, 1, "ro"),
    "status": Reg(65, 1, "ro"),
    "moving": Reg(66, 1, "ro"),
    "present_current": Reg(69, 2, "ro", 15),
}
STATUS_BITS = {0: "voltage", 1: "encoder", 2: "temperature", 3: "current", 5: "overload"}
TORQUE_OFF, TORQUE_ON, TORQUE_DAMPING = 0, 1, 2
MODE_POSITION, MODE_SPEED, MODE_CURRENT, MODE_PWM = 0, 1, 2, 3


def to_signed(raw: int, sign_bit: int | None) -> int:
    """Feetech sign-magnitude -> int."""
    if sign_bit is None:
        return raw
    mag = raw & ((1 << sign_bit) - 1)
    return -mag if raw & (1 << sign_bit) else mag


def from_signed(value: int, sign_bit: int | None) -> int:
    """int -> Feetech sign-magnitude."""
    if sign_bit is None:
        if value < 0:
            raise ValueError(f"negative value {value} for an unsigned register")
        return value
    if abs(value) >= (1 << sign_bit):
        raise ValueError(f"|{value}| does not fit below sign bit {sign_bit}")
    return (-value) | (1 << sign_bit) if value < 0 else value


def le_bytes(value: int, size: int) -> list[int]:
    return [(value >> (8 * i)) & 0xFF for i in range(size)]


def le_int(data) -> int:
    return sum(int(b) << (8 * i) for i, b in enumerate(data))


def rad_to_count(rad: float) -> int:
    return int(round(CENTER + rad / RAD_PER_COUNT))


def count_to_rad(count: int) -> float:
    return (count - CENTER) * RAD_PER_COUNT


@dataclass
class Feedback:
    """One decoded 15-byte feedback block (addresses 56-70)."""
    pos_rad: float  # encoder angle, 0 = center count 2048
    vel_rad_s: float
    load_duty: float  # signed PWM duty, -1..1
    volts: float
    temp_c: int
    status: int
    moving: int
    current_a: float  # signed
    t: float | None = None  # host timestamp of the read

    def faults(self) -> list[str]:
        return [name for bit, name in STATUS_BITS.items() if self.status & (1 << bit)]

    def as_dict(self):
        return asdict(self)


def decode_feedback(block, t: float | None = None) -> Feedback:
    """Decode `read_raw_data(id, 56, 15)`: offsets 0 pos, 2 speed, 4 load, 6 volt, 7 temp, 9 status,
    10 moving, 13 current (8, 11, 12 unused)."""
    if len(block) != FEEDBACK_LEN:
        raise ValueError(f"feedback block has {len(block)} bytes, expected {FEEDBACK_LEN}")
    pos = to_signed(le_int(block[0:2]), 15)
    return Feedback(
        pos_rad=count_to_rad(pos),
        vel_rad_s=to_signed(le_int(block[2:4]), 15) * RAD_S_PER_SPEED_COUNT,
        load_duty=to_signed(le_int(block[4:6]), 10) * DUTY_PER_LOAD_COUNT,
        volts=block[6] * VOLT_PER_COUNT,
        temp_c=int(block[7]),
        status=int(block[9]),
        moving=int(block[10]),
        current_a=to_signed(le_int(block[13:15]), 15) * AMP_PER_CURRENT_COUNT,
        t=t,
    )


class HlsBus:
    """Register-level access to one or more HD-1910 servos.

    Reads return decoded ints (sign-magnitude handled); EEPROM writes unlock (55 = 0), write and
    re-lock (55 = 1). Gains are written to RAM, where the firmware reads them; they revert to the
    EEPROM copies at power-up, so a runtime must write them after every boot.
    """

    def __init__(self, ctrl, ids, clock=None):
        self.ctrl = ctrl
        self.ids = list(ids)
        self.clock = clock  # object with monotonic(); stamps feedback

    # -- raw registers ---------------------------------------------------------------------------
    def read_reg(self, sid: int, name: str) -> int:
        r = REG[name]
        return to_signed(le_int(self.ctrl.read_raw_data(sid, r.addr, r.size)), r.sign_bit)

    def write_reg(self, sid: int, name: str, value: int) -> None:
        r = REG[name]
        if r.area == "ro":
            raise ValueError(f"{name} is read-only")
        if r.area == "eeprom":
            raise ValueError(f"{name} is EEPROM: use write_eeprom() (unlock, write, lock)")
        self.ctrl.write_raw_data(sid, r.addr, le_bytes(from_signed(int(value), r.sign_bit), r.size))

    def write_eeprom(self, sid: int, values: dict) -> None:
        self.ctrl.write_raw_data(sid, REG["lock"].addr, [0])
        try:
            for name, value in values.items():
                r = REG[name]
                if r.area != "eeprom":
                    raise ValueError(f"{name} is not an EEPROM register")
                self.ctrl.write_raw_data(sid, r.addr, le_bytes(from_signed(int(value), r.sign_bit), r.size))
        finally:
            self.ctrl.write_raw_data(sid, REG["lock"].addr, [1])

    def read_table(self, sid: int) -> dict:
        """Every named register (one read per register; for inspection, not for loops)."""
        return {name: self.read_reg(sid, name) for name in REG}

    # -- control ---------------------------------------------------------------------------------
    def ping(self, sid: int) -> bool:
        try:
            return self.read_reg(sid, "model") >= 0
        except RuntimeError:
            return False

    def torque(self, sid: int, mode: int) -> None:
        if mode not in (TORQUE_OFF, TORQUE_ON, TORQUE_DAMPING):
            raise ValueError(f"torque mode {mode}: 0 off, 1 on, 2 damping")
        self.write_reg(sid, "torque_enable", mode)

    def set_gains(self, sid: int, p: int, d: int = 0, i: int = 0) -> None:
        """Live position-loop gains (RAM 50/51/52)."""
        for name, v in (("p_gain", p), ("d_gain", d), ("i_gain", i)):
            if not 0 <= int(v) <= 255:
                raise ValueError(f"{name} {v} outside 0..255")
            self.write_reg(sid, name, int(v))

    def goal(self, sid: int, rad: float) -> None:
        self.write_reg(sid, "goal_position", rad_to_count(rad))

    def sync_goal(self, goals: dict) -> None:
        ids = list(goals)
        data = [le_bytes(from_signed(rad_to_count(goals[i]), 15), 2) for i in ids]
        self.ctrl.sync_write_raw_data(ids, REG["goal_position"].addr, data)

    def _now(self):
        return self.clock.monotonic() if self.clock is not None else None

    def feedback(self, sid: int) -> Feedback:
        t0 = self._now()
        block = self.ctrl.read_raw_data(sid, FEEDBACK_ADDR, FEEDBACK_LEN)
        t1 = self._now()
        return decode_feedback(block, None if t0 is None else 0.5 * (t0 + t1))

    def sync_feedback(self, ids=None) -> dict:
        ids = list(ids or self.ids)
        t0 = self._now()
        blocks = self.ctrl.sync_read_raw_data(ids, FEEDBACK_ADDR, FEEDBACK_LEN)
        t1 = self._now()
        t = None if t0 is None else 0.5 * (t0 + t1)
        return {i: decode_feedback(b, t) for i, b in zip(ids, blocks)}


class RealClock:
    def monotonic(self):
        import time
        return time.monotonic()

    def sleep(self, s):
        import time
        if s > 0:
            time.sleep(s)


def open_bus(port: str, ids, baud: int = BAUD_1M, timeout_s: float = 0.02) -> HlsBus:
    """A real bus through rustypot's STS3215 controller (same frame and checksum as HLS; only the raw
    calls are used)."""
    from rustypot import Sts3215PyController
    return HlsBus(Sts3215PyController(port, baud, timeout_s), ids, clock=RealClock())
