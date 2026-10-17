# HD-1910 bench: from a servo in a box to a fitted sim model

The simulation's servo model (`actuator/params/hd1910_provisional_m6.json`) is
datasheet-derived with the XL330's friction and rotor inertia, and the
P-register → stiffness mapping has never been measured. This page is the
procedure that replaces it with measurements. All tools run without hardware
(`--fake`, a simulated servo behind the same registers), so every step can be
rehearsed before the first session.

What matters most, from the servo-model sensitivity matrix (`docs/hd1910.md`,
"Servo-model sensitivity"): the **firmware gain** (the policy falls at ×0.7 of
the trained stiffness and cannot stand at ×4) and **friction** (×2 brings back
the slow-walk dead zone). kt and R within 20 %, rotor inertia within ×2, ±1°
of backlash and the battery voltage barely matter.

## Hardware

- One HD-1910-C001, a half-duplex TTL adapter for Feetech servos (Feetech
  FE-URT-1, or a Waveshare bus-servo driver board), a bench supply settable
  6.0–8.4 V that can deliver 2 A.
- A clamp holding the servo with its shaft horizontal, high enough for the arm
  to swing ±120° freely.
- A printed arm, 80 mm from the shaft to the tip-mass centre, its own mass
  weighed to 0.5 g (`--arm-mass`), and tip masses of about **150 g and 300 g**
  weighed to 0.5 g. Lighter tips (50 g) are held up by the gearbox friction
  and the recorder refuses them: gravity with the arm horizontal should be at
  least ~8× the friction.
- For the gain map: a way to clamp the horn so it cannot turn (the arm screwed
  against a block works).
- The servo reports its own temperature, voltage and current; no extra sensors.

## Software

```bash
uv sync                                              # rustypot is already a dependency
ls /dev/ttyUSB* /dev/ttyACM*                         # the adapter's port
sudo usermod -aG dialout $USER                       # once, then log out and in
```

Every script takes `--port /dev/ttyUSB0` (real) or `--fake` (simulated servo),
`--id` (default 1; a new servo ships as id 1), and stops on any status flag or
above `--temp-limit` (55 °C).

## Session checklist (one servo, about 2 hours)

1. **Read and configure** (`scripts/bench_configure.py`). First without
   `--apply`: it prints the registers and what the profile would change. Note
   what the factory left in `goal_speed` (46) and `goal_current` (44): their
   meaning in HLS position mode is unconfirmed, so the profile does not touch
   them. Then apply: position mode, deadband 0, acceleration 0, duty limit
   98 %, D = I = 0.
   ```bash
   uv run scripts/bench_configure.py --port /dev/ttyUSB0 --kp 8 --apply
   ```
2. **Gain map** (`scripts/bench_static.py gainmap`), horn clamped. Measures
   the duty per radian per P unit. It prints the P that reproduces the
   stiffness the policies trained at (kp_fw 156 = 0.449 duty/rad). Check that
   the per-P values agree (duty linear in P) and the max duty reads ~0.98.
   ```bash
   uv run scripts/bench_static.py gainmap --port /dev/ttyUSB0 --out runs/bench/gainmap.json
   ```
   This is the single most important number of the session. If the trained
   stiffness needs P ≈ 3, the integer steps are 33 % apart; the policy was
   trained with the gain randomized ×0.7–1.4 (`hd1910_v2c`), which covers
   them.
3. **Static torque point** (`bench_static.py hold`), arm on, 300 g: kt and R
   from the current needed to hold the arm at 45° and 90°, approached from
   both sides to cancel friction. A cross-check for the fit.
   ```bash
   uv run scripts/bench_static.py hold --port /dev/ttyUSB0 --mass 0.3 --length 0.08 --arm-mass 0.012 --p 16
   ```
4. **Record** (`scripts/bench_record.py`): for each tip mass and supply
   voltage, BAM's four 6 s trajectories at three P values (~3 min per
   invocation). It finds the hanging zero by releasing the arm from both sides
   and stops if friction holds it far from hanging. Change the mass or the
   supply between invocations; all logs go to one folder.
   ```bash
   for V in 6.0 7.4 8.2; do   # set the supply to $V first
     uv run scripts/bench_record.py --port /dev/ttyUSB0 --mass 0.15 --length 0.08 --arm-mass 0.012 --kp 4 8 16 --logdir runs/bench/raw
     uv run scripts/bench_record.py --port /dev/ttyUSB0 --mass 0.30 --length 0.08 --arm-mass 0.012 --kp 4 8 16 --logdir runs/bench/raw
   done
   uv run scripts/bench_record.py --port /dev/ttyUSB0 --mass 0.15 --length 0.08 --arm-mass 0.012 --kp 8 --kd 40 --logdir runs/bench/raw_d40
   ```
   The last line records once with the factory D gain, to check whether the
   sim's pure-P law explains it (step 5 on that folder). If it does not, the
   runtime must keep D = 0.
5. **Fit** (`scripts/bench_fit.py`), on the laptop, no servo needed. ~3 min
   at 20 000 evaluations on 12 cores; hold out one P for validation.
   ```bash
   uv run scripts/bench_fit.py --raw runs/bench/raw --gainmap runs/bench/gainmap.json --evals 20000 --workers -1 --validation-kp 16 --out runs/bench/fit_v1
   ```
   Look at `plots/`: the model should overlay every log, including the free
   swing after the drop. The validation error should be close to the fit
   error. A large error only on the fast trajectories (`sin_time_square`)
   means the firmware slews the goal (a speed limit the sim does not model):
   check `goal_speed`.
6. **Thermal** (`scripts/bench_thermal.py`), 300 g held horizontal, 20 min or
   60 °C: the real continuous torque at your ambient temperature. Repeat at a
   second torque if time allows.
   ```bash
   uv run scripts/bench_thermal.py --port /dev/ttyUSB0 --mass 0.3 --length 0.08 --arm-mass 0.012 --angle 90 --p 16
   ```

## Into the simulator

`bench_fit.py` writes `hd1910_m6_sim.json`: the fit in the XL330-law format
the sim's actuator reads, with `kp_fw_equivalent`, the kp_fw that reproduces
the measured stiffness at the chosen P (default: the P closest to the trained
stiffness). Then:

1. Copy it to `src/mjlab_microduck/actuator/params/hd1910_m6.json`, point
   `HD1910_PARAMS_JSON` at it in `microduck_hd1910_constants.py` and
   `scripts/infer_policy.py`, and set `HD1910_KP_FW` in both to
   `kp_fw_equivalent`.
2. `tests/test_hd1910.py` locks the provisional numbers (kt 0.62 / R 3.75,
   stiffness 0.55 N·m/rad at kp_fw 156, the datasheet stall points): replace
   them with the measured values (keep a `bench/hd1910_measured.json` with
   the gain map, the hold, the fit summary and the thermal result, and read
   the tests' numbers from it).
3. Narrow the servo randomization in `tasks/hd1910.py` to the measured
   spread (several servos measured is better than one), rerun the sensitivity
   matrix on the current policy with the new model, retrain.

## Rehearsal without hardware

```bash
S=runs/bench_sim
uv run scripts/bench_configure.py --fake --kp 3 --apply
uv run scripts/bench_static.py gainmap --fake --out $S/gainmap.json
uv run scripts/bench_record.py --fake --mass 0.15 --length 0.08 --arm-mass 0.01 --kp 4 8 16 --logdir $S/raw
uv run scripts/bench_record.py --fake --mass 0.30 --length 0.08 --arm-mass 0.01 --kp 4 8 16 --logdir $S/raw
uv run scripts/bench_fit.py --raw $S/raw --gainmap $S/gainmap.json --truth $S/raw/truth.json.planted --evals 20000 --workers -1 --out $S/fit
uv run scripts/bench_thermal.py --fake --mass 0.3 --length 0.08 --minutes 15
```

The fake (`src/mjlab_microduck/bench/fake.py`) runs BAM's pendulum on a
planted servo behind the HLS registers (4096-count encoder, 0.732 rpm speed
counts, sign-magnitude registers, bus latency on a virtual clock). The fit
must recover it within the tolerances in `bench_fit.py` (`TOLERANCES`: 10 % on
torque per volt and back-EMF damping, 5 % on the gain, 30 % on the holding
friction, 25 % on rotor inertia). On 2026-10-17 it did: 0.5 / 3.0 / 0.2 / 1.8
/ 4.5 %. The fake does not model the D and I gains, acceleration profiles or
overcurrent unloading.

## Traps

- **Address 44 is goal current on HLS** (goal time on the STS3215). rustypot's
  `write_goal_time` would write it. The bench code uses only raw reads and
  writes (`bench/hls.py`) and never writes 44.
- **Present load's sign is BIT10**, position/speed/current use BIT15, all
  sign-magnitude. rustypot's typed STS readers decode load with the wrong bit.
- **Gains live in RAM 50/51/52.** EEPROM 21/22/23 are copied there at power-up,
  so a runtime must write the RAM gains after every boot (or persist them with
  `bench_configure.py --persist-gains`).
- **EEPROM writes need the lock register at 0**; the tools unlock, write and
  re-lock. `--set-id` with more than one servo on the bus renames all of them.
- **The pendulum zero:** never trust a single torque-off reading; friction
  stops the arm anywhere in a band. The recorder releases from both sides.
- **Overcurrent unloading:** above the protection current (reg 28) for the
  protection time (reg 38, 2 s) the servo cuts torque. The gain map uses
  0.3 s pulses with rests; long blocked-rotor holds will trip it.
- **Heat:** the gain map and the thermal run heat the servo. Every script
  stops at 55 °C (`--temp-limit`); the thermal run at `--limit` (60 °C).
