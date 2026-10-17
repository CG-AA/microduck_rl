# HD-1910 runtime specification

What any program that runs an HD-1910 policy on the robot must do so the
policy sees what it saw in training. The planned runtime is Python on a small
Linux board with a BMI270 IMU and a half-duplex TTL adapter for the servo bus
(decided 2026-10-17); nothing below depends on that choice except the IMU
section. The policy file's side of the contract is locked by
`tests/test_deploy_contract.py`; the simulator's side by
`tests/test_joint_layout.py` and `scripts/infer_policy.py --hd1910`.

## The loop: 50 Hz, one tick = 20 ms

| Step | What | Time budget |
|---|---|---|
| 1 | Sync-read the 15-byte feedback block (address 56) of the 14 policy servos | ~5 ms |
| 2 | Take the latest fused IMU sample | < 0.1 ms |
| 3 | Build the 61 observations (below) | < 0.2 ms |
| 4 | Run the network (onnxruntime, 1 thread) | 0.1–0.5 ms |
| 5 | Sync-write the 14 goal positions | ~0.5 ms |

- **Bus time is wire time.** At 1 Mbps a feedback reply is ~0.2 ms, and each
  servo listed before another delays its reply by ~0.3 ms (measured on real
  HD-1910s by the community kit), so 14 servos take ~4.5 ms in any language.
- **USB adapters buffer.** An FTDI adapter's default latency timer is 16 ms,
  which alone breaks the tick: set it to 1 ms
  (`echo 1 > /sys/bus/usb-serial/devices/ttyUSB0/latency_timer`, or a udev
  rule).
- **Missed ticks are skipped, never bursted.** A late tick runs once, on the
  next 20 ms boundary.
- **Log a tick-time histogram** from the first run. Target: p99 jitter under
  2 ms.

**Latency budget.** Training delayed every action by 15–30 ms before it
reached the motor, re-drawn every 5 ms. The servo-model sensitivity matrix
shows 25–40 ms is still fine, while 40–60 ms halves turning. The real
latency is the sum of four parts, and it must stay under ~40 ms:
- the feedback's age when the network runs;
- inference time;
- the write;
- the servo firmware's reaction, which the bench step response measures.

## Observation: 61 float32 values, in this order

| Index | Block | Source and units | What training added |
|---|---|---|---|
| 0–2 | angular velocity | IMU gyro in the trunk frame (x forward, y left, z up), rad/s, bias removed | ±0.03 rad/s noise, up to 6° mounting error, 0–1 tick lag |
| 3–5 | projected gravity | unit gravity vector in the trunk frame, Rᵀ·(0, 0, −1); upright reads (0, 0, −1) | ±0.01 noise, same mounting error, 0–1 tick lag |
| 6–19 | joint positions | servo angle minus the default pose, policy order, rad | ±0.001 noise, ±0.015 rad per-joint encoder bias |
| 20–33 | joint velocities | servo speed register (0.732 rpm per count), rad/s, **from the previous tick** | ±0.25 rad/s noise, exactly 1 tick lag |
| 34–47 | last action | the network's raw output from the previous tick (zeros at start) | — |
| 48–50 | twist command | vx, vy (m/s) and yaw rate (rad/s) in the trunk frame | trained ranges below |
| 51–54 | head command | neck pitch, head pitch, head yaw, head roll offsets (rad) | small ranges: keep 0 until tested |
| 55–60 | body command | six zeros | tiny ranges: keep 0 |

- **Joint velocity is one tick old on purpose.** Training fed it lagged
  exactly one control step. Feed the value read on the previous tick.
- **Twist ranges trained:**
  - vx ±0.4 m/s, vy ±0.3 m/s, yaw rate ±1.0 rad/s;
  - creeping at 0.05–0.15 m/s on one axis;
  - exact zero means stand.
  Outside these ranges the behaviour is untested.
- **Policy joint order** is the ONNX metadata `joint_names`:
  - left hip yaw, hip roll, hip pitch, knee, ankle;
  - neck pitch, head pitch, head yaw, head roll;
  - right hip yaw, hip roll, hip pitch, knee, ankle.

  The mouth servo is not part of the policy.
- **Default pose** is the ONNX metadata `default_joint_pos`, which is the
  HD-1910 stand pose: knee −0.6, hip pitch −0.782, ankle +0.182, hip roll
  4.5°. It is not the XL330 `HOME` pose that Pollen's runtime hardcodes. Read
  it from the file, never from a constant.
- **Ignore `joint_stiffness` and `joint_damping`** in the metadata. Under the
  BAM servo model they are placeholders (1.0 / 0.0). The servo's stiffness is
  the P register (below).

## Action

```
target[i] = default_pose[i] + action[i]        # action_scale 1.0, radians, policy order
goal_count = round(2048 + target_servo_rad * 4096 / (2*pi))
```

- **No filtering, no clipping** beyond the servo's angle limits. The policy
  was trained unfiltered. A low-pass on the targets changes the dynamics it
  learned.
- **Joint angle to servo angle.** Each joint's servo direction and zero must
  make the servo's angle equal the simulator's joint angle. The mirrored legs
  have opposite signs (the sign table is checked against the geometry in
  `tests/test_joint_layout.py`). Store a sign and an offset per joint in the
  runtime's config.

## Servo configuration

- **Settings profile:** `src/mjlab_microduck/bench/profiles/hd1910_sim_match.json`.
  It sets position mode, deadband 0, acceleration 0, duty limit 98 %, and
  D = I = 0. `scripts/bench_configure.py --apply` writes it.
- **P gain:** from the bench gain map, `p_for_sim`, the P closest to the
  trained stiffness of 0.449 duty/rad. The gain is the axis the policy is most
  sensitive to: ×0.7 falls, ×4 cannot stand. `hd1910_v2c` was trained with the
  gain randomized ×0.7–1.4 to cover integer P steps.
- **Write the RAM gains (50/51/52) after every power-up.** The firmware reloads
  them from the EEPROM copies at boot.
- **Never write register 44.** On HLS it is goal current, not goal time.

## Calibration

- **Joint zeros.** With the robot on a jig in the stand pose, record each
  servo's raw count and store the offset that makes it read the metadata
  default pose. Check: commanding the default pose with the robot standing
  gives a level trunk (under 2°) and flat soles.
- **IMU mounting.** Mount it rigidly on the trunk, axes aligned with the
  trunk (x forward, y left, z up). The simulated IMU sits 21 mm behind and
  15 mm below the trunk origin; anywhere on the trunk is fine for gravity and
  angular rate. Training tolerates up to 6° of mounting error in a random
  direction, but not a systematic tilt. Measure the static roll and pitch
  offset on a level surface once and subtract it.
- **Gyro bias.** Hold the robot still for 2 s at every boot and subtract the
  mean.

## IMU: BMI270

- **Read rate.** Read gyro and accelerometer at 200–400 Hz in their own
  thread.
- **Fusion.** Fuse with a Mahony or Madgwick filter (low accelerometer gain,
  for example Madgwick β ≈ 0.03–0.05, because walking accelerations corrupt
  the gravity estimate). Publish the latest gyro and gravity to the control
  loop.
- **On-chip filter.** Set the BMI270's low-pass to at least 40 Hz bandwidth:
  a tighter filter adds latency the policy did not train with.
- **Margins against training.** BMI270 gyro noise at 50 Hz bandwidth is about
  0.001 rad/s, far inside the trained ±0.03. A fused gravity estimate within
  1–2° during walking is also inside what training tolerated (±0.6° noise plus
  up to 6° fixed error). Total IMU latency must stay under 20 ms.

## Safety

| Condition | Action |
|---|---|
| trunk tilt > 60° for 0.1 s (training ends episodes at 70°) | damping mode (torque_enable 2) on all servos, then torque off when still |
| any servo ≥ 60 °C | zero command, then sit or limp; torque off at 65 °C (firmware limit 70 °C) |
| supply < 6.6 V (10 s mean; training covered 6.5–8.2 V) | zero command, sit, power down |
| no command for 0.5 s | zero twist: stand, do not limp |
| a servo reports a status flag (voltage, encoder, temperature, current, overload) | torque off that servo, stop the policy |
| 5 consecutive failed bus reads | hold the last goals, then damping mode |

**Startup sequence:**
1. Read the positions.
2. Turn torque on at P = 1, holding the present position.
3. Ramp the goals to the default pose over 2 s.
4. Raise P to the configured value and hold for 1 s.
5. Start the policy with a zero command.

## Python notes

At 50 Hz the language is not the bottleneck. The time goes to the wire and to
onnxruntime's compiled kernels, the same from any language. Python costs about
1–2 ms per tick plus occasional pauses. Keep those out of the loop:
- preallocate the observation and goal arrays;
- run `gc.freeze()` after start-up and disable the collector inside the loop;
- run the loop thread with `SCHED_FIFO` priority, optionally on an isolated
  core;
- keep the IMU fusion in its own thread;
- measure the tick histogram before optimizing anything.

## Before the floor

1. `uv run --with pytest pytest tests/test_deploy_contract.py` on the policy
   file (`MICRODUCK_TEST_ONNX=<file>`).
2. Run the runtime's observation builder and action path against the
   simulator. `scripts/eval_walk.py` and `scripts/infer_policy.py --hd1910
   --joint-vel-lag 1` are the reference implementations of every row above.
3. Robot suspended, feet free: run the stand policy for 10 s. The targets
   should stay near the default pose. Tilt the robot by hand: the legs should
   move to counter it. A leg that moves the wrong way has a wrong sign.
4. On the floor on a leash: stand for 30 s, then try 0.1 m/s.
