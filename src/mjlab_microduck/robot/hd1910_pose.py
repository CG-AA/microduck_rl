"""The HD-1910 Microduck's standing pose: the one source the robot cfg, the CPU rehearsal, the eval
harness and the scene keyframe all read.

numpy only (no mjlab import) so scripts/infer_policy.py can import it in milliseconds.

Why not HOME_FRAME (the stock "STAND2" pose, knee ~0): on this body the "straight" leg is a dogleg,
the thigh and shank meet at ~121 deg, so with the knee at 0 the knee axis sits 20 mm off the
hip-ankle line and carries a gravity lever of ~0.055 N.m per knee just standing, and the 5 deg
inward hip roll leaves ~0.035 N.m on each hip roll. Extending the knee moves it toward the line;
4.5 deg of roll moves the centre of pressure to where the roll moment vanishes with the sole still
within 0.5 deg of flat. Measured with scripts/stand_torque.py (stiff hold, kp x8, CPU MuJoCo + BAM,
2026-10-16): legs sum of mean tau^2 0.0090 -> 0.0016 (-82 %), worst joint 0.056 (knee) -> 0.026
(head_pitch), static equilibrium from +-3 deg trunk tilts and +-0.05 rad joint noise. Knee -0.8
would give -86 % but sits 4.6 mm from the collinear singularity with 0.5 rad of hip flexion left;
-0.6 keeps 8.8 mm and 0.63 rad.

Regenerate: uv run scripts/stand_torque.py --knee=-0.6 --com-mm 0 --hip-roll-deg 4.5
"""
import numpy as np

JOINT_NAMES = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
)

# Left-leg sign; the right leg is the mirror (every value negated). Flat sole: ankle == knee - hip_pitch.
HD1910_STAND = dict(hip_yaw=0.0, hip_roll=-0.0785, hip_pitch=-0.7821, knee=-0.6, ankle=0.1821)
NECK_PITCH = HEAD_PITCH = 0.3491  # unchanged from HOME_FRAME

HD1910_TRUNK_Z_CONTACT = 0.1258  # trunk z with the soles touching the floor, trunk level (HOME: 0.1172)
HD1910_SPAWN_Z = (HD1910_TRUNK_Z_CONTACT + 0.003, HD1910_TRUNK_Z_CONTACT + 0.013)  # training reset, as (0.12, 0.13) was for HOME
HD1910_SPAWN_Z_CPU = HD1910_TRUNK_Z_CONTACT + 0.008  # infer_policy / eval_walk, as 0.125 was for HOME


def stand_pose_vector(chain=HD1910_STAND) -> np.ndarray:
    """14 joint positions in actuator order (= joint-tree order), float32."""
    left = [chain["hip_yaw"], chain["hip_roll"], chain["hip_pitch"], chain["knee"], chain["ankle"]]
    right = [-x + 0.0 for x in left]
    return np.array(left + [NECK_PITCH, HEAD_PITCH, 0.0, 0.0] + right, dtype=np.float32)


def init_state_joint_pos(chain=HD1910_STAND) -> dict:
    """{joint name: value} for EntityCfg.InitialStateCfg (exact names, so first-match order is irrelevant)."""
    return {name: float(v) for name, v in zip(JOINT_NAMES, stand_pose_vector(chain))}


HD1910_DEFAULT_POSE = stand_pose_vector()
