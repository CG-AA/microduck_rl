| condition | fwd .10 | fwd .20 | fwd .30 | back .10 | left .10 | turn | turn rest | falls | 0.4 kick | stand max | stand legs | peak | Δfwd vs nominal | verdict | falls in | failing gate rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nominal | 0.092 | 0.187 | 0.267 | -0.110 | 0.084 | 1.228 | 1.132 | 0 | 1 | 0.043 | 0.237 | 0.549 | +0.000 / +0.000 / +0.000 | robust | - | - |
| vin6.5 | 0.086 | 0.176 | 0.251 | -0.099 | 0.076 | 1.168 | 1.060 | 0 | 1 | 0.045 | 0.247 | 0.539 | -0.006 / -0.011 / -0.016 | robust | - | - |
| vin8.2 | 0.095 | 0.193 | 0.277 | -0.121 | 0.094 | 1.253 | 1.185 | 0 | 0 | 0.042 | 0.230 | 0.650 ⚠ | +0.003 / +0.006 / +0.010 | degraded | - | stand drift |wz| |
| vin6.5_sag0.2 | 0.085 | 0.175 | 0.249 | -0.097 | 0.075 | 1.156 | 1.057 | 0 | 1 | 0.045 | 0.248 | 0.594 | -0.007 / -0.012 / -0.018 | robust | - | - |
| friction_x0.5 | 0.095 | 0.191 | 0.273 | -0.115 | 0.088 | 1.222 | 1.138 | 0 | 0 | 0.050 | 0.188 | 0.601 ⚠ | +0.003 / +0.004 / +0.006 | degraded | - | stand drift |wz| |
| friction_x1.5 | 0.083 | 0.175 | 0.251 | -0.098 | 0.073 | 1.180 | 1.084 | 0 | 1 | 0.068 | 0.348 ✗ | 0.724 ⚠ | -0.009 / -0.012 / -0.016 | degraded | - | step 0->0.10 t90, stand torque leg sum |
| friction_x2 | 0.060 ✗ | 0.140 | 0.215 | -0.056 ✗ | 0.040 ✗ | 1.081 | 1.040 | 0 | 0 | 0.094 ✗ | 0.458 ✗ | 1.082 ✗ | -0.032 / -0.047 / -0.052 | degraded | - | fwd 0.10 vx, back 0.10 vx, left 0.10 vy, step 0->0.10 |ss_err|, step 0->back 0.10 |ss_err|, step 0->left 0.10 |ss_err|, stand torque max joint, stand torque leg sum, settled RMS worst, substep peak worst |
| armature_x0.5 | 0.090 | 0.190 | 0.268 | -0.108 | 0.085 | 1.201 | 1.086 | 0 | 1 | 0.042 | 0.231 | 0.519 | -0.002 / +0.003 / +0.001 | degraded | - | stand drift |wz| |
| armature_x1.5 | 0.092 | 0.184 | 0.265 | -0.117 | 0.086 | 1.212 | 1.136 | 1 ✗ | 0 | 0.043 | 0.238 | 0.626 ⚠ | -0.000 / -0.003 / -0.002 | fails | push x1 | falls |
| armature_x2 | 0.092 | 0.177 | 0.254 | -0.110 | 0.068 | 1.207 | 1.151 | 1 ✗ | 0 | 0.044 | 0.238 | 0.658 ⚠ | -0.000 / -0.010 / -0.013 | fails | push x3 | step 0->left 0.10 |ss_err|, falls |
| kp_x0.5 | 0.063 ✗ | 0.120 | 0.166 | -0.045 ✗ | 0.035 ✗ | 0.921 | 0.924 | 0 | 1 | 0.061 | 0.321 ✗ | 0.483 | -0.029 / -0.067 / -0.101 | degraded | - | fwd 0.10 vx, back 0.10 vx, left 0.10 vy, step 0->0.10 |ss_err|, step 0->back 0.10 |ss_err|, step 0->left 0.10 |ss_err|, stand torque leg sum |
| kp_x0.7 | 0.077 | 0.152 | 0.218 | -0.077 | 0.061 | 1.080 | 0.989 | 0 | 0 | 0.050 | 0.272 | 0.598 | -0.015 / -0.035 / -0.049 | degraded | - | step 0->left 0.10 |ss_err| |
| kp_x1.3 | 0.100 | 0.208 | 0.291 | -0.137 | 0.094 | 1.264 | 1.206 | 0 | 0 | 0.041 | 0.221 | 0.718 ⚠ | +0.008 / +0.021 / +0.024 | degraded | - | step 0->back 0.10 |ss_err|, stand drift |wz| |
| kp_x1.4 | 0.103 | 0.212 | 0.296 | -0.140 | 0.087 | 1.260 | 1.236 | 0 | 0 | 0.041 | 0.217 | 0.754 ⚠ | +0.011 / +0.025 / +0.029 | degraded | - | step 0->back 0.10 |ss_err| |
| kp_x2 | 0.114 | 0.218 | 0.315 | -0.157 | 0.096 | 1.338 | 1.271 | 0 | 0 | 0.040 | 0.197 | 0.863 ⚠ | +0.022 / +0.031 / +0.049 | degraded | - | step 0->back 0.10 |ss_err| |
| kp_x4 | 0.102 | 0.215 | 0.236 | -0.090 | 0.071 | 0.887 | 0.852 | 1 ✗ | 0 | 0.308 ✗ | 1.439 ✗ | 1.929 ✗ | +0.010 / +0.028 / -0.031 | fails | schedule x1 | step 0->back 0.10 |ss_err|, turn |wz| min, stand drift |v_xy|, stand drift |wz|, falls, stand torque max joint, stand torque leg sum, settled RMS worst, substep peak worst |
| kt_x0.8 | 0.101 | 0.188 | 0.264 | -0.113 | 0.096 | 1.305 | 1.240 | 0 | 0 | 0.047 | 0.251 | 0.561 | +0.009 / +0.001 / -0.003 | robust | - | - |
| kt_x1.2 | 0.082 | 0.178 | 0.254 | -0.103 | 0.068 | 1.121 | 1.033 | 0 | 1 | 0.041 | 0.225 | 0.659 ⚠ | -0.010 / -0.009 / -0.013 | degraded | - | stand drift |wz| |
| R_x0.8 | 0.089 | 0.191 | 0.270 | -0.118 | 0.087 | 1.193 | 1.110 | 0 | 0 | 0.041 | 0.223 | 0.772 ⚠ | -0.003 / +0.004 / +0.003 | degraded | - | stand drift |wz| |
| R_x1.2 | 0.092 | 0.183 | 0.259 | -0.105 | 0.085 | 1.230 | 1.132 | 0 | 0 | 0.046 | 0.250 | 0.531 | -0.000 / -0.004 / -0.008 | robust | - | - |
| delay5-8 | 0.090 | 0.189 | 0.264 | -0.109 | 0.089 | 1.207 | 1.180 | 1 ✗ | 0 | 0.043 | 0.238 | 0.560 | -0.002 / +0.002 / -0.003 | fails | push x2 | falls |
| delay8-12 | 0.090 | 0.180 | 0.241 | -0.094 | 0.099 | 1.142 | 1.098 | 1 ✗ | 0 | 0.043 | 0.236 | 0.651 ⚠ | -0.002 / -0.008 / -0.026 | fails | push x3 | stand drift |wz|, falls |
| jvlag2 | 0.090 | 0.187 | 0.271 | -0.104 | 0.078 | 1.221 | 1.220 | 0 | 0 | 0.043 | 0.238 | 0.588 | -0.002 / +0.000 / +0.004 | robust | - | - |
| backlash | 0.093 | 0.192 | 0.262 | -0.092 | 0.094 | 1.260 | 1.183 | 0 | 1 | 0.044 | 0.242 | 0.607 ⚠ | +0.001 / +0.004 / -0.005 | robust | - | - |

Gate rows judged against `hd1910_v1_final_eval_h2.json` (✗ = gated row fails, ⚠ = warning). Verdicts judge standing torque against the nominal row (see the script docstring).
Δfwd = fwd 0.10 / 0.20 / 0.30 vx minus the nominal condition's. Push-rollout falls are at the 0.2 or 0.3 m/s kick while walking at 0.10 m/s.
- **robust** (7): nominal, vin6.5, vin6.5_sag0.2, kt_x0.8, R_x1.2, jvlag2, backlash
- **degraded** (12): vin8.2, friction_x0.5, friction_x1.5, friction_x2, armature_x0.5, kp_x0.5, kp_x0.7, kp_x1.3, kp_x1.4, kp_x2, kt_x1.2, R_x0.8
- **fails** (5): armature_x1.5, armature_x2, kp_x4, delay5-8, delay8-12
