| condition | fwd .10 | fwd .20 | fwd .30 | back .10 | left .10 | turn | turn rest | falls | 0.4 kick | stand max | stand legs | peak | Δfwd vs nominal | verdict | falls in | failing gate rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nominal | 0.097 | 0.202 | 0.285 | -0.086 | 0.095 | 1.284 | 1.146 | 0 | 0 | 0.043 | 0.174 | 0.782 ⚠ | +0.000 / +0.000 / +0.000 | robust | - | - |
| vin6.5 | 0.098 | 0.191 | 0.267 | -0.081 | 0.094 | 1.269 | 1.246 | 0 | 0 | 0.043 | 0.166 | 0.803 ⚠ | +0.001 / -0.011 / -0.018 | robust | - | - |
| vin8.2 | 0.097 | 0.215 | 0.297 | -0.089 | 0.102 | 1.263 | 1.201 | 0 | 0 | 0.042 | 0.179 | 1.022 ✗ | -0.001 / +0.013 / +0.012 | degraded | - | substep peak worst |
| vin6.5_sag0.2 | 0.098 | 0.191 | 0.265 | -0.080 | 0.087 | 1.228 | 1.275 | 0 | 0 | 0.043 | 0.166 | 0.755 ⚠ | +0.001 / -0.011 / -0.020 | robust | - | - |
| friction_x0.5 | 0.094 | 0.217 | 0.299 | -0.089 | 0.086 | 1.270 | 1.098 | 0 | 0 | 0.035 | 0.151 | 0.840 ⚠ | -0.003 / +0.015 / +0.015 | robust | - | - |
| friction_x1.5 | 0.099 | 0.193 | 0.267 | -0.085 | 0.102 | 1.311 | 1.277 | 0 | 1 | 0.055 | 0.213 | 0.883 ⚠ | +0.002 / -0.009 / -0.017 | degraded | - | - |
| friction_x2 | 0.106 | 0.175 | 0.257 | -0.054 ✗ | 0.085 | 1.163 | 1.030 | 1 ✗ | 0 | 0.059 | 0.231 | 0.964 ✗ | +0.009 / -0.027 / -0.027 | fails | push x1 | back 0.10 vx, step 0->back 0.10 |ss_err|, falls, settled RMS worst, substep peak worst |
| armature_x0.5 | 0.108 | 0.238 | 0.309 | -0.083 | 0.099 | 1.317 | 1.224 | 0 | 0 | 0.041 | 0.172 | 1.029 ✗ | +0.011 / +0.035 / +0.025 | degraded | - | substep peak worst |
| armature_x1.5 | 0.086 | 0.188 | 0.265 | -0.094 | 0.095 | 1.266 | 1.095 | 0 | 0 | 0.043 | 0.174 | 0.936 ✗ | -0.011 / -0.015 / -0.020 | degraded | - | substep peak worst |
| armature_x2 | 0.082 | 0.180 | 0.254 | -0.098 | 0.078 | 1.316 | 1.103 | 0 | 0 | 0.043 | 0.174 | 0.820 ⚠ | -0.016 / -0.022 / -0.031 | degraded | - | step 0.20->0 t_stop |
| kp_x0.5 | 0.087 | 0.156 | 0.309 | -0.042 ✗ | 0.075 | 1.018 | 0.983 | 1 ✗ | 0 | 0.037 | 0.134 | 0.752 ⚠ | -0.010 / -0.046 / +0.024 | fails | push x2 | back 0.10 vx, step 0->back 0.10 |ss_err|, step 0->left 0.10 |ss_err|, falls, settled RMS worst |
| kp_x0.7 | 0.096 | 0.181 | 0.241 | -0.066 | 0.093 | 1.109 | 1.038 | 0 | 0 | 0.043 | 0.155 | 0.689 ⚠ | -0.001 / -0.021 / -0.043 | degraded | - | step 0->back 0.10 |ss_err| |
| kp_x1.3 | 0.095 | 0.233 | 0.321 | -0.099 | 0.105 | 1.466 | 1.226 | 0 | 0 | 0.048 | 0.187 | 1.108 ✗ | -0.002 / +0.030 / +0.036 | degraded | - | substep peak worst |
| kp_x1.4 | 0.095 | 0.235 | 0.333 | -0.103 | 0.095 | 1.570 | 1.296 | 0 | 1 | 0.050 | 0.190 | 1.234 ✗ | -0.002 / +0.033 / +0.048 | degraded | - | settled RMS worst, substep peak worst |
| kp_x2 | 0.103 | 0.242 | 0.372 | -0.115 | 0.112 | 1.904 | 1.720 | 0 | 0 | 0.062 | 0.213 | 1.177 ✗ | +0.006 / +0.040 / +0.087 | degraded | - | settled RMS worst, substep peak worst |
| kp_x4 | 0.089 | 0.122 | 0.144 | n/a | n/a | n/a | 0.459 ✗ | 8 ✗ | 0 | 0.426 ✗ | 3.191 ✗ | 2.253 ✗ | -0.008 / -0.080 / -0.140 | fails | schedule x3, steps x3, push x1, push_stand x1, turn_rest x2 | step 0->back 0.10 |ss_err|, step 0->left 0.10 |ss_err|, turn from rest |wz| min @1.0, turn from rest |wz| min @0.5, stand drift |v_xy|, stand drift |wz|, falls, stand torque max joint, stand torque leg sum, settled RMS worst, substep peak worst |
| kt_x0.8 | 0.104 | 0.192 | 0.269 | -0.089 | 0.101 | 1.482 | 1.420 | 0 | 0 | 0.043 | 0.161 | 0.627 ⚠ | +0.006 / -0.010 / -0.015 | robust | - | - |
| kt_x1.2 | 0.085 | 0.201 | 0.275 | -0.082 | 0.075 | 1.175 | 0.996 | 0 | 1 | 0.047 | 0.184 | 0.958 ✗ | -0.012 / -0.001 / -0.009 | degraded | - | substep peak worst |
| R_x0.8 | 0.088 | 0.212 | 0.296 | -0.091 | 0.078 | 1.225 | 1.054 | 0 | 0 | 0.047 | 0.185 | 1.053 ✗ | -0.010 / +0.009 / +0.011 | degraded | - | substep peak worst |
| R_x1.2 | 0.101 | 0.192 | 0.267 | -0.085 | 0.114 | 1.352 | 1.358 | 0 | 0 | 0.043 | 0.162 | 0.742 ⚠ | +0.004 / -0.010 / -0.017 | robust | - | - |
| delay5-8 | 0.091 | 0.190 | 0.266 | -0.097 | 0.096 | 1.288 | 1.053 | 0 | 0 | 0.043 | 0.176 | 0.818 ⚠ | -0.007 / -0.013 / -0.018 | robust | - | - |
| delay8-12 | 0.090 | 0.175 | 0.222 | -0.099 | 0.089 | 1.291 | 0.353 ✗ | 2 ✗ | 1 | 0.043 | 0.175 | 1.045 ✗ | -0.007 / -0.027 / -0.062 | fails | steps x3, turn_rest x1 | turn from rest |wz| min @1.0, falls, substep peak worst |
| jvlag2 | 0.090 | 0.186 | 0.264 | -0.083 | 0.095 | 1.274 | 1.064 | 0 | 0 | 0.043 | 0.174 | 0.890 ⚠ | -0.007 / -0.016 / -0.020 | robust | - | - |
| backlash | 0.101 | 0.207 | 0.294 | -0.079 | 0.102 | 1.258 | 1.300 | 0 | 0 | 0.047 | 0.227 | 0.906 ✗ | +0.004 / +0.005 / +0.009 | degraded | - | substep peak worst |

Gate rows judged against `hd1910_v1_final_eval_h2.json` (✗ = gated row fails, ⚠ = warning). Verdicts judge standing torque against the nominal row (see the script docstring).
Δfwd = fwd 0.10 / 0.20 / 0.30 vx minus the nominal condition's. Push-rollout falls are at the 0.2 or 0.3 m/s kick while walking at 0.10 m/s.
- **robust** (8): nominal, vin6.5, vin6.5_sag0.2, friction_x0.5, kt_x0.8, R_x1.2, delay5-8, jvlag2
- **degraded** (12): vin8.2, friction_x1.5, armature_x0.5, armature_x1.5, armature_x2, kp_x0.7, kp_x1.3, kp_x1.4, kp_x2, kt_x1.2, R_x0.8, backlash
- **fails** (4): friction_x2, kp_x0.5, kp_x4, delay8-12
