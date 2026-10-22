| condition | fwd .10 | fwd .20 | fwd .30 | back .10 | left .10 | turn | turn rest | falls | 0.4 kick | stand max | stand legs | peak | Δfwd vs nominal | verdict | falls in | failing gate rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nominal | 0.100 | 0.210 | 0.296 | -0.112 | 0.116 | 1.152 | 1.082 | 0 | 0 | 0.025 | 0.114 | 0.594 | +0.000 / +0.000 / +0.000 | robust | - | - |
| vin6.5 | 0.092 | 0.199 | 0.276 | -0.105 | 0.105 | 1.138 | 1.072 | 0 | 1 | 0.023 | 0.104 | 0.575 | -0.007 / -0.011 / -0.020 | robust | - | - |
| vin8.2 | 0.104 | 0.213 | 0.311 | -0.115 | 0.120 | 1.156 | 1.086 | 0 | 1 | 0.026 | 0.123 | 0.691 ⚠ | +0.005 / +0.003 / +0.015 | robust | - | - |
| vin6.5_sag0.2 | 0.091 | 0.196 | 0.273 | -0.104 | 0.100 | 1.137 | 1.072 | 0 | 1 | 0.023 | 0.104 | 0.559 | -0.009 / -0.014 / -0.023 | robust | - | - |
| friction_x0.5 | 0.105 | 0.215 | 0.310 | -0.112 | 0.118 | 1.138 | 1.080 | 0 | 0 | 0.032 | 0.133 | 0.566 | +0.005 / +0.005 / +0.014 | degraded | - | - |
| friction_x1.5 | 0.089 | 0.201 | 0.274 | -0.109 | 0.099 | 1.133 | 1.105 | 1 ✗ | 0 | 0.037 | 0.119 | 0.710 ⚠ | -0.010 / -0.009 / -0.022 | fails | push x1 | falls |
| friction_x2 | 0.073 | 0.184 | 0.267 | -0.089 | 0.064 | 1.034 | 1.023 | 2 ✗ | 0 | 0.064 | 0.167 | 1.091 ✗ | -0.027 / -0.026 / -0.029 | fails | steps x1, push x2 | step 0->left 0.10 |ss_err|, falls, settled RMS worst, substep peak worst |
| armature_x0.5 | 0.101 | 0.219 | 0.310 | -0.111 | 0.116 | 1.113 | 1.032 | 0 | 1 | 0.026 | 0.118 | 0.691 ⚠ | +0.001 / +0.009 / +0.014 | robust | - | - |
| armature_x1.5 | 0.098 | 0.201 | 0.273 | -0.107 | 0.107 | 1.155 | 1.100 | 0 | 0 | 0.025 | 0.113 | 0.702 ⚠ | -0.002 / -0.009 / -0.023 | robust | - | - |
| armature_x2 | 0.097 | 0.188 | 0.261 | -0.096 | 0.101 | 1.141 | 1.093 | 1 ✗ | 0 | 0.025 | 0.114 | 0.787 ⚠ | -0.002 / -0.022 / -0.035 | fails | push x3 | falls |
| kp_x0.5 | 0.066 ✗ | 0.143 | 0.209 | -0.061 | 0.057 ✗ | 0.942 | 0.866 | 0 | 1 | 0.020 | 0.075 | 0.570 | -0.033 / -0.067 / -0.087 | degraded | - | fwd 0.10 vx, left 0.10 vy, step 0->0.10 |ss_err|, step 0->back 0.10 |ss_err|, step 0->left 0.10 |ss_err| |
| kp_x0.7 | 0.080 | 0.174 | 0.248 | -0.088 | 0.084 | 1.071 | 1.015 | 0 | 1 | 0.020 | 0.090 | 0.537 | -0.019 / -0.036 / -0.048 | degraded | - | - |
| kp_x1.3 | 0.113 | 0.222 | 0.331 | -0.122 | 0.129 | 1.170 | 1.076 | 0 | 0 | 0.029 | 0.137 | 0.815 ⚠ | +0.013 / +0.012 / +0.035 | degraded | - | - |
| kp_x1.4 | 0.117 | 0.231 | 0.345 | -0.127 | 0.126 | 1.192 | 1.057 | 0 | 0 | 0.030 | 0.144 | 0.895 ⚠ | +0.017 / +0.021 / +0.049 | degraded | - | step 0->left 0.10 |ss_err| |
| kp_x2 | 0.127 | 0.262 | 0.363 | -0.133 | 0.128 | 1.131 | 1.000 | 0 | 0 | 0.040 | 0.181 | 1.070 ✗ | +0.027 / +0.052 / +0.067 | degraded | - | step 0->back 0.10 |ss_err|, step 0->left 0.10 |ss_err|, settled RMS worst, substep peak worst |
| kp_x4 | 0.024 ✗ | 0.245 | n/a | n/a | n/a | n/a | 0.515 | 9 ✗ | 0 | 0.441 ✗ | 3.303 ✗ | 2.377 ✗ | -0.076 / +0.035 / n/a | fails | schedule x3, steps x3, push x2, push_stand x2, turn_rest x3 | fwd 0.10 vx, step 0->0.10 |ss_err|, turn from rest |wz| min @0.5, stand drift |v_xy|, stand drift |wz|, falls, stand torque max joint, stand torque leg sum, settled RMS worst, substep peak worst, speed_util_frac |
| kt_x0.8 | 0.107 | 0.203 | 0.281 | -0.111 | 0.129 | 1.161 | 1.142 | 0 | 1 | 0.026 | 0.105 | 0.594 | +0.008 / -0.007 / -0.015 | degraded | - | step 0->left 0.10 |ss_err| |
| kt_x1.2 | 0.091 | 0.200 | 0.282 | -0.106 | 0.102 | 1.075 | 0.996 | 1 ✗ | 0 | 0.024 | 0.124 | 0.682 ⚠ | -0.008 / -0.010 / -0.014 | fails | push x1 | falls |
| R_x0.8 | 0.100 | 0.216 | 0.310 | -0.117 | 0.111 | 1.130 | 1.044 | 0 | 1 | 0.026 | 0.129 | 0.733 ⚠ | +0.001 / +0.006 / +0.014 | robust | - | - |
| R_x1.2 | 0.099 | 0.202 | 0.278 | -0.106 | 0.115 | 1.159 | 1.110 | 0 | 1 | 0.024 | 0.103 | 0.537 | -0.000 / -0.008 / -0.018 | robust | - | - |
| delay5-8 | 0.099 | 0.202 | 0.273 | -0.109 | 0.111 | 1.150 | 1.105 | 0 | 0 | 0.026 | 0.117 | 0.725 ⚠ | -0.000 / -0.008 / -0.023 | robust | - | - |
| delay8-12 | 0.082 | 0.110 | 0.028 ✗ | -0.001 ✗ | n/a | n/a | 1.012 | 3 ✗ | 0 | 0.030 | 0.125 | 1.353 ✗ | -0.018 / -0.100 / -0.268 | fails | schedule x3, push x2 | back 0.10 vx, step 0->left 0.10 |ss_err|, fwd 0.30 vx, falls, settled RMS worst, substep peak worst |
| jvlag2 | 0.096 | 0.205 | 0.282 | -0.106 | 0.105 | 1.119 | 1.111 | 1 ✗ | 0 | 0.025 | 0.114 | 0.625 ⚠ | -0.004 / -0.005 / -0.014 | fails | push x1 | falls |
| backlash | 0.106 | 0.209 | 0.289 | -0.096 | 0.119 | 1.223 | 1.175 | 0 | 0 | 0.037 | 0.123 | 0.683 ⚠ | +0.006 / -0.001 / -0.007 | degraded | - | - |

Gate rows judged against `hd1910_v1_final_eval_h2.json` (✗ = gated row fails, ⚠ = warning). Verdicts judge standing torque against the nominal row (see the script docstring).
Δfwd = fwd 0.10 / 0.20 / 0.30 vx minus the nominal condition's. Push-rollout falls are at the 0.2 or 0.3 m/s kick while walking at 0.10 m/s.
- **robust** (9): nominal, vin6.5, vin8.2, vin6.5_sag0.2, armature_x0.5, armature_x1.5, R_x0.8, R_x1.2, delay5-8
- **degraded** (8): friction_x0.5, kp_x0.5, kp_x0.7, kp_x1.3, kp_x1.4, kp_x2, kt_x0.8, backlash
- **fails** (7): friction_x1.5, friction_x2, armature_x2, kp_x4, kt_x1.2, delay8-12, jvlag2
