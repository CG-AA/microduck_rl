| condition | fwd .10 | fwd .20 | fwd .30 | back .10 | left .10 | turn | turn rest | falls | 0.4 kick | stand max | stand legs | peak | Δfwd vs nominal | verdict | falls in | failing gate rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nominal | 0.101 | 0.221 | 0.314 | -0.106 | 0.100 | 1.175 | 1.131 | 0 | 1 | 0.027 | 0.120 | 0.645 ⚠ | +0.000 / +0.000 / +0.000 | robust | - | - |
| vin6.5 | 0.099 | 0.208 | 0.290 | -0.097 | 0.103 | 1.098 | 1.090 | 0 | 0 | 0.025 | 0.115 | 0.618 ⚠ | -0.002 / -0.012 / -0.024 | robust | - | - |
| vin8.2 | 0.105 | 0.230 | 0.325 | -0.109 | 0.095 | 1.251 | 1.175 | 0 | 0 | 0.028 | 0.123 | 0.747 ⚠ | +0.003 / +0.010 / +0.011 | robust | - | - |
| vin6.5_sag0.2 | 0.099 | 0.206 | 0.286 | -0.096 | 0.092 | 1.086 | 1.087 | 0 | 0 | 0.025 | 0.114 | 0.746 ⚠ | -0.002 / -0.015 / -0.028 | robust | - | - |
| friction_x0.5 | 0.105 | 0.229 | 0.327 | -0.109 | 0.107 | 1.237 | 1.178 | 0 | 1 | 0.033 | 0.121 | 0.655 ⚠ | +0.004 / +0.009 / +0.013 | degraded | - | - |
| friction_x1.5 | 0.103 | 0.203 | 0.289 | -0.094 | 0.086 | 1.069 | 1.060 | 1 ✗ | 0 | 0.037 | 0.169 | 0.837 ⚠ | +0.002 / -0.017 / -0.025 | fails | push x2 | falls |
| friction_x2 | 0.089 | 0.179 | 0.245 | -0.061 | 0.080 | 1.024 | 0.988 | 1 ✗ | 0 | 0.104 ✗ | 0.200 | 1.021 ✗ | -0.013 / -0.042 / -0.069 | fails | push x3 | step 0->back 0.10 |ss_err|, step 0->left 0.10 |ss_err|, falls, stand torque max joint, settled RMS worst, substep peak worst |
| armature_x0.5 | 0.110 | 0.233 | 0.332 | -0.104 | 0.103 | 1.328 | 1.189 | 0 | 0 | 0.026 | 0.120 | 0.868 ⚠ | +0.009 / +0.012 / +0.018 | robust | - | - |
| armature_x1.5 | 0.098 | 0.207 | 0.293 | -0.104 | 0.085 | 1.047 | 1.007 | 0 | 1 | 0.027 | 0.121 | 0.649 ⚠ | -0.004 / -0.014 / -0.021 | robust | - | - |
| armature_x2 | 0.099 | 0.195 | 0.281 | -0.097 | 0.083 | 0.987 | 0.942 | 1 ✗ | 0 | 0.028 | 0.120 | 0.727 ⚠ | -0.003 / -0.026 / -0.033 | fails | push x3 | falls |
| kp_x0.5 | 0.087 | 0.145 | 0.334 | -0.049 ✗ | 0.070 | 0.906 | 0.879 | 1 ✗ | 0 | 0.028 | 0.106 | 0.682 ⚠ | -0.015 / -0.076 / +0.020 | fails | push x2 | back 0.10 vx, step 0->back 0.10 |ss_err|, step 0->left 0.10 |ss_err|, falls, settled RMS worst |
| kp_x0.7 | 0.097 | 0.184 | 0.252 | -0.079 | 0.084 | 1.027 | 1.003 | 1 ✗ | 0 | 0.024 | 0.111 | 0.583 | -0.005 / -0.037 / -0.061 | fails | push x1 | falls |
| kp_x1.3 | 0.110 | 0.241 | 0.344 | -0.116 | 0.104 | 1.299 | 1.224 | 0 | 0 | 0.029 | 0.142 | 1.045 ✗ | +0.008 / +0.020 / +0.031 | degraded | - | substep peak worst |
| kp_x1.4 | 0.114 | 0.248 | 0.350 | -0.117 | 0.103 | 1.335 | 1.248 | 0 | 0 | 0.031 | 0.150 | 1.011 ✗ | +0.012 / +0.027 / +0.036 | degraded | - | substep peak worst |
| kp_x2 | 0.137 | 0.288 | 0.379 | -0.123 | 0.094 | 1.517 | 1.199 | 0 | 0 | 0.048 | 0.204 | 1.058 ✗ | +0.036 / +0.067 / +0.065 | degraded | - | step 0->0.10 |ss_err|, substep peak worst |
| kp_x4 | 0.136 | 0.007 ✗ | n/a | n/a | n/a | n/a | 0.999 | 8 ✗ | 0 | 0.416 ✗ | 3.324 ✗ | 2.205 ✗ | +0.034 / -0.213 / n/a | fails | schedule x3, steps x3, push x2, push_stand x2, turn_rest x1 | fwd 0.20 vx, turn from rest |wz| min @0.5, stand drift |v_xy|, stand drift |wz|, falls, stand torque max joint, stand torque leg sum, settled RMS worst, substep peak worst |
| kt_x0.8 | 0.113 | 0.213 | 0.297 | -0.098 | 0.105 | 1.170 | 1.159 | 0 | 0 | 0.026 | 0.110 | 0.580 | +0.011 / -0.008 / -0.017 | robust | - | - |
| kt_x1.2 | 0.093 | 0.215 | 0.302 | -0.104 | 0.089 | 1.155 | 1.107 | 0 | 0 | 0.029 | 0.132 | 0.850 ⚠ | -0.009 / -0.006 / -0.012 | robust | - | - |
| R_x0.8 | 0.101 | 0.229 | 0.326 | -0.109 | 0.094 | 1.231 | 1.152 | 0 | 1 | 0.029 | 0.139 | 1.064 ✗ | -0.001 / +0.008 / +0.012 | degraded | - | substep peak worst |
| R_x1.2 | 0.107 | 0.211 | 0.289 | -0.101 | 0.101 | 1.143 | 1.121 | 0 | 0 | 0.025 | 0.112 | 0.574 | +0.005 / -0.010 / -0.025 | robust | - | - |
| delay5-8 | 0.101 | 0.212 | 0.290 | -0.105 | 0.112 | 1.109 | 1.067 | 0 | 1 | 0.027 | 0.121 | 0.705 ⚠ | -0.000 / -0.009 / -0.024 | robust | - | - |
| delay8-12 | 0.092 | 0.185 | 0.259 | -0.102 | 0.100 | 1.043 | 0.967 | 0 | 1 | 0.028 | 0.128 | 0.711 ⚠ | -0.009 / -0.035 / -0.055 | degraded | - | - |
| jvlag2 | 0.098 | 0.214 | 0.314 | -0.100 | 0.091 | 1.127 | 1.116 | 1 ✗ | 0 | 0.027 | 0.120 | 0.713 ⚠ | -0.004 / -0.007 / -0.000 | fails | push x2 | falls |
| backlash | 0.106 | 0.225 | 0.310 | -0.088 | 0.107 | 1.257 | 1.162 | 0 | 1 | 0.026 | 0.136 | 0.845 ⚠ | +0.005 / +0.004 / -0.004 | robust | - | - |

Gate rows judged against `hd1910_v1_final_eval_h2.json` (✗ = gated row fails, ⚠ = warning). Verdicts judge standing torque against the nominal row (see the script docstring).
Δfwd = fwd 0.10 / 0.20 / 0.30 vx minus the nominal condition's. Push-rollout falls are at the 0.2 or 0.3 m/s kick while walking at 0.10 m/s.
- **robust** (11): nominal, vin6.5, vin8.2, vin6.5_sag0.2, armature_x0.5, armature_x1.5, kt_x0.8, kt_x1.2, R_x1.2, delay5-8, backlash
- **degraded** (6): friction_x0.5, kp_x1.3, kp_x1.4, kp_x2, R_x0.8, delay8-12
- **fails** (7): friction_x1.5, friction_x2, armature_x2, kp_x0.5, kp_x0.7, kp_x4, jvlag2
