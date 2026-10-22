| condition | fwd .10 | fwd .20 | fwd .30 | back .10 | left .10 | turn | turn rest | falls | 0.4 kick | stand max | stand legs | peak | Δfwd vs nominal | verdict | falls in | failing gate rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nominal | 0.087 | 0.173 | 0.275 | -0.104 | 0.105 | 1.197 | 1.055 | 0 | 0 | 0.038 | 0.136 | 0.869 ⚠ | +0.000 / +0.000 / +0.000 | robust | - | - |
| vin6.5 | 0.082 | 0.163 | 0.260 | -0.094 | 0.098 | 1.097 | 0.992 | 0 | 0 | 0.037 | 0.132 | 0.894 ⚠ | -0.005 / -0.010 / -0.015 | robust | - | - |
| vin8.2 | 0.093 | 0.180 | 0.298 | -0.112 | 0.117 | 1.257 | 1.095 | 0 | 0 | 0.038 | 0.138 | 0.940 ✗ | +0.006 / +0.007 / +0.024 | degraded | - | substep peak worst |
| vin6.5_sag0.2 | 0.081 | 0.161 | 0.255 | -0.091 | 0.097 | 1.096 | 0.998 | 0 | 0 | 0.037 | 0.131 | 0.871 ⚠ | -0.006 / -0.013 / -0.020 | robust | - | - |
| friction_x0.5 | 0.094 | 0.182 | 0.302 | -0.111 | 0.116 | 1.251 | 1.097 | 0 | 0 | 0.037 | 0.114 | 0.743 ⚠ | +0.007 / +0.009 / +0.028 | robust | - | - |
| friction_x1.5 | 0.083 | 0.160 | 0.252 | -0.094 | 0.095 | 1.043 | 1.000 | 0 | 0 | 0.039 | 0.167 | 0.940 ✗ | -0.004 / -0.013 / -0.023 | degraded | - | substep peak worst |
| friction_x2 | 0.075 | 0.141 | 0.212 | -0.073 | 0.074 | 0.748 ✗ | 0.728 | 0 | 1 | 0.040 | 0.183 | 0.977 ✗ | -0.012 / -0.032 / -0.063 | degraded | - | turn |wz| mean, turn |wz| min, settled RMS worst, substep peak worst |
| armature_x0.5 | 0.090 | 0.182 | 0.301 | -0.109 | 0.113 | 1.289 | 1.100 | 0 | 0 | 0.037 | 0.135 | 0.828 ⚠ | +0.003 / +0.009 / +0.027 | robust | - | - |
| armature_x1.5 | 0.088 | 0.167 | 0.267 | -0.099 | 0.108 | 1.060 | 1.015 | 0 | 0 | 0.038 | 0.135 | 1.139 ✗ | +0.001 / -0.006 / -0.008 | degraded | - | substep peak worst |
| armature_x2 | 0.081 | 0.161 | 0.252 | -0.090 | 0.116 | 0.972 | 0.944 | 0 | 1 | 0.038 | 0.134 | 0.902 ✗ | -0.006 / -0.012 / -0.023 | degraded | - | substep peak worst |
| kp_x0.5 | 0.068 ✗ | 0.110 | 0.186 | -0.052 ✗ | 0.067 | 0.839 | 0.785 | 0 | 1 | 0.031 | 0.123 | 0.638 ⚠ | -0.019 / -0.063 / -0.089 | degraded | - | fwd 0.10 vx, back 0.10 vx, step 0->0.10 |ss_err|, step 0->back 0.10 |ss_err|, step 0->left 0.10 |ss_err| |
| kp_x0.7 | 0.077 | 0.144 | 0.231 | -0.079 | 0.091 | 0.987 | 0.934 | 0 | 1 | 0.034 | 0.129 | 0.728 ⚠ | -0.010 / -0.029 / -0.044 | degraded | - | - |
| kp_x1.3 | 0.102 | 0.190 | 0.326 | -0.119 | 0.124 | 1.343 | 1.178 | 0 | 0 | 0.037 | 0.138 | 1.020 ✗ | +0.015 / +0.017 / +0.051 | degraded | - | substep peak worst |
| kp_x1.4 | 0.109 | 0.196 | 0.336 | -0.123 | 0.125 | 1.367 | 1.211 | 0 | 0 | 0.035 | 0.140 | 0.982 ✗ | +0.022 / +0.023 / +0.061 | degraded | - | substep peak worst |
| kp_x2 | 0.132 | 0.206 | 0.350 | -0.126 | 0.138 | 1.093 | 0.717 | 0 | 0 | 0.046 | 0.135 | 1.050 ✗ | +0.045 / +0.033 / +0.075 | degraded | - | step 0->left 0.10 |ss_err|, turn |wz| min, settled RMS worst, substep peak worst |
| kp_x4 | 0.136 | 0.121 | 0.046 ✗ | n/a | n/a | n/a | 0.832 | 9 ✗ | 0 | 0.412 ✗ | 3.253 ✗ | 2.157 ✗ | +0.049 / -0.052 / -0.229 | fails | schedule x3, steps x3, push x2, push_stand x3, turn_rest x2 | step 0->0.10 |ss_err|, fwd 0.30 vx, turn from rest |wz| min @0.5, stand drift |v_xy|, stand drift |wz|, falls, stand torque max joint, stand torque leg sum, settled RMS worst, substep peak worst, speed_util_frac |
| kt_x0.8 | 0.089 | 0.161 | 0.245 | -0.104 | 0.111 | 1.190 | 1.095 | 0 | 0 | 0.034 | 0.123 | 0.690 ⚠ | +0.002 / -0.012 / -0.029 | robust | - | - |
| kt_x1.2 | 0.086 | 0.170 | 0.273 | -0.103 | 0.100 | 1.095 | 0.929 | 0 | 1 | 0.039 | 0.141 | 0.835 ⚠ | -0.001 / -0.003 / -0.001 | robust | - | - |
| R_x0.8 | 0.090 | 0.183 | 0.297 | -0.112 | 0.115 | 1.205 | 0.973 | 0 | 0 | 0.038 | 0.140 | 1.168 ✗ | +0.003 / +0.010 / +0.022 | degraded | - | substep peak worst |
| R_x1.2 | 0.086 | 0.165 | 0.257 | -0.096 | 0.109 | 1.141 | 1.029 | 0 | 0 | 0.035 | 0.127 | 0.806 ⚠ | -0.001 / -0.008 / -0.018 | robust | - | - |
| delay5-8 | 0.086 | 0.169 | 0.265 | -0.095 | 0.122 | 1.161 | 1.105 | 0 | 0 | 0.038 | 0.135 | 0.762 ⚠ | -0.001 / -0.004 / -0.010 | robust | - | - |
| delay8-12 | 0.081 | 0.123 | 0.205 | -0.093 | 0.123 | 1.018 | 0.810 | 0 | 0 | 0.038 | 0.135 | 0.900 ✗ | -0.006 / -0.050 / -0.070 | degraded | - | turn |wz| min, substep peak worst |
| jvlag2 | 0.086 | 0.188 | 0.289 | -0.098 | 0.113 | 1.120 | 1.052 | 0 | 0 | 0.037 | 0.134 | 0.875 ⚠ | -0.001 / +0.014 / +0.015 | robust | - | - |
| backlash | 0.093 | 0.176 | 0.284 | -0.093 | 0.118 | 1.148 | 1.027 | 0 | 0 | 0.050 | 0.177 | 0.781 ⚠ | +0.006 / +0.003 / +0.009 | degraded | - | - |

Gate rows judged against `hd1910_v1_final_eval_h2.json` (✗ = gated row fails, ⚠ = warning). Verdicts judge standing torque against the nominal row (see the script docstring).
Δfwd = fwd 0.10 / 0.20 / 0.30 vx minus the nominal condition's. Push-rollout falls are at the 0.2 or 0.3 m/s kick while walking at 0.10 m/s.
- **robust** (10): nominal, vin6.5, vin6.5_sag0.2, friction_x0.5, armature_x0.5, kt_x0.8, kt_x1.2, R_x1.2, delay5-8, jvlag2
- **degraded** (13): vin8.2, friction_x1.5, friction_x2, armature_x1.5, armature_x2, kp_x0.5, kp_x0.7, kp_x1.3, kp_x1.4, kp_x2, R_x0.8, delay8-12, backlash
- **fails** (1): kp_x4
