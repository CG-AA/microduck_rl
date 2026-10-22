| condition | fwd .10 | fwd .20 | fwd .30 | back .10 | left .10 | turn | turn rest | falls | 0.4 kick | stand max | stand legs | peak | Δfwd vs nominal | verdict | falls in | failing gate rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nominal | 0.114 | 0.218 | 0.293 | -0.114 | 0.097 | 1.146 | 1.092 | 0 | 0 | 0.050 | 0.150 | 0.874 ⚠ | +0.000 / +0.000 / +0.000 | robust | - | - |
| vin6.5 | 0.110 | 0.202 | 0.270 | -0.101 | 0.088 | 1.058 | 1.034 | 0 | 0 | 0.045 | 0.142 | 0.732 ⚠ | -0.005 / -0.016 / -0.022 | robust | - | - |
| vin8.2 | 0.120 | 0.229 | 0.305 | -0.117 | 0.101 | 1.190 | 1.125 | 0 | 1 | 0.054 | 0.159 | 0.957 ✗ | +0.005 / +0.012 / +0.013 | degraded | - | substep peak worst |
| vin6.5_sag0.2 | 0.109 | 0.201 | 0.266 | -0.101 | 0.088 | 1.053 | 1.033 | 1 ✗ | 0 | 0.045 | 0.142 | 0.726 ⚠ | -0.006 / -0.016 / -0.027 | fails | push x1 | falls |
| friction_x0.5 | 0.116 | 0.225 | 0.304 | -0.114 | 0.113 | 1.158 | 1.116 | 0 | 0 | 0.039 | 0.161 | 0.696 ⚠ | +0.002 / +0.007 / +0.012 | robust | - | - |
| friction_x1.5 | 0.109 | 0.202 | 0.276 | -0.100 | 0.091 | 1.009 | 1.010 | 0 | 1 | 0.058 | 0.178 | 0.746 ⚠ | -0.006 / -0.015 / -0.016 | degraded | - | - |
| friction_x2 | 0.087 | 0.176 | 0.249 | -0.062 | 0.081 | 0.967 | 0.916 | 1 ✗ | 0 | 0.062 | 0.197 | 1.013 ✗ | -0.028 / -0.042 / -0.044 | fails | push x3 | step 0->back 0.10 |ss_err|, step 0->left 0.10 |ss_err|, falls, settled RMS worst, substep peak worst |
| armature_x0.5 | 0.121 | 0.233 | 0.312 | -0.109 | 0.107 | 1.206 | 1.066 | 0 | 0 | 0.049 | 0.150 | 0.888 ⚠ | +0.007 / +0.016 / +0.020 | robust | - | - |
| armature_x1.5 | 0.107 | 0.205 | 0.273 | -0.110 | 0.095 | 1.072 | 1.059 | 0 | 0 | 0.050 | 0.151 | 0.633 ⚠ | -0.007 / -0.013 / -0.019 | robust | - | - |
| armature_x2 | 0.105 | 0.193 | 0.264 | -0.101 | 0.098 | 1.046 | 1.012 | 0 | 1 | 0.050 | 0.151 | 0.675 ⚠ | -0.009 / -0.024 / -0.029 | robust | - | - |
| kp_x0.5 | 0.083 | 0.143 | 0.223 | -0.046 ✗ | 0.069 | 0.859 | 0.776 | 0 | 1 | 0.028 | 0.110 | 0.533 | -0.031 / -0.075 / -0.069 | degraded | - | back 0.10 vx, step 0->back 0.10 |ss_err|, step 0->left 0.10 |ss_err|, turn |wz| min |
| kp_x0.7 | 0.098 | 0.178 | 0.235 | -0.082 | 0.079 | 0.993 | 0.954 | 0 | 0 | 0.037 | 0.129 | 0.549 | -0.016 / -0.040 / -0.057 | degraded | - | - |
| kp_x1.3 | 0.125 | 0.241 | 0.326 | -0.125 | 0.100 | 1.260 | 1.162 | 0 | 0 | 0.062 | 0.179 | 1.069 ✗ | +0.010 / +0.024 / +0.034 | degraded | - | substep peak worst |
| kp_x1.4 | 0.128 | 0.248 | 0.337 | -0.126 | 0.105 | 1.306 | 1.155 | 0 | 0 | 0.066 | 0.194 | 0.903 ✗ | +0.013 / +0.030 / +0.045 | degraded | - | substep peak worst |
| kp_x2 | 0.145 | 0.277 | 0.371 | -0.139 | 0.109 | 1.381 | 1.184 | 0 | 0 | 0.079 ✗ | 0.252 | 1.215 ✗ | +0.031 / +0.060 / +0.078 | degraded | - | step 0->0.10 |ss_err|, step 0->back 0.10 |ss_err|, stand torque max joint, substep peak worst |
| kp_x4 | 0.127 | 0.200 | 0.139 | -0.001 ✗ | n/a | n/a | 0.870 | 7 ✗ | 0 | 0.409 ✗ | 3.159 ✗ | 2.154 ✗ | +0.013 / -0.018 / -0.154 | fails | schedule x3, steps x2, push x1, push_stand x1, turn_rest x2 | back 0.10 vx, step 0->0.10 |ss_err|, step 0->back 0.10 |ss_err|, step 0->left 0.10 |ss_err|, turn from rest |wz| min @0.5, stand drift |v_xy|, stand drift |wz|, falls, stand torque max joint, stand torque leg sum, settled RMS worst, substep peak worst |
| kt_x0.8 | 0.123 | 0.215 | 0.281 | -0.111 | 0.113 | 1.255 | 1.166 | 1 ✗ | 0 | 0.041 | 0.140 | 0.526 | +0.009 / -0.003 / -0.011 | fails | push x1 | falls |
| kt_x1.2 | 0.099 | 0.206 | 0.285 | -0.109 | 0.101 | 1.080 | 1.017 | 0 | 0 | 0.059 | 0.164 | 0.842 ⚠ | -0.015 / -0.011 / -0.008 | degraded | - | - |
| R_x0.8 | 0.111 | 0.223 | 0.308 | -0.116 | 0.086 | 1.145 | 1.087 | 0 | 0 | 0.061 | 0.169 | 0.787 ⚠ | -0.004 / +0.006 / +0.016 | degraded | - | - |
| R_x1.2 | 0.115 | 0.206 | 0.271 | -0.106 | 0.113 | 1.125 | 1.092 | 1 ✗ | 0 | 0.043 | 0.139 | 0.560 | +0.001 / -0.012 / -0.022 | fails | push x2 | falls |
| delay5-8 | 0.112 | 0.205 | 0.274 | -0.113 | 0.097 | 1.092 | 1.089 | 0 | 1 | 0.050 | 0.152 | 0.768 ⚠ | -0.002 / -0.013 / -0.019 | robust | - | - |
| delay8-12 | 0.104 | 0.184 | 0.253 | -0.103 | 0.119 | 1.037 | 0.991 | 1 ✗ | 0 | 0.051 | 0.165 | 0.986 ✗ | -0.010 / -0.033 / -0.040 | fails | steps x2 | step 0->left 0.10 |ss_err|, falls, substep peak worst |
| jvlag2 | 0.109 | 0.206 | 0.285 | -0.107 | 0.106 | 1.127 | 1.110 | 0 | 0 | 0.050 | 0.151 | 0.709 ⚠ | -0.006 / -0.012 / -0.008 | robust | - | - |
| backlash | 0.118 | 0.223 | 0.296 | -0.098 | 0.113 | 1.248 | 1.107 | 0 | 0 | 0.042 | 0.218 | 0.732 ⚠ | +0.003 / +0.005 / +0.003 | degraded | - | - |

Gate rows judged against `hd1910_v1_final_eval_h2.json` (✗ = gated row fails, ⚠ = warning). Verdicts judge standing torque against the nominal row (see the script docstring).
Δfwd = fwd 0.10 / 0.20 / 0.30 vx minus the nominal condition's. Push-rollout falls are at the 0.2 or 0.3 m/s kick while walking at 0.10 m/s.
- **robust** (8): nominal, vin6.5, friction_x0.5, armature_x0.5, armature_x1.5, armature_x2, delay5-8, jvlag2
- **degraded** (10): vin8.2, friction_x1.5, kp_x0.5, kp_x0.7, kp_x1.3, kp_x1.4, kp_x2, kt_x1.2, R_x0.8, backlash
- **fails** (6): vin6.5_sag0.2, friction_x2, kp_x4, kt_x0.8, R_x1.2, delay8-12
