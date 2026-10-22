| condition | fwd .10 | fwd .20 | fwd .30 | back .10 | left .10 | turn | turn rest | falls | 0.4 kick | stand max | stand legs | peak | Δfwd vs nominal | verdict | falls in | failing gate rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nominal | 0.087 | 0.201 | 0.278 | -0.107 | 0.099 | 1.111 | 1.070 | 0 | 1 | 0.038 | 0.173 | 0.671 ⚠ | +0.000 / +0.000 / +0.000 | robust | - | - |
| vin6.5 | 0.082 | 0.184 | 0.261 | -0.097 | 0.093 | 1.069 | 1.050 | 0 | 1 | 0.039 | 0.177 | 0.651 ⚠ | -0.005 / -0.016 / -0.017 | robust | - | - |
| vin8.2 | 0.091 | 0.213 | 0.292 | -0.113 | 0.101 | 1.139 | 1.077 | 0 | 0 | 0.037 | 0.169 | 0.715 ⚠ | +0.003 / +0.013 / +0.014 | robust | - | - |
| vin6.5_sag0.2 | 0.082 | 0.187 | 0.258 | -0.098 | 0.088 | 1.064 | 1.047 | 0 | 1 | 0.039 | 0.178 | 0.693 ⚠ | -0.005 / -0.014 / -0.020 | robust | - | - |
| friction_x0.5 | 0.092 | 0.217 | 0.297 | -0.109 | 0.110 | 1.148 | 1.129 | 0 | 0 | 0.039 | 0.149 | 0.747 ⚠ | +0.004 / +0.016 / +0.019 | robust | - | - |
| friction_x1.5 | 0.083 | 0.191 | 0.248 | -0.095 | 0.083 | 1.026 | 1.007 | 0 | 1 | 0.031 | 0.217 | 0.828 ⚠ | -0.005 / -0.010 / -0.030 | degraded | - | - |
| friction_x2 | 0.073 | 0.146 | 0.186 | -0.070 | 0.079 | 0.789 ✗ | 0.776 | 2 ✗ | 0 | 0.060 | 0.281 | 1.001 ✗ | -0.015 / -0.055 / -0.093 | fails | steps x1, push x3 | step 0->back 0.10 |ss_err|, step 0->left 0.10 |ss_err|, step 0.20->0 t_stop, turn |wz| mean, turn |wz| min, falls, settled RMS worst, substep peak worst |
| armature_x0.5 | 0.088 | 0.218 | 0.298 | -0.104 | 0.094 | 1.133 | 1.126 | 0 | 0 | 0.039 | 0.173 | 0.719 ⚠ | +0.000 / +0.017 / +0.020 | robust | - | - |
| armature_x1.5 | 0.090 | 0.200 | 0.260 | -0.101 | 0.098 | 1.092 | 1.043 | 1 ✗ | 0 | 0.037 | 0.170 | 0.720 ⚠ | +0.003 / -0.001 / -0.019 | fails | push x3 | falls |
| armature_x2 | 0.092 | 0.190 | 0.254 | -0.098 | 0.101 | 1.058 | 1.047 | 0 | 1 | 0.036 | 0.167 | 0.878 ⚠ | +0.005 / -0.011 / -0.025 | robust | - | - |
| kp_x0.5 | 0.053 ✗ | 0.095 | 0.198 | -0.023 ✗ | 0.074 | 0.836 | 0.828 | 2 ✗ | 1 | 0.036 | 0.178 | 0.886 ⚠ | -0.034 / -0.106 / -0.080 | fails | schedule x2, steps x1 | fwd 0.10 vx, back 0.10 vx, step 0->0.10 |ss_err|, step 0->back 0.10 |ss_err|, step 0->left 0.10 |ss_err|, falls, settled RMS worst |
| kp_x0.7 | 0.071 | 0.155 | 0.224 | -0.083 | 0.080 | 0.997 | 0.985 | 1 ✗ | 0 | 0.039 | 0.181 | 0.705 ⚠ | -0.016 / -0.046 / -0.054 | fails | push x1 | falls |
| kp_x1.3 | 0.098 | 0.226 | 0.307 | -0.119 | 0.111 | 1.157 | 1.099 | 0 | 0 | 0.036 | 0.162 | 0.788 ⚠ | +0.011 / +0.025 / +0.029 | robust | - | - |
| kp_x1.4 | 0.101 | 0.234 | 0.315 | -0.125 | 0.117 | 1.177 | 1.124 | 0 | 1 | 0.038 | 0.157 | 0.971 ✗ | +0.013 / +0.033 / +0.037 | degraded | - | substep peak worst |
| kp_x2 | 0.114 | 0.263 | 0.338 | -0.135 | 0.127 | 1.207 | 1.131 | 0 | 0 | 0.044 | 0.173 | 1.014 ✗ | +0.026 / +0.062 / +0.059 | degraded | - | step 0->back 0.10 |ss_err|, substep peak worst |
| kp_x4 | -0.004 ✗ | n/a | n/a | n/a | n/a | n/a | 1.105 | 7 ✗ | 0 | 0.447 ✗ | 3.199 ✗ | 2.205 ✗ | -0.091 / n/a / n/a | fails | schedule x3, steps x3, push x1, push_stand x2, turn_rest x1 | fwd 0.10 vx, stand drift |v_xy|, stand drift |wz|, falls, stand torque max joint, stand torque leg sum, settled RMS worst, substep peak worst |
| kt_x0.8 | 0.093 | 0.187 | 0.215 | -0.106 | 0.113 | 1.090 | 1.031 | 0 | 1 | 0.036 | 0.161 | 0.614 ⚠ | +0.006 / -0.014 / -0.063 | degraded | - | - |
| kt_x1.2 | 0.080 | 0.207 | 0.276 | -0.104 | 0.084 | 1.102 | 1.092 | 0 | 1 | 0.035 | 0.172 | 0.817 ⚠ | -0.007 / +0.006 / -0.002 | robust | - | - |
| R_x0.8 | 0.087 | 0.219 | 0.295 | -0.111 | 0.092 | 1.130 | 1.100 | 0 | 0 | 0.036 | 0.169 | 0.908 ✗ | +0.000 / +0.018 / +0.016 | degraded | - | substep peak worst |
| R_x1.2 | 0.087 | 0.190 | 0.253 | -0.100 | 0.098 | 1.079 | 1.047 | 0 | 1 | 0.039 | 0.174 | 0.646 ⚠ | -0.001 / -0.010 / -0.026 | robust | - | - |
| delay5-8 | 0.093 | 0.193 | 0.261 | -0.101 | 0.110 | 1.086 | 1.031 | 1 ✗ | 0 | 0.038 | 0.174 | 0.763 ⚠ | +0.005 / -0.008 / -0.018 | fails | push x3 | falls |
| delay8-12 | 0.091 | 0.181 | 0.179 | -0.087 | 0.107 | 1.026 | 1.001 | 2 ✗ | 0 | 0.039 | 0.176 | 0.824 ⚠ | +0.003 / -0.020 / -0.099 | fails | steps x1, push x1 | falls |
| jvlag2 | 0.085 | 0.208 | 0.281 | -0.100 | 0.099 | 1.090 | 1.036 | 0 | 0 | 0.037 | 0.169 | 0.759 ⚠ | -0.003 / +0.007 / +0.003 | robust | - | - |
| backlash | 0.098 | 0.214 | 0.287 | -0.090 | 0.117 | 1.068 | 1.045 | 0 | 1 | 0.046 | 0.241 | 0.749 ⚠ | +0.011 / +0.013 / +0.009 | degraded | - | - |

Gate rows judged against `hd1910_v1_final_eval_h2.json` (✗ = gated row fails, ⚠ = warning). Verdicts judge standing torque against the nominal row (see the script docstring).
Δfwd = fwd 0.10 / 0.20 / 0.30 vx minus the nominal condition's. Push-rollout falls are at the 0.2 or 0.3 m/s kick while walking at 0.10 m/s.
- **robust** (11): nominal, vin6.5, vin8.2, vin6.5_sag0.2, friction_x0.5, armature_x0.5, armature_x2, kp_x1.3, kt_x1.2, R_x1.2, jvlag2
- **degraded** (6): friction_x1.5, kp_x1.4, kp_x2, kt_x0.8, R_x0.8, backlash
- **fails** (7): friction_x2, armature_x1.5, kp_x0.5, kp_x0.7, kp_x4, delay5-8, delay8-12
