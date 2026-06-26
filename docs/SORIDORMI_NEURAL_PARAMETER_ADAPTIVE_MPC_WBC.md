# Soridormi Neural Parameter-Adaptive MPC/WBC Design

> Status: design note, no runtime implementation yet.
> Target platform: Open Duck Mini v2 / Soridormi.
> Core rule: **the neural network estimates model error; MPC/WBC enforces physics.**

This document captures the agreed design for a future Soridormi controller that
combines a physics-constrained MPC/WBC controller with a neural parameter
estimator. It is written as a durable handoff for future LLM sessions and
future implementation work.

## 1. Motivation

The existing Open Duck Mini v2 learned policy is a strong low-compute locomotion
baseline, but it does not explicitly enforce physical constraints. A pure neural
controller can learn common behavior, but it cannot guarantee constraints such
as friction cones, joint/rate limits, contact schedules, foot clearance, support
stability, or safe fallback behavior.

MPC/WBC has the opposite strength: it can explicitly reason about physical
limits, contact, posture, foot placement, and support stability. Its weakness is
that it depends on model parameters that drift in the real robot:

- servo strength, delay, stiffness, damping, friction, backlash, and rate limits;
- battery voltage and thermal effects;
- foot-ground friction and contact reliability;
- mass, center of mass, and inertia changes caused by payloads or assembly
  differences;
- wear, part abrasion, looseness, and robot-to-robot variation.

The intended controller therefore keeps MPC/WBC as the physical command
producer and uses a neural network only to estimate bounded model-parameter
errors from history.

```text
history/state/task context
        ↓
Neural Parameter Estimator
        ↓
bounded, filtered parameter correction Δθ
        ↓
MPC + WBC with adapted parameters
        ↓
position-servo command q_cmd
        ↓
Safety layer
```

## 2. Architectural principle

The most important system boundary is:

```text
Neural network estimates the model error.
MPC/WBC enforces the physics.
```

The neural estimator should not directly bypass hard constraints. It should not
be allowed to relax hard joint limits, hardware speed limits, fall thresholds,
emergency-stop rules, or other safety invariants. It may estimate softer model
parameters such as friction, servo response, mass scale, task weights, or gait
clearance offsets. MPC/WBC then solves the constrained control problem using
those adapted parameters.

## 3. Runtime control loop

At control time `t`:

```text
1. Read robot_state_t.
2. Build short history from recent state, command, prior command, measured
   response, contact, and tracking-error records.
3. Neural estimator outputs raw parameter corrections θ_raw_t.
4. Bound-map θ_raw_t into physically meaningful adapted parameters.
5. Low-pass filter adapted parameters so they change slowly.
6. Run MPC with adapted body/contact/weight parameters.
7. Run position-servo-friendly WBC/IK with adapted WBC/gait parameters.
8. Apply final safety projection and send MotorCommand.
9. Log q_prior, q_cmd, q_measured, errors, contacts, and θ_adapted for the next
   history window.
```

Compact formula:

```text
h_t = [e_{t-K}, ..., e_t]
θ_raw_t = NN(h_t, obs_t, desired_command_t, task_context_t)
θ_adapted_t = BoundMap(θ_raw_t)
θ_filtered_t = λ θ_filtered_{t-1} + (1 - λ) θ_adapted_t
q_prior_t = MPC_WBC(state_t, desired_command_t, task_context_t, θ_filtered_t)
q_cmd_t = SafetyLayer(q_prior_t)
```

Recommended first implementation frequencies:

```text
Servo command loop:     50 Hz
WBC / IK:               50 Hz
MPC:                    25-50 Hz
NN parameter estimate:  10-25 Hz
```

Parameter updates should be slower and more filtered than joint commands.
Suggested initial low-pass factor: `λ = 0.90-0.98`.

## 4. What the NN should estimate first

Do not let the first network output dozens of unrelated free parameters. Start
with a small set of high-impact, bounded, physically meaningful parameters.

### 4.1 V1 24D parameter head

Recommended first parameter vector:

```text
θ_adapt[24]

Servo / actuator:
0  global_servo_strength_scale
1  global_servo_delay_scale
2  global_servo_stiffness_scale
3  global_servo_damping_scale
4  hip_strength_scale
5  knee_strength_scale
6  ankle_strength_scale
7  action_rate_limit_scale

Contact:
8  friction_mu_estimate
9  contact_confidence_left
10 contact_confidence_right
11 slip_risk_left
12 slip_risk_right

Body:
13 mass_scale
14 com_x_offset
15 com_y_offset
16 inertia_roll_pitch_scale

MPC:
17 mpc_velocity_tracking_weight_scale
18 mpc_yaw_tracking_weight_scale
19 mpc_roll_pitch_weight_scale
20 mpc_force_smoothness_scale

WBC / gait:
21 swing_height_offset
22 target_clearance_offset
23 double_support_ratio_offset
```

The highest-priority parameters are:

```text
global_servo_strength_scale
global_servo_delay_scale
friction_mu_estimate
mass_scale
mpc_roll_pitch_weight_scale
mpc_force_smoothness_scale
swing_height_offset
target_clearance_offset
double_support_ratio_offset
```

### 4.2 Parameters that may be adjusted

The NN may estimate bounded changes to:

- servo strength, delay, stiffness, damping, and rate-limit scales;
- contact friction, contact confidence, and slip risk;
- mass scale and small CoM/inertia offsets;
- MPC tracking, posture, force-regularization, and force-smoothness weights;
- WBC stance/swing/posture weights;
- gait parameters such as swing height, target clearance, step timing, and
  double-support ratio.

### 4.3 Parameters that must remain hard safety invariants

The NN must not directly relax:

- hard joint angle limits;
- hard joint speed/rate limits;
- hardware current/torque limits when available;
- maximum safe base roll/pitch thresholds;
- emergency-stop and fall-detection rules;
- maximum step length/width/yaw-step limits;
- minimum required clearance gates;
- residual/action safety-layer bounds.

## 5. Bounded parameter mapping

The network should output raw values, never direct physical parameters. Convert
raw values through bounded maps.

Scale parameters:

```text
scale = exp(log_range * tanh(raw))
```

Offset parameters:

```text
offset = max_offset * tanh(raw)
```

Positive-only offsets, such as clearance increase:

```text
clearance_offset = max_offset * sigmoid(raw)
```

Friction estimate:

```text
mu = mu_min + (mu_max - mu_min) * sigmoid(raw)
```

Confidence/risk values:

```text
confidence = sigmoid(raw)
risk = sigmoid(raw)
```

Initial safe ranges:

```text
servo_strength_scale:     0.70-1.20
servo_delay_scale:        0.50-2.00
servo_stiffness_scale:    0.60-1.40
servo_damping_scale:      0.60-1.60
action_rate_limit_scale:  0.60-1.20
friction_mu_estimate:     0.30-1.20
contact_confidence:       0.00-1.00
slip_risk:                0.00-1.00
mass_scale:               0.80-1.30
com_x_offset:            -0.03 to +0.03 m
com_y_offset:            -0.02 to +0.02 m
inertia_roll_pitch_scale: 0.70-1.50
MPC weight scales:        0.50-3.00, depending on weight
swing_height_offset:     -0.005 to +0.025 m
target_clearance_offset:  0.000 to +0.025 m
double_support_offset:   -0.05 to +0.15
```

## 6. History features

The estimator should receive signals that reveal parameter drift. A first
history step can include:

```text
desired_command[3]
q_prior[14]
q_cmd/q_final[14]
q_measured[14]
q_error = q_cmd - q_measured[14]
dq_measured[14]
base_roll_pitch[2]
base_gyro[3]
base_accel[3]
velocity_error[3]
contact_expected[2]
contact_observed[2]
contact_error[2]
foot_clearance_error[2]
previous_theta_adapt[24] optional
battery_voltage[1] optional
temperature[1] optional
```

Initial history length:

```text
K = 20 control steps at 50 Hz ≈ 0.4 s
history_step_dim ≈ 80-120
```

A later version may add a slow encoder for temperature, battery, and wear-like
changes:

```text
fast_history: 20 steps, about 0.4 s
slow_history: 100-250 steps, about 2-5 s at low sample rate
```

## 7. Network backbone

Recommended first backbone:

```text
current obs + desired command + structured task context -> MLP -> z_now
short error/state/action history -> GRU -> z_fast
optional slow history -> GRU -> z_slow
concat(z_now, z_fast, z_slow) -> MLP parameter head -> θ_raw[24]
BoundMap(θ_raw) -> θ_adapted[24]
```

Start with parameter adaptation only. Do not output a joint residual in V1.
After the parameter estimator is stable, a later residual head may output a
small bounded residual:

```text
Δq, α = ResidualHead(obs, command, q_prior, history, task_context)
q_cmd = SafetyLayer(q_prior + α * bounded(Δq))
```

Residual must be small, rate-limited, and disabled when the robot is unstable.
Suggested first residual bound, if enabled later: `0.03-0.08 rad`.

## 8. Task context

Task context is useful, but the low-level controller should not consume raw
natural language. A planner/skill router should convert task descriptions into
structured motion-relevant context:

```text
skill_id
gait_mode
task_phase
stability_priority
target_clearance
max_speed / max_accel
carry_object flag
approach_person / cautious mode
recovery mode
```

This context may affect both:

1. MPC/WBC parameters, such as step length, acceleration limit, posture weight,
   double support, and target clearance.
2. The NN estimator, so it can infer different parameter/error meanings under
   different behaviors.

Raw global task descriptions remain high-level planner information.

## 9. Relationship to the original Open Duck policy

The original Open Duck Mini v2 learned policy remains valuable as:

- a low-compute baseline;
- a gait prior for normal flat walking;
- a fallback or comparison candidate;
- a teacher/data source for distillation.

The long-term combined architecture may use three priors:

```text
q_open_duck = original learned policy(obs, command)
q_mpc_wbc   = MPC_WBC(state, command, task_context, θ_adapted)
β           = fusion confidence, optional
q_base      = β q_open_duck + (1 - β) q_mpc_wbc
q_cmd       = SafetyLayer(q_base + α Δq)
```

This fusion is a later stage. The first implementation should focus on neural
parameter estimation for MPC/WBC, not policy fusion.

## 10. Recommended implementation stages

### Stage V0: MPC/WBC baseline

MPC/WBC runs with nominal parameters and produces safe position-servo commands.
Validate stance, stepping, flat walk, start/stop, and turning before adding NN.

### Stage V1: NN parameter estimator only

```text
NN(history, obs, command, task_context) -> θ_adapted
MPC_WBC(θ_adapted) -> q_cmd
```

No direct joint residual. This is the safest and most interpretable version.

### Stage V2: Clearance and contact adaptation

Focus the estimator on:

```text
friction_mu_estimate
contact_confidence
slip_risk
swing_height_offset
target_clearance_offset
double_support_ratio_offset
```

Measure improvement with Soridormi scenario suites and clearance gates.

### Stage V3: Small residual head

Add a bounded residual only after V1/V2 are stable. Disable residual when
roll/pitch, contact, or solver diagnostics indicate risk.

### Stage V4: Open Duck policy fusion

Optionally combine original policy prior and MPC/WBC prior with a learned fusion
weight. Keep MPC/WBC as the physical fallback.

### Stage V5: Low-compute distillation

For Raspberry Pi Zero 2W or similar low-compute targets, distill the full
adaptive MPC/WBC teacher into a pure ONNX policy, while keeping the high-compute
controller for training, evaluation, and capable hardware.

## 11. Evaluation gates

Every implementation stage should be evaluated with at least:

- scenario acceptance/failure count;
- fall/reset count;
- forward and yaw tracking error;
- base roll/pitch limits;
- foot-clearance p50/p05 and low-clearance ratio;
- toe-drag / scuff indicators;
- contact mismatch and slip-risk diagnostics;
- q_cmd vs q_measured tracking error;
- parameter smoothness and range-bound violations;
- solver infeasibility/fallback count.

For Soridormi's current M10 direction, the clearance gate remains critical:
new controllers must not hide low swing-foot clearance behind aggregate pass
metrics.

Before using this WBC direction as the main development focus, close the M10
engineering-process gate:

```bash
./scripts/validate_m10_engineering_process.sh
```

After that, WBC/model fine-tuning can focus on better clearance, startup/tail
behavior, and turning without weakening the existing scenario evidence path.

The first control-side implementation artifact is documented in
`docs/SORIDORMI_WBC_CLEARANCE_CONTROL.md`:

```bash
./scripts/plan_wbc_clearance_experiment.sh
./scripts/validate_wbc_clearance_contract.sh
```

This stage validates bounded clearance parameters and candidate plans only. A
runtime WBC backend still needs to be implemented before any candidate can run
in MuJoCo.

## 12. Safety policy

If the estimator is uncertain, stale, missing history, or producing out-of-range
signals, fall back to nominal safe MPC/WBC parameters.

Recommended hard behaviors:

```text
if base roll/pitch exceeds safe threshold:
  ignore neural residuals
  freeze or reset adaptive parameters toward nominal
  use recovery/stand-safe MPC/WBC mode

if MPC is infeasible:
  use last safe command or stand-safe fallback

if WBC violates joint/rate limits:
  project through safety layer and record diagnostics

if parameter updates oscillate:
  increase filtering, reduce update rate, or disable estimator
```

The estimator is an adaptation aid, not a safety authority.

## 13. One-sentence summary

Use the neural network to estimate how the real robot differs from the nominal
model; then let MPC/WBC, with physical constraints still enforced, compute the
new control command from those adapted parameters.
