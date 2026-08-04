# Soridormi skill execution contract

skill execution contract adds the first execution-facing layer for the structured body-skill interface skill platform.  It is
still **dry-run only**: it resolves manifest-declared skills into high-level
velocity command plans, but it does not connect to MuJoCo, hardware, MCP, or
motor commands.

This registry is the planned Chromie-to-Soridormi body boundary. Chromie is the
brain that chooses high-level actions; Soridormi is the cerebellum that
validates skill parameters, checks availability/safety, and executes body
controllers.

The purpose is to make skill implementation incremental and testable:

1. declare a skill in `configs/skills/open_duck_mini_v2_skills.json`;
2. validate it with `./scripts/list_skills.sh --validate-only`;
3. register a safe dry-run planner when the skill becomes executable in sim;
4. later connect that planner to MuJoCo/runtime execution;
5. only after sim acceptance, consider hardware execution.

## Dry-run examples

List executable dry-run skills:

```bash
./scripts/run_skill_dry_run.sh --list
```

Create a walking command plan:

```bash
./scripts/run_skill_dry_run.sh walk_velocity \
  --args '{"vx_mps":0.12,"vy_mps":0.0,"yaw_radps":0.05,"duration_s":3.0}'
```

For normal human speech, prefer semantic wrappers over raw velocity numbers:

```bash
./scripts/run_skill_dry_run.sh walk_forward \
  --args '{"speed":"slow","duration_s":3.0}'
```

`walk_forward` accepts `slow`, `normal`, `medium`, `quick`, and
`fast_limited` speed labels and lowers them to bounded `walk_velocity`
commands. Chromie should use this path for requests such as "walk slowly" or
"walk quickly". Keep `walk_velocity` for engineering/debug cases where an
explicit velocity is genuinely intended.

Forward walking commands below `0.12 m/s` are raised to `0.12 m/s` before
runtime command overrides are emitted. This keeps tiny "walk" requests from
turning into a wiggle-in-place pattern. Use `stop`, `stand_idle`, or
`turn_in_place` when the intended behavior is stationary.

Machine-readable output:

```bash
./scripts/run_skill_dry_run.sh walk_velocity \
  --args '{"vx_mps":0.12,"duration_s":2.0}' \
  --json | python -m json.tool
```

## Executable subset

skill execution contract only registers dry-run planners for the currently available locomotion
wrappers:

- `stand_idle`
- `stop`
- `walk_velocity`
- `turn_in_place`
- `curve_walk`
- `sidestep`

Future skills such as `step_over_obstacle`, `sit_down`, `bow`, and `run` remain
manifest-declared but are rejected by the dry-run registry until their controller
and safety validation are added.

## Policy input boundary

Skill execution should not pass natural-language task descriptions to the low-level controller. A planner or skill router should translate user intent into bounded structured context first:

```text
skill_id
desired velocity or target state
task mode / gait style / clearance intent
environment labels such as terrain or obstacle metadata
```

The current skill execution contract/skill simulation execution implementation only lowers single-segment locomotion skills to velocity command overrides. Future skill execution should preserve the same boundary: validate manifest parameters, build policy context, then call the runtime policy that outputs the 14D action.

`walk_forward`, `walk_velocity`, and `curve_walk` also apply the shared minimum
useful forward walk speed of `0.12 m/s`. The resolved plan keeps
`requested_vx_mps` and `min_forward_speed_mps` metadata when a numeric command
is adjusted, so debugging can see both the user/planner request and the command
sent to the runtime.

## Safety rules

- Skill dry-run output is a high-level command plan, not motor targets.
- Hardware remains disabled in the skill manifest.
- Parameters are range-checked against the manifest.
- Unsupported or future skills are rejected rather than silently executed.
- Social speech/TTS remains outside Soridormi and belongs to Chromie.
- Chromie must call structured skills/context, not raw joint targets or
  low-level `action_14d` policy outputs.

## Functional validation

```bash
PYTHONPATH=src pytest -q \
  tests/test_skill_manifest.py \
  tests/test_skill_manifest_cli.py \
  tests/test_skill_execution.py

python -m compileall -q src tests

bash -n scripts/run_skill_dry_run.sh

./scripts/run_skill_dry_run.sh --list

./scripts/run_skill_dry_run.sh walk_velocity \
  --args '{"vx_mps":0.12,"duration_s":2.0}' \
  --json | python -m json.tool >/dev/null
```

## skill simulation execution: MuJoCo skill execution wrapper

`skill execution contract` only dry-runs skills into high-level velocity segments. `skill simulation execution` adds a
safe sim-only wrapper that takes one available locomotion skill, converts it to
runtime command overrides, and runs the existing policy rollout harness against
an already-running MuJoCo simulator.

This still does not execute hardware. It is only a bridge from manifest skill
vocabulary to the existing MuJoCo policy runtime.

Start MuJoCo explicitly with the same profile first:

```bash
./scripts/run_sim_server.sh \
  --backend mujoco \
  --profile open_duck_forward \
  --viewer \
  --follow-camera
```

Then execute an available locomotion skill:

```bash
./scripts/run_skill_in_sim.sh walk_velocity \
  --args '{"vx_mps":0.12,"vy_mps":0.0,"yaw_radps":0.05,"duration_s":3.0}' \
  --profile open_duck_forward \
  --log-format jsonl
```

By default, the wrapper converts `duration_s` to control steps using 50 Hz:

```text
steps = ceil(duration_s * control_hz)
```

It intentionally does **not** pass a wall-clock `--seconds` limit unless the
user explicitly supplies one. This avoids ending a skill after one simulator
step when CUDA/ONNX warm-up consumes the wall-clock seconds budget. Use
`--steps` to override the computed control-step count, or `--seconds` only when
you truly want an additional wall-clock cutoff.

For script/debug validation without launching the runtime container:

```bash
./scripts/run_skill_in_sim.sh walk_velocity \
  --args '{"vx_mps":0.12,"duration_s":2.0}' \
  --dry-run-only
```

`run_skill_in_sim.sh` currently supports single-segment velocity skills only.
Multi-segment choreography should land later through an interruptible scheduler
so each segment can be traced, cancelled, and safety-gated.

The wrapper sets command override environment variables after the policy profile
is resolved:

```text
SORIDORMI_COMMAND_X_OVERRIDE
SORIDORMI_COMMAND_Y_OVERRIDE
SORIDORMI_COMMAND_YAW_OVERRIDE
SORIDORMI_COMMAND_RAMP_SECONDS_OVERRIDE
```

This is important because policy profiles such as `open_duck_forward` define
command defaults; skill execution must intentionally override those defaults
without mutating profile YAML files.

## scripted social skills: scripted head/neck social skill

`look_direction` is now the first experimental sim-available social skill. It
uses a scripted head/neck keyframe path instead of the walking policy velocity
wrapper, so run it through the dedicated social script:

```bash
./scripts/run_scripted_social_skill_in_sim.sh look_direction \
  --args '{"head_yaw_rad":0.25,"head_pitch_rad":-0.08,"duration_s":1.2}' \
  --backend mujoco \
  --dry-run
```

For live MuJoCo execution, start the simulator first with:

```bash
./scripts/run_sim_server.sh \
  --backend mujoco \
  --profile open_duck_forward \
  --viewer \
  --follow-camera
```

Then run the same `run_scripted_social_skill_in_sim.sh` command without
`--dry-run`. The executor preserves non-head joints at their current simulator
positions and only targets `neck_pitch`, `head_pitch`, `head_yaw`, and
`head_roll`. Hardware remains unavailable.
