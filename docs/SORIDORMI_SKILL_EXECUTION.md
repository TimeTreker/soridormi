# Soridormi M7C Skill Execution Registry

M7C adds the first execution-facing layer for the M7 skill platform.  It is
still **dry-run only**: it resolves manifest-declared skills into high-level
velocity command plans, but it does not connect to MuJoCo, hardware, MCP, or
motor commands.

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

Machine-readable output:

```bash
./scripts/run_skill_dry_run.sh walk_velocity \
  --args '{"vx_mps":0.12,"duration_s":2.0}' \
  --json | python -m json.tool
```

## Executable subset

M7C only registers dry-run planners for the currently available locomotion
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

The current M7C/M7D implementation only lowers single-segment locomotion skills to velocity command overrides. Future skill execution should preserve the same boundary: validate manifest parameters, build policy context, then call the runtime policy that outputs the 14D action.

## Safety rules

- Skill dry-run output is a high-level command plan, not motor targets.
- Hardware remains disabled in the skill manifest.
- Parameters are range-checked against the manifest.
- Unsupported or future skills are rejected rather than silently executed.
- Social speech/TTS remains outside Soridormi and belongs to Chromie.

## Functional validation

```bash
PYTHONPATH=src pytest -q \
  tests/test_skill_manifest_m7.py \
  tests/test_skill_manifest_cli_m7.py \
  tests/test_skill_execution_m7c.py

python -m compileall -q src tests

bash -n scripts/run_skill_dry_run.sh

./scripts/run_skill_dry_run.sh --list

./scripts/run_skill_dry_run.sh walk_velocity \
  --args '{"vx_mps":0.12,"duration_s":2.0}' \
  --json | python -m json.tool >/dev/null
```

## M7D: MuJoCo skill execution wrapper

`M7C` only dry-runs skills into high-level velocity segments. `M7D` adds a
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
