# Soridormi M8E scripted social skills

M8E lands the first safe social skill for Open Duck Mini v2: `look_direction`.
M8F expands the same safe path with `nod_yes` and `shake_no`. M8F.1 fixes
those repeated gestures so they are visible two-cycle motions rather than tiny
near-neutral keyframes. M8F.2 makes repeated social gestures strict neutral-home
axis gestures: pre-center the head, move only the intended axis, then return to
neutral. These skills are intentionally narrow, MuJoCo-first, and head/neck-only.
They do not use arms, hands, speech, natural language, or hardware.

## What is executable

The following skills are promoted to `available_sim_experimental` in
`configs/skills/open_duck_mini_v2_skills.json` with `execution` set to
`scripted_keyframe`:

- `look_direction`: one bounded target pose for head yaw/pitch;
- `nod_yes`: neutral start, repeated bounded down/up head-pitch keyframes, neutral end;
- `shake_no`: neutral start, repeated bounded left/right head-yaw keyframes, neutral end.

The `look_direction` planner produces one bounded head/neck keyframe:

```text
neck_pitch = 0.0
head_pitch = head_pitch_rad
head_yaw   = head_yaw_rad
head_roll  = 0.0
```

All parameters are validated through the skill manifest before any simulator
command is created. `nod_yes` and `shake_no` also validate `count`, `amplitude`,
and `duration_s`; `count` must be an integer number of cycles and must be at
least `2` for the visible social gestures.

`nod_yes` and `shake_no` are neutral-home gestures. They intentionally do **not**
preserve prior head pitch/yaw drift: the first keyframe commands a straight head
pose, the gesture moves only its intended axis, and the final keyframe returns to
neutral. This matches the expected social behavior: `shake_no` starts straight,
turns left/right at least twice, and ends straight; `nod_yes` starts straight,
moves down/up at least twice, and ends straight.

## Safety boundary

The scripted executor:

- accepts only `--backend mujoco`;
- preserves all non-head joints at the simulator-reported current position;
- targets only `neck_pitch`, `head_pitch`, `head_yaw`, and `head_roll`;
- supports single-keyframe and multi-keyframe head gestures;
- interpolates smoothly with a bounded smoothstep ramp for each segment;
- exposes no hardware backend;
- keeps arm and hand social skills as `unsupported_current_robot`.

This is a body capability primitive. Higher-level intent, speech, perception,
and language routing remain outside Soridormi.

## Dry-run validation

```bash
./scripts/run_skill_dry_run.sh look_direction \
  --args '{"head_yaw_rad":0.25,"head_pitch_rad":-0.08,"duration_s":1.2}' \
  --json | python -m json.tool

./scripts/run_scripted_social_skill_in_sim.sh look_direction \
  --args '{"head_yaw_rad":0.25,"head_pitch_rad":-0.08,"duration_s":1.2}' \
  --backend mujoco \
  --dry-run \
  --json | python -m json.tool

./scripts/run_scripted_social_skill_in_sim.sh nod_yes \
  --args '{"count":2,"amplitude":"small","duration_s":1.6}' \
  --backend mujoco \
  --dry-run \
  --json | python -m json.tool

./scripts/run_scripted_social_skill_in_sim.sh shake_no \
  --args '{"count":2,"amplitude":"small","duration_s":1.6}' \
  --backend mujoco \
  --dry-run \
  --json | python -m json.tool
```

## MuJoCo execution

Start the simulator explicitly:

```bash
./scripts/run_sim_server.sh \
  --backend mujoco \
  --profile open_duck_forward \
  --viewer \
  --follow-camera
```

Then run the scripted social skill in a second terminal:

```bash
./scripts/run_scripted_social_skill_in_sim.sh look_direction \
  --args '{"head_yaw_rad":0.25,"head_pitch_rad":-0.08,"duration_s":1.2}' \
  --backend mujoco \
  --control-hz 50

./scripts/run_scripted_social_skill_in_sim.sh nod_yes \
  --args '{"count":2,"amplitude":"small","duration_s":1.6}' \
  --backend mujoco \
  --control-hz 50

./scripts/run_scripted_social_skill_in_sim.sh shake_no \
  --args '{"count":2,"amplitude":"small","duration_s":1.6}' \
  --backend mujoco \
  --control-hz 50
```

This path talks to the same simulator API as the runtime loop, but it does not
launch the policy runtime or modify policy profiles.

## Dependency note

The shell wrapper runs the scripted skill inside `compose.sim.yaml` service
`runtime`, matching the rest of the Soridormi sim/runtime scripts. Host Python
packages such as `pyzmq` are therefore not required for normal wrapper use.

Direct module execution with `python -m soridormi_runtime.scripted_head_skill` is
still supported for development, but live MuJoCo execution through that direct
module path requires the project Python dependencies, including `pyzmq`.

## Validation

```bash
PYTHONPATH=src pytest -q \
  tests/test_scripted_social_skill_m8e.py \
  tests/test_skill_execution_m7c.py \
  tests/test_skill_sim_execution_m7d.py \
  tests/test_skill_manifest_m7.py \
  tests/test_scenario_curriculum_m8b.py \
  tests/test_scenario_manifest_m8.py

python -m compileall -q src tests
bash -n scripts/run_scripted_social_skill_in_sim.sh
```

## M8F visible gesture execution note

For repeated gestures (`nod_yes`, `shake_no`), each keyframe target is now
ramped only for the first part of the segment and then held. The default
`--transition-fraction 0.40` means a 0.40s keyframe spends about 0.16s ramping
and about 0.30s holding the extreme. This fixes the earlier behavior where the
command reached the left/right or down/up target only at the last step and then
immediately reversed, which could look like no visible shake in MuJoCo.

The live command output also prints the keyframe targets, commanded target
ranges, and observed simulator joint ranges. For `shake_no`, check that
`head_yaw` has a negative and positive commanded range. For `nod_yes`, check
that `head_pitch` has a negative and positive commanded range. If commanded
ranges are correct but observed ranges remain near zero, the next thing to debug
is actuator/joint mapping or simulator control response, not the skill plan.

## M8F.2 neutral-home axis isolation

`shake_no` is now treated as a yaw-only gesture. During live execution the
non-yaw head joints (`neck_pitch`, `head_pitch`, `head_roll`) are commanded
directly to their neutral targets every step instead of being blended from any
small simulator drift left by the previous keyframe. `nod_yes` uses the same
axis-isolation idea for pitch-only motion.

For the visual check below, the command report should show a neutral start
keyframe, two right/left cycles, and a neutral end keyframe. The commanded
`head_pitch` range for `shake_no` should stay at `0.000 .. 0.000`; the commanded
`head_yaw` range should show the left/right amplitude.

```bash
./scripts/run_scripted_social_skill_in_sim.sh shake_no \
  --args '{"count":2,"amplitude":"medium","duration_s":2.0}' \
  --backend mujoco \
  --control-hz 50
```

## M8F.3 head pose trajectory and stability fix

Scripted social skills now follow a stricter execution model:

1. Build a bounded head-pose trajectory for the requested action.
2. Stream one pose command per MuJoCo control step.
3. Preserve every non-head actuator from MuJoCo `actuator_ctrl` instead of raw
   joint `qpos`.

That third point is important for stability. `qpos` is the physical pose at the
current instant; using it as the next command for every leg joint can retarget
balancing joints to transient values while a head gesture is running. For social
head-only skills, only the head/neck target should change.

`shake_no` is now a neutral-home yaw-only trajectory:

- start at neutral head pose,
- yaw right/left for the requested cycle count,
- return to neutral,
- keep pitch, roll, and neck pitch at neutral throughout the gesture.

`nod_yes` is the pitch-only equivalent:

- start at neutral head pose,
- pitch down/up for the requested cycle count,
- return to neutral,
- keep yaw, roll, and neck pitch at neutral throughout the gesture.

The executor also limits planned head target speed with
`--max-head-velocity-radps` defaulting to `0.35`. If the requested `duration_s` is
too short for the amplitude/count, the executor auto-stretches the live and
dry-run duration. For example, this command requests `2.0s`, but the effective
sim duration is stretched so the head trajectory is slow enough to be visible and
less destabilizing:

```bash
./scripts/run_scripted_social_skill_in_sim.sh shake_no \
  --args '{"count":2,"amplitude":"medium","duration_s":2.0}' \
  --backend mujoco \
  --control-hz 50
```

Expected report highlights:

```text
Requested duration: 2.00s
Effective duration: 17.14s (auto-stretched for head speed limit)
Commanded target head range:
- head_pitch: min=0.000, max=0.000
- head_yaw: min=-0.400, max=0.400
```

For debugging only, `--no-auto-stretch-duration` or
`--max-head-velocity-radps 0.8` can restore the earlier debug speed, and `--max-head-velocity-radps 0` disables the limiter entirely. Those faster modes are not recommended for viewer validation.
