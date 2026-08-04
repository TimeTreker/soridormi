# Patch delivery and validation expectations

This document is a handoff rule for future LLM sessions working on Soridormi.

The user normally downloads generated patch files to:

```bash
~/Downloads
```

Use that path in user-facing commands unless the user says otherwise.

## Patch format

Prefer plain git patch files over zip archives.

Recommended delivery:

```text
<descriptive_patch_name>.patch
```

Do not assume the user has applied earlier experimental patches unless they explicitly say so. If a new patch replaces or merges a previous patch, say that clearly and give only the patch that should be applied.

## Every patch must include two validation sections

Whenever giving the user a patch, include both of these sections.

### 1. Patch integrity check

This proves the patch can be applied to the expected repository state.

Use the user's download location in commands:

```bash
cd /path/to/soridormi
git apply --check ~/Downloads/<patch_name>.patch
```

When appropriate, also give the apply command:

```bash
git apply ~/Downloads/<patch_name>.patch
```

If the patch is incremental after another patch, state the required apply order before the commands.

### 2. Functional validation

This proves the new behavior, documentation, or interface works after the patch is applied.

For code patches, include the most relevant tests and smoke commands, for example:

```bash
PYTHONPATH=src pytest -q tests/test_<feature>.py
python -m compileall -q src tests
PYTHONPATH=src python -m <module> --help
```

For sim/training patches, separate checks into:

```text
A. local/unit validation that can run without MuJoCo
B. live sim validation that requires the simulator
```

For docs-only patches, still include functional validation. Use checks that prove the expected files and sections exist and that Markdown fences are balanced:

````bash
test -f docs/<new_doc>.md
grep -R "expected phrase" docs/<new_doc>.md
python - <<'PY'
from pathlib import Path
for path in [Path("docs/<new_doc>.md")]:
    text = path.read_text(encoding="utf-8")
    if text.count("```") % 2:
        raise SystemExit(f"Unbalanced markdown fences: {path}")
    print(f"ok: {path}")
PY
````


## MuJoCo backend and viewer requirements for sim validation

For any patch that includes live simulator functional validation, the simulator command must explicitly use the MuJoCo backend. The default viewer mode is off/headless so tests can run unattended:

```bash
./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --no-viewer
```

When a patch benefits from visual inspection, also provide a viewer-enabled variant. This is not the default functional test, but it must be available in the command line so the user can watch the robot in MuJoCo:

```bash
./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --viewer
```

When the duck may move out of the initial frame, include the follow-camera variant for visual inspection:

```bash
./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --viewer --follow-camera
```

Use `--viewer` only when the host has a working graphical session/X11 forwarding. If `DISPLAY` is missing, the script should warn rather than treating the missing viewer as a locomotion failure.

## Be explicit about what was not validated

If a command cannot be run in the sandbox, say so. Examples:

```text
I did not run the live MuJoCo rollout because the simulator is not running in this sandbox.
I did not run ruff because ruff is not installed in this environment.
```

Do not claim that a model was trained, a simulator rollout passed, or hardware was tested unless that was actually done.

## Functional validation should match the patch scope

Examples:

```text
Docs-only patch:
  - grep expected roadmap/status text
  - Markdown fence check

Training/data patch:
  - dataset collector unit tests
  - grouped split tests
  - CLI smoke with --help or tiny dry-run if available
  - live MuJoCo command for the user to run locally

Runtime/API patch:
  - unit tests for the new API
  - compile checks
  - CLI/tool smoke test

Hardware patch:
  - dry-run validation only by default
  - explicit statement that no real actuator command was sent
```

## Current-work authority

Do not encode the current project priority in this patch-delivery contract. Read
`docs/STATUS.md` for the current work order. Every patch should state which
semantic contract it affects, for example:

```text
locomotion and closed-loop evaluation
body-skill or embodied-task contract
simulator/runtime/backend boundary
hardware commissioning
documentation governance
```

An orchestration or contract patch must not imply improved walking capability
without closed-loop MuJoCo evidence.
