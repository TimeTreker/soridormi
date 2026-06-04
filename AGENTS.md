# AGENTS.md

Repository-local guidance for coding agents working on Soridormi.

## Primary objective

Build a reusable sim-to-real engineering runtime, not a one-off demo. Official Open Duck code is the reference; Soridormi should reproduce it through clean runtime/API/backend contracts and then make model replacement/training/hardware transfer easy.

## Current focus

M4.x: make the official ONNX walking policy run correctly in Soridormi.

Current next task: M4.13 exact official loop-order parity.

## Guardrails

- Do not replace the current task with open-loop gait.
- Do not start training before official policy parity is solved.
- Do not hide failures behind tuning.
- Do not remove the official baseline, replay, or parity scripts.
- Preserve Docker host wrapper behavior: user usually runs scripts from host.
- If a host script needs Python package imports, it should enter the correct Docker service internally.
- If official compatibility needs reference files, fail fast if they are missing.

## Patch style

The user prefers plain git patch files, not zip archives. Assume downloaded patches live in `~/Downloads` unless the user says otherwise.

When giving a patch, include:

```bash
cd /path/to/soridormi
git apply --check ~/Downloads/<patch_name>.patch
git apply ~/Downloads/<patch_name>.patch
```

Every patch response must also include functional validation commands, not only `git apply --check`. Match validation to the patch scope:

- docs-only: grep expected sections and check Markdown fences;
- code: run relevant `pytest`, compile checks, and CLI smoke tests;
- sim/training: provide local/unit checks plus live MuJoCo validation commands; start live sim with `./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --no-viewer`, and include `./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --viewer` as the optional visual inspection command when relevant;
- hardware: default to dry-run/read-only validation and state explicitly if no actuator command was sent.

If a patch is incremental, state the required apply order. If a patch merges/replaces a previous patch, say that the user should apply only the merged patch.

Include tests when behavior changes. Include docs when a milestone changes usage. Mention whether Docker rebuild is needed.

See `docs/PATCH_DELIVERY_AND_VALIDATION.md`.

For live simulator functional tests, do not rely on an implicit backend. Use MuJoCo explicitly; the viewer is off by default, but the command-line must make `--viewer` available for visual inspection. If the robot may walk out of the initial frame, include the viewer follow-camera command: `./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --viewer --follow-camera`. Exception: random teacher dataset collection owns its MuJoCo collection lifecycle by starting and stopping its own temporary sim container, so do not pair `collect_random_teacher_dataset.sh` with a second `run_sim_server.sh`; use the collector's own `--viewer` and usually `--follow-camera` flags. Use `--external-sim` only for advanced debugging.

## Validation expectations

Preferred validation:

```bash
pytest -q
```

For policy behavior:

```bash
SORIDORMI_OFFICIAL_MAX_SECONDS=10 ./scripts/run_official_forward_baseline.sh
./scripts/run_official_compatible_policy_server.sh open_duck_forward
./scripts/run_policy_experiment.sh open_duck_forward
./scripts/compare_latest_official_soridormi_trace.sh
./scripts/check_latest_observation_action_parity.sh
```

For backend isolation:

```bash
./scripts/replay_latest_official_targets.sh
./scripts/compare_official_replay_trace.sh
```
