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

When giving the user a zip:

- Include only involved/new/updated files, not the entire project.
- Preserve directory structure under `soridormi/` so the user can run:

```bash
unzip update.zip -d /tmp/update
rsync -av /tmp/update/soridormi/ ./
```

- Include tests when behavior changes.
- Include docs when a milestone changes usage.
- Mention whether Docker rebuild is needed.

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
