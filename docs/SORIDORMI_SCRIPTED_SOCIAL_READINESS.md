# Soridormi scripted social readiness reports

M8N adds a promotion-readiness report for scripted social skills.  The report is
an explicit gate between `available_sim_experimental` and `available_sim`; it
never edits the skill manifest by itself.

The intended workflow is:

1. Keep new social skills experimental while designing and tuning them.
2. Run dry-run acceptance in CI.
3. Run live MuJoCo acceptance against `open_duck_forward` and save the JSON.
4. Generate a readiness report with `--require-live`.
5. Only then make a separate manifest patch promoting a skill to
   `available_sim`.

Dry-run report:

```bash
./scripts/report_scripted_social_readiness.sh --json | python -m json.tool
```

Live promotion gate:

```bash
./scripts/run_sim_server.sh \
  --backend mujoco \
  --profile open_duck_forward \
  --viewer \
  --follow-camera
```

Second terminal:

```bash
mkdir -p artifacts/scripted_social

./scripts/evaluate_scripted_social_skills.sh \
  --execute \
  --backend mujoco \
  --require-observed \
  --json \
  > artifacts/scripted_social/live_acceptance.json

./scripts/report_scripted_social_readiness.sh \
  --live-acceptance-json artifacts/scripted_social/live_acceptance.json \
  --require-live \
  --output-dir artifacts/scripted_social/readiness \
  --json | python -m json.tool
```

A skill is only a promotion candidate when:

- it is currently `available_sim_experimental`,
- dry-run acceptance passes,
- live MuJoCo acceptance passes,
- live telemetry reports no fall, and
- the required observed head-axis range is present.

The report writes:

- `scripted_social_readiness_report.json`
- `scripted_social_readiness_report.md`

The JSON is suitable for CI, and the Markdown is suitable for PR review.

## Docker path and JSON note

The readiness wrapper runs inside `compose.sim.yaml` service `runtime`.  Because
`artifacts/...` is not part of the standard `/app` or `/data` mounts, the wrapper
mounts the host repository at `/host_repo` and rewrites repo-relative path
arguments for `--live-acceptance-json`, `--output-dir`, and `--manifest`.  A host
file such as `artifacts/scripted_social/live_acceptance.json` is therefore
readable inside the container without moving it to `/data`.

The scripted-social wrappers also override the CUDA image entrypoint.  This keeps
`--json` stdout free of the NVIDIA container banner so commands such as
`./scripts/evaluate_scripted_social_skills.sh --json > file.json` and
`./scripts/report_scripted_social_readiness.sh --json | python -m json.tool` stay
parseable.

Absolute paths outside the repository are intentionally rejected by the readiness
wrapper; use a repo-relative path or `/data/...`.
