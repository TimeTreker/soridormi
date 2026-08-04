# Official observation/action parity

`check_latest_observation_action_parity.sh` is intended to be run on the host.
It now discovers the latest official trace and Soridormi policy log on the host,
then calls `check_observation_action_parity.sh`.

`check_observation_action_parity.sh` now runs the Python checker inside the
runtime Docker container, so `soridormi_runtime` is importable and `/data` paths
resolve correctly.

Typical sequence:

```bash
SORIDORMI_OFFICIAL_MAX_SECONDS=10 ./scripts/run_official_forward_baseline.sh
./scripts/run_official_compatible_policy_server.sh open_duck_forward
./scripts/run_policy_experiment.sh open_duck_forward
./scripts/check_latest_observation_action_parity.sh
```

You may also pass explicit host paths:

```bash
./scripts/check_observation_action_parity.sh \
  data/official_baseline/latest_official_baseline.trace.jsonl \
  data/logs/policy_open_duck_forward_YYYYMMDD_HHMMSS.mcap
```
