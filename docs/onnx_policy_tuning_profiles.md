# ONNX policy tuning profiles and log comparison

policy tuning profiles and log comparison makes closed-loop policy debugging repeatable. Instead of manually changing several environment variables at once, run named profiles and compare the resulting logs.

## Profiles

List profiles:

```bash
./scripts/list_policy_tuning_profiles.sh
```

Useful starting profiles:

```text
idle_debug         static baseline, zero command and zero phase
crawl_very_safe   smallest dynamic forward command
crawl_safe        recommended first dynamic test
walk_cautious     slightly stronger forward command/action
walk_default_soft near-default action scale, still softer than raw policy defaults
turn_cautious     yaw-only sign/order check
```

## Run a profile

Start MuJoCo with viewer and auto-reset:

```bash
SORIDORMI_SIM_BACKEND=mujoco \
SORIDORMI_MUJOCO_VIEWER=1 \
SORIDORMI_AUTO_RESET=1 \
./scripts/run_sim_server.sh
```

In another terminal, run one profile:

```bash
./scripts/run_onnx_profile_runtime.sh crawl_safe
```

The runtime log filename includes the profile name, for example:

```text
/data/logs/runtime_crawl_safe_20260526_060102.mcap
```

## Compare logs

After collecting several logs:

```bash
./scripts/compare_policy_logs.sh
```

The comparison table ranks logs by mean reset-cycle duration. It also shows resets, policy record count, action magnitude, action scale, and motor velocity limit.

## Recommended experiment order

Run each profile for 20 to 30 seconds:

```bash
./scripts/run_onnx_profile_runtime.sh idle_debug
./scripts/run_onnx_profile_runtime.sh crawl_very_safe
./scripts/run_onnx_profile_runtime.sh crawl_safe
./scripts/run_onnx_profile_runtime.sh walk_cautious
```

Then compare:

```bash
./scripts/compare_policy_logs.sh
```

Use the best-surviving profile as the next baseline. Do not tune more than one variable at once until the logs show which profile survives longer.
