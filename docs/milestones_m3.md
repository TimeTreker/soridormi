# M3 Milestones

## M3.0: ONNX policy inspection

Inspect `BEST_WALK_ONNX_2.onnx`, verify input and output shapes, and run dummy inference.

## M3.1: Observation builder

Build the 101-dimensional observation vector expected by the Open Duck Mini v2 ONNX policy.

## M3.2: Persistent ONNX policy wrapper

Load the ONNX model once and run repeated inference with action history.

## M3.3: Action-to-MotorCommand mapper

Convert the 14D policy output into Soridormi `MotorCommand` targets.

## M3.4: Experimental ONNX policy runtime mode

Add explicit opt-in runtime mode:

```bash
SORIDORMI_RUNTIME_MODE=onnx_policy
```

## M3.5: Command, phase, and speed limiting

Add dynamic command vector, gait phase oscillator, and motor target speed limiting:

```bash
./scripts/run_onnx_walk_runtime.sh
```

This is the first policy mode expected to produce dynamic motor targets. Stable walking is not required yet.

## M3.6: Policy debug logging

Add ONNX-policy-specific runtime log payloads for closed-loop fall debugging:

```text
/soridormi/policy_action
/soridormi/policy_debug
/soridormi/policy_observation_stats
```

`inspect_latest_log.sh` should now show these topics and print a compact latest policy snapshot.

This milestone does not attempt to tune walking yet. It makes repeated falls diagnosable.

## M3.7 Policy log analysis

Status: added.

Adds `soridormi_runtime.analyze_policy_log` and
`scripts/analyze_latest_policy_log.sh` to summarize policy debug MCAP/JSONL logs,
detect robot-time reset cycles, and report action/motor/observation statistics.
Use this before tuning walking parameters after repeated falls.

