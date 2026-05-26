# M3.0 ONNX Policy Inspection

M3.0 is the safe starting point for policy work. It only inspects the ONNX model
and runs one dummy inference. It does not connect the policy to the robot control loop.

## Goal

Verify:

- ONNX policy file exists
- ONNX Runtime can load it
- CUDAExecutionProvider is available when supported
- input names/shapes are known
- output names/shapes are known
- dummy inference runs successfully

## Default policy path

The default Open Duck Mini v2 policy path is:

```bash
/workspaces/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx
```

You can override it with:

```bash
SORIDORMI_POLICY_PATH=/path/to/policy.onnx
```

## Run inspection

From the host:

```bash
./scripts/inspect_policy.sh
```

Or with an explicit policy path:

```bash
SORIDORMI_POLICY_PATH=/workspaces/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx \
./scripts/inspect_policy.sh
```

Expected output includes:

```text
Available providers:  [...]
Selected providers:   [...]
Inputs
Outputs
```

If CUDA is active, `Selected providers` should include:

```text
CUDAExecutionProvider
```

If CUDA is unavailable or incompatible, ONNX Runtime may fall back to:

```text
CPUExecutionProvider
```

## Run inside runtime container manually

```bash
./scripts/enter_runtime_dev.sh
python -m soridormi_runtime.inspect_onnx_policy \
  /workspaces/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx
```

For JSON output:

```bash
python -m soridormi_runtime.inspect_onnx_policy \
  /workspaces/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx \
  --json
```

## M3.0 success criteria

- `./scripts/inspect_policy.sh` runs successfully
- the policy input shape is known
- the policy output/action shape is known
- dummy inference succeeds
- selected execution provider is understood

## Next sections

After M3.0:

- M3.1 observation builder
- M3.2 action-to-MotorCommand mapper
- M3.3 `onnx_policy` runtime mode
- M3.4 policy logging
- M3.5 policy test in MuJoCo with auto-reset
