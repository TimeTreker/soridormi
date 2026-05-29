# Project status after M6

## Current milestone state

```text
M4: official policy parity in Soridormi — complete for checked trace window
M5: model replacement interface and packaging — complete enough
M6: sim-side policy learning loop — not complete until a trained candidate is validated in MuJoCo
M7: hardware bridge — blocked for walking until the M6 training/comparison gate passes
```

## M6 outcome

Soridormi has most of the sim-side policy-improvement backbone, but this should not be treated as a finished training result yet:

```text
teacher/default policy
→ supervised data and behavior cloning
→ neural ONNX replacement path
→ rollout comparison
→ walking reward
→ residual policy fine-tuning
→ residual ONNX runtime profile
→ default-vs-residual MuJoCo comparison
```

The practical next result should be produced from a fresh teacher-dataset or residual-RL run, followed by ONNX/profile validation and a default-vs-candidate MuJoCo rollout comparison. The direct teacher collection path is documented in `docs/M6_SIM_TRAINING_LOOP.md`.

## Remaining M6 work that can continue in parallel

M6 is not “globally solved.” Continue improving simulation policies while M7 hardware bridge begins:

```text
command-grid validation
longer rollouts
multiple seeds / repeated trials
stronger residual policy architecture
PPO/SAC or recurrent residual actor
terrain/perception/context-aware policy design
```

## M7 starting rule

M7 should not start by commanding motors aggressively.

Start with:

```text
read-only hardware state
motor-command dry-run
safety limits
watchdog
emergency stop
single-joint low-power tests
```

Only after that should standing and walking be attempted.
