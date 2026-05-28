# Project status after M6

## Current milestone state

```text
M4: official policy parity in Soridormi — complete for checked trace window
M5: model replacement interface and packaging — complete enough
M6: sim-side policy learning loop — complete enough for forward-walk case
M7: hardware bridge — next focus
```

## M6 outcome

Soridormi now has a full sim-side policy-improvement loop:

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

The successful forward-walk comparison showed the residual policy completed the same 1000-step rollout as the default policy with zero resets and improved measured forward/lateral metrics.

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
