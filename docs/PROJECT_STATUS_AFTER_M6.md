# Project status after M6

## Current milestone state

```text
M4: official policy parity in Soridormi — complete for checked trace window
M5: model replacement interface and packaging — complete enough
M6: sim-side command-conditioned free-walk loop — not complete until teacher and candidate policies are validated across a MuJoCo command suite
M7: hardware bridge — blocked for walking until the M6 commanded free-walk gate passes
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

The practical next result should be a commanded free-walk evaluation report for the default teacher, followed by command-distribution data collection, neural BC or residual candidate training, ONNX/profile validation, and teacher-vs-candidate MuJoCo comparison across the same command suite. The high-level plan is documented in `docs/SORIDORMI_FREE_WALK_PLAN.md`; the direct teacher collection path remains in `docs/M6_SIM_TRAINING_LOOP.md`.

## Remaining M6 work that can continue in parallel

M6 is not “globally solved.” Continue improving simulation policies before walking hardware begins:

```text
command-grid validation
command-switching validation
longer rollouts
multiple seeds / repeated trials
teacher-vs-neural-BC closed-loop comparisons
stronger residual policy architecture
PPO/SAC or recurrent residual actor, after BC and residual scaffolds are proven
terrain/perception/context-aware policy design, after flat-ground command walking is stable
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


## Orchestration boundary

Chromie/MCP/LLM orchestration is not the next Soridormi milestone. Soridormi should eventually expose safe robot capabilities, but the current repository still needs a stronger sim-side locomotion gate. Keep Soridormi focused on robot capability and safety:

```text
Soridormi owns: simulation, policy runtime, training/evaluation, action mapping, safety limits.
Chromie owns later: LLM routing, user speech/TTS, global MCP registry, multi-agent DAG planning.
```
