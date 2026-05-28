# Next Session Prompt: Finish M6 Simulation Fine-Tuning

We should not move to M7 yet unless M6 simulation fine-tuning is proven.

Current state:

- M6 has the policy-improvement backbone.
- M6 is not complete as a simulation result until the residual fine-tuned policy exports to ONNX, runs in MuJoCo, and is compared against the default policy.
- The most recent residual training run produced a best score, but failed during ONNX export because `onnxscript` was missing from the training environment.

Next actions:

1. Apply the training dependency/export fix if not already applied.
2. Rebuild the training runtime image:

```bash
./scripts/build_runtime_training.sh
```

3. Start MuJoCo sim server.
4. Rerun residual training:

```bash
./scripts/train_residual_policy.sh open_duck_forward \
  --output-dir data/rl_finetune/residual_open_duck \
  --profile-name residual_open_duck \
  --iterations 5 \
  --population 16 \
  --steps-per-episode 300 \
  --residual-scale 0.05 \
  --force-profile
```

5. Validate the exported profile:

```bash
./scripts/check_policy_model.sh \
  --profile residual_open_duck \
  --require-provider CUDAExecutionProvider
```

6. Compare default vs residual policy:

```bash
./scripts/run_residual_finetune_comparison.sh residual_open_duck \
  --teacher-profile open_duck_forward \
  --steps 1000 \
  --require-provider CUDAExecutionProvider
```

Only after this should we call M6 simulation-side work complete enough to begin M7 hardware bridge.

Architecture rule to keep:

- Chromie is the Brain.
- Soridormi is the Cerebellum / Motor Executive.
