# Next Session Prompt

Copy this into a new ChatGPT/Claude session:

```text
We are working on Soridormi: https://github.com/TimeTreker/soridormi.git main.

Please first read these project-local files:
- CLAUDE.md
- LLM_CONTEXT.md
- AGENTS.md
- docs/LLM_HANDOFF_M4.md
- docs/ROADMAP_M4_M7.md

Context summary:
Soridormi is a sim-to-real Open Duck Mini v2 engineering platform. We are in M4, trying to make the official ONNX walking policy run correctly through Soridormi's own runtime. Official Open Duck baseline walks forward. Official motor-target replay through Soridormi backend matches official trajectory exactly. Soridormi ONNX wrapper reproduces official actions exactly when given official observations. Command and phase now match exactly. But Soridormi closed-loop policy still diverges after step 0, with remaining errors in accelerometer, gyro, contacts, and action/motor-target history; forward displacement is much smaller than official.

Please continue with M4.13: exact official loop-order parity and first-divergence analyzer. Do not switch to open-loop gait or training. Do not rewrite backend unless trace evidence proves it. The target is to identify and fix the first closed-loop divergence between official trace and Soridormi trace.
```
