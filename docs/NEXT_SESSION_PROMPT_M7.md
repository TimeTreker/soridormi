# Next session prompt: start M7 hardware bridge

Copy this into a new ChatGPT/Claude session.

```text
We are working on Soridormi: https://github.com/TimeTreker/soridormi.git main.

Please first read these project-local files:
- CLAUDE.md
- LLM_CONTEXT.md
- AGENTS.md
- docs/PROJECT_SOP.md
- docs/M6_SUMMARY.md
- docs/ROADMAP_M4_M7.md
- docs/model_replacement_interface_m5.md

Current status:
- M4.13 is complete: Soridormi official-policy loop order and first-divergence analysis matched the official Open Duck policy path. The key bug was policy default-pose aliasing, not kp/kd.
- M5 is complete enough: policy profiles, contract export, model preflight, provider selection, profile creation, package/install/index workflows exist.
- M6 is complete enough: runtime logs can become supervised datasets; linear and neural behavior-cloning trainers exist; neural BC exports ONNX and profile YAML; bounded rollout, rollout comparison, failure diagnosis, relabeling, and retrain/promote iteration exist.

Important project principle:
Build the project backbone first. Avoid adding more reports, manifests, or helper wrappers unless they directly support the core path.

Project backbone:
official baseline -> Soridormi parity -> rollout evaluation -> data collection -> policy training -> policy deployment -> rollout comparison -> improvement loop -> hardware bridge.

Next milestone:
Start M7: hardware bridge / real robot backend.

Do not attempt walking on hardware immediately.
Start with safe, read-only and dry-run phases:
1. Define hardware backend safety contract.
2. Add an Open Duck Mini hardware backend skeleton implementing the same runtime API shape.
3. Add read-only state streaming into RobotState.
4. Add motor command dry-run mode that logs commands without moving motors.
5. Add watchdog, command timeout, joint/velocity/current limits, and emergency-stop plumbing.
6. Only then plan low-power single-joint and standing-pose tests.

Keep the invariant:
Same runtime.
Same policy interface.
Same RobotState.
Same MotorCommand.
Different backend.
```
