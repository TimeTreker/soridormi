# Soridormi execution roadmap and milestone plan

> Principle: **Every milestone must produce verifiable artifacts and pass an
> explicit acceptance gate.**
>
> Work remains MuJoCo-first. A policy that trains successfully or has low
> offline error is not accepted until it demonstrates the required closed-loop
> behavior.

This document defines execution order, milestone gates, major risks, and the
immediate work queue. See `docs/SORIDORMI_TARGET_AND_ROADMAP.md` for the system
target, brain/cerebellum boundary, and detailed evidence for current policy
candidates.

## Overview

```text
M0-M3 Runtime foundations
      |
      v
M4 Official parity
      |
      +--> M5 Replaceable policy interface
      |       |
      |       v
      +--> M6 Training/evaluation backbone
              |
              v
M7 Structured body skills --> M8 Scenarios and social behavior
              |
              v
M9 Context BC data pipeline --> M10 Runtime context policy
                                      |
                                      v
                               M11 Generalization
                                      |
                                      v
                               M12 Targeted BC/RL improvement
                                      |
                                      v
                               M13 Hardware bridge
                                      |
                                      v
                               M14 Chromie integration
```

Current position:

```text
M4-M9: substantially complete
M10: functional Stage 1 candidate available
Current blocker: low swing-foot clearance and limited held-out generalization
Hardware: intentionally not started
```

---

## M0-M3 - Runtime foundations [Completed]

**Goal:** Establish stable robot, API, simulator, observation, action, and
controller contracts.

- [x] Define `RobotState` and `MotorCommand`
- [x] Load Open Duck Mini v2 in MuJoCo
- [x] Implement simulator API and runtime loop
- [x] Define the 101D official-policy observation
- [x] Map 14D policy actions to actuator commands
- [x] Add reset, fixed-base, logging, and debugging modes
- [x] Preserve backend separation for future hardware

**Gate G3:** The same runtime interfaces can operate against MuJoCo and support
a future hardware backend without changing policy semantics.

---

## M4 - Official Open Duck policy parity [Completed]

**Goal:** Reproduce the official Open Duck locomotion policy through Soridormi.

- [x] Port official ONNX inference behavior
- [x] Match observation construction
- [x] Match phase and command handling
- [x] Match action-to-motor mapping
- [x] Add official baseline, replay, and trace comparison
- [x] Add first-divergence analysis
- [x] Validate synchronous policy stepping

**Gate G4:**

- Official reference files are required and checked.
- Observation, action, loop ordering, and reset behavior show no unexplained
  meaningful divergence.
- Official baseline and comparison tools remain available.

> Official parity is the permanent trusted baseline. Later models may replace
> the policy, but not the parity evidence.

---

## M5 - Replaceable policy interface [Completed]

**Goal:** Replace compatible models without rewriting runtime code.

- [x] Define policy profiles
- [x] Validate input/output shapes
- [x] Validate ONNX providers
- [x] Export policy contracts and manifests
- [x] Package, install, restore, and promote candidates
- [x] Add profile acceptance checks
- [x] Preserve the official model as a fallback

**Gate G5:** A compatible policy package can be validated and loaded through
the existing runtime using only its declared profile and contract.

---

## M6 - Training and evaluation backbone [Completed]

**Goal:** Build a reproducible policy-improvement loop.

- [x] Export policy-supervision datasets
- [x] Validate observations and actions
- [x] Produce deterministic train/validation/test splits
- [x] Train linear BC baselines
- [x] Train neural BC candidates
- [x] Evaluate candidates offline
- [x] Evaluate candidates in MuJoCo
- [x] Compare candidates with the official teacher
- [x] Support relabeling and iterative retraining

**Gate G6:**

```text
dataset valid
training reproducible
model/profile contract valid
bounded MuJoCo rollout completed
candidate compared with teacher
```

> Offline action error alone is never a promotion criterion.

---

## M7 - Structured body-skill interface [Completed]

**Goal:** Expose bounded body capabilities to higher-level systems.

- [x] Define named skills
- [x] Validate skill arguments and availability
- [x] Translate locomotion skills into structured commands
- [x] Reject unsupported or unsafe requests
- [x] Prevent planners from sending raw joint actions
- [x] Export a capability manifest

**Gate G7:** Every externally callable body action is represented by a bounded
skill or structured context request.

---

## M8 - Scenario and interaction layer [Completed]

**Goal:** Test body behavior in meaningful scenarios rather than isolated
commands.

- [x] Define a scenario registry and curriculum
- [x] Add single-scenario rollout evaluation
- [x] Add batch scenario-suite evaluation
- [x] Add scenario-specific acceptance thresholds
- [x] Implement bounded head and social skills
- [x] Add social-skill readiness and acceptance reports
- [x] Add structured look-target input

**Gate G8:** A scenario produces reproducible evidence containing commands,
outcomes, failures, and acceptance results.

---

## M9 - Context BC data pipeline [Current: core complete]

**Goal:** Produce auditable, scenario-aware training datasets.

Policy stage:

```text
observation[101]
+ desired_command(vx_mps, vy_mps, yaw_radps)
-> action_14d
```

- [x] Collect scenario-aware teacher rows
- [x] Report dataset coverage
- [x] Gate required scenarios and command ranges
- [x] Export context BC rows
- [x] Validate the BC training contract
- [x] Split by rollout group to prevent leakage
- [x] Gate prepared train/validation/test datasets
- [x] Reject empty or invalid upstream outputs
- [x] Add a training-ready manifest and report command
- [ ] Generate and retain a training-ready report for the current
  three-scenario dataset

**Gate G9:**

- All required scenarios meet coverage thresholds.
- Train, validation, and test splits are non-empty.
- No rollout group appears in multiple splits.
- Contracts and file hashes are recorded.
- Both dataset gates pass.
- A training-ready manifest is generated before training.

---

## M10 - Runtime context policy [Current: clearance blocker identified]

**Goal:** Run context-conditioned policies through the production runtime path.

- [x] Define `context_stage1_command`
- [x] Build 104D policy inputs
- [x] Require `[1, 104]` profile/model contracts
- [x] Export runnable context-policy ONNX profiles
- [x] Train flat-only candidates
- [x] Diagnose low-speed and curve failures
- [x] Train a three-scenario candidate
- [x] Pass the flat/start-stop/curve suite
- [x] Diagnose low swing-foot clearance ✗ FAILS
  - flat_walk_varied_speed_v1: 0.0102m (need +0.0048m)
  - start_stop_velocity_ramp_v1: 0.0076m (need +0.0074m)
  - curve_turn_walk_v1: 0.0063m (need +0.0087m)
- [ ] Perform follow-camera visual inspection
- [x] Define clearance-focused promotion thresholds
- [x] Add threshold-aligned M10 clearance readiness report
- [x] Add an M10 evidence package and visual-review template
- [x] Add scenario-suite comparison against the official teacher
- [x] Re-evaluate the current candidate against the official teacher
  - flat distance/speed ratio: 0.978
  - start-stop distance/speed ratio: 0.939
  - curve distance/speed ratio: 1.242
  - no falls; stuck-ratio regression within 0.10
- [x] Test clearance-aware constant residual bias
  - result: rejected; clearance regressed or remained unchanged
  - flat: 0.01023m -> 0.00943m
  - start-stop: 0.00759m -> 0.00759m
  - curve: 0.00632m -> 0.00599m
- [ ] Implement a bounded phase/state-conditioned residual policy
- [ ] **DECISION REQUIRED:** Clearance refinement or experimental M10.0?

**Gate G10:**

```text
profile/model contract: PASS
bounded rollout: PASS
required scenario suite: PASS
fall/reset limits: PASS
foot-clearance threshold: ✗ FAIL (all scenarios < 0.015m)
visual inspection: PENDING
teacher comparison: PASS (relative behavior only; does not replace clearance)
```

> **Current status:** The candidate passes 3/3 scenarios but does not pass G10
> due to swing clearance deficit across all three scenarios.
>
> **Clearance readiness:** Generate with `./scripts/analyze_m10_clearance_readiness.sh --profile-name context_stage1_three_scenario_10ep_e80 --output-dir artifacts/m10_clearance_readiness/context_stage1_three_scenario_10ep_e80`.
>
> **Decision path:**
> - **Option 1 (Recommended):** Pursue clearance-focused refinement before M10.0 release
> - **Option 2:** Accept as experimental M10.0, document limitation, refine in M11

---

## M11 - Broader locomotion generalization

**Goal:** Generalize beyond the initial three training scenarios.

- [ ] Add lateral walking
- [ ] Expand turn and curve ranges
- [ ] Add randomized start/stop transitions
- [ ] Add rough-ground scenarios
- [ ] Add slopes
- [ ] Add low obstacles
- [ ] Add command-transition stress tests
- [ ] Create held-out randomized evaluation suites
- [ ] Add bounded task context
- [ ] Add bounded environment context
- [ ] Evaluate short-history input only when evidence supports it

**Gate G11:**

- Training and held-out scenarios are explicitly separated.
- Required held-out suites pass across multiple fixed seeds.
- No major regression occurs on the original three-scenario suite.
- The candidate remains within fall, drift, yaw, and clearance limits.

---

## M12 - Targeted policy improvement

**Goal:** Improve measured weaknesses without hiding failures behind tuning.

Possible methods:

```text
targeted recollection
teacher relabeling
dataset balancing
DAgger-style recovery data
residual policy
RL fine-tuning
```

- [ ] Rank failure modes from M11 evidence
- [ ] Select one measurable weakness
- [ ] Establish a frozen baseline
- [ ] Collect targeted data
- [ ] Train a BC or residual candidate
- [ ] Compare with the frozen baseline
- [ ] Reject improvements that damage safety or generalization

**Gate G12:** The candidate improves its declared target metric and does not
regress required safety or held-out suites.

> RL is introduced only after the BC baseline and failure metric are clear.

---

## M13 - Hardware bridge

**Goal:** Transfer the validated runtime to the physical robot in controlled
stages.

```text
H0 read-only state
 -> H1 motor-command dry-run
 -> H2 safety/watchdog
 -> H3 low-power single joint
 -> H4 standing pose
 -> H5 tethered low-speed locomotion
```

- [ ] Implement hardware state streaming
- [ ] Verify joint ordering, signs, limits, and units
- [ ] Implement command dry-run logging
- [ ] Add emergency stop
- [ ] Add command timeout and watchdog
- [ ] Add voltage, thermal, and communication checks
- [ ] Perform low-power single-joint validation
- [ ] Perform tethered standing
- [ ] Perform tethered low-speed walking

**Gate G13:**

- Every earlier hardware phase passes before actuator authority increases.
- Commands remain inside validated limits.
- Loss of communication leads to a safe stop.
- Logs are retained for every actuator test.
- MuJoCo acceptance remains green.

> No actuator command is permitted during read-only or dry-run phases.

---

## M14 - Chromie integration

**Goal:** Connect the Chromie brain to Soridormi through structured skills.

- [ ] Publish Soridormi capability and availability data
- [ ] Define versioned request/response schemas
- [ ] Add execution status and failure reasons
- [ ] Add cancellation and stop semantics
- [ ] Validate integration in MuJoCo
- [ ] Add bounded locomotion requests
- [ ] Add bounded social and attention requests
- [ ] Add safety refusal tests
- [ ] Keep raw joint and `action_14d` interfaces inaccessible

**Gate G14:** Chromie can select, monitor, cancel, and recover from body skills
without producing low-level motor or policy actions.

---

## Critical path

```text
M9 training-ready evidence
  -> M10 clearance refinement
  -> M11 held-out generalization
  -> M12 targeted improvement
  -> M13 staged hardware transfer
  -> M14 physical Chromie integration
```

M14 API work may begin in MuJoCo before M13 finishes, but physical integration
depends on the hardware safety gate.

## Highest risks

| Risk | Impact | Mitigation |
|---|---|---|
| Low foot clearance | Trips and obstacle failures | Add clearance metrics, visual inspection, targeted data, and scenarios |
| Dataset leakage | Misleading validation results | Preserve rollout-group splits and enforce the prepared dataset gate |
| Narrow scenario coverage | Policy overfits nominal walking | Require held-out randomized suites before promotion |
| Offline/online mismatch | Low-MAE model fails in closed loop | Make MuJoCo rollout gates mandatory |
| Simulator contention | Empty or corrupted datasets | Preserve collector-owned simulator lifecycle |
| Premature hardware testing | Damage or unsafe motion | Enforce staged H0-H5 hardware gates |
| Brain/body boundary erosion | Unsafe planner-generated control | Accept only structured skills and bounded context |

## Milestone-to-gate map

| Milestone | Gate | Primary evidence |
|---|---|---|
| M4 | G4 | Official/Soridormi parity traces and first-divergence reports |
| M5 | G5 | Policy contract, profile validation, package checks |
| M6 | G6 | Dataset, training, offline evaluation, MuJoCo rollout |
| M7 | G7 | Skill manifest, validation, refusal tests |
| M8 | G8 | Scenario and social-skill acceptance reports |
| M9 | G9 | Coverage gate, prepared gate, training-ready manifest |
| M10 | G10 | Context profile contract, scenario suite, clearance evidence |
| M11 | G11 | Held-out multi-seed scenario suites |
| M12 | G12 | Frozen-baseline comparison and non-regression evidence |
| M13 | G13 | H0-H5 hardware safety and execution logs |
| M14 | G14 | Structured integration and safety-refusal tests |

## Immediate execution plan

1. Generate the M9 training-ready report for the current three-scenario
   dataset.
2. Run the current candidate with `--viewer --follow-camera`.
3. Record swing-clearance evidence for all three scenarios.
4. Fill the M10 visual-review template and rebuild the evidence package.
5. Replace the rejected constant residual bias with a bounded
   phase/state-conditioned residual actor.
6. Train the actor with explicit swing-clearance reward across multiple command
   conditions.
7. Pass the quantitative clearance readiness gate without regressing the
   original scenario suite.
8. Compare the new candidate suite against the official teacher with
   `compare_m10_teacher_suite.sh`.
9. Begin M11 held-out scenario development only after G10 passes.

## Project success criteria

Soridormi succeeds when:

- policies are replaceable through stable contracts;
- datasets and training runs are reproducible;
- locomotion works across held-out scenarios;
- failures are visible rather than hidden by tuning;
- simulation and hardware share the same runtime concepts;
- hardware rollout is staged and reversible; and
- Chromie controls the body exclusively through safe structured skills.
