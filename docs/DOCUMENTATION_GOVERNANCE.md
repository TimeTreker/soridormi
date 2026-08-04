# Documentation and issue naming governance

## Purpose

Soridormi previously accumulated current status, roadmap, experiment history,
and LLM handoff material in multiple active files. Temporary sequence labels
then leaked into filenames, schemas, tests, CLI messages, and runtime contracts.

This contract prevents recurrence.

## Document classes

### Durable contract

Architecture, ownership, API, schema, safety, and validation rules. Durable
contracts describe semantics rather than an implementation sequence number.

### Current status

`docs/STATUS.md` is the only current-state summary. It records verified surface,
non-claims, blockers, and work order. It must not become a candidate log.

### Operator runbook

Commands and procedures needed to build, run, validate, recover, or commission
the system. A runbook does not redefine architecture or current project status.

### Evidence

Generated reports, traces, datasets, metrics, screenshots, and model outputs.
They belong under ignored `artifacts/` or `data/`.

### Historical record

Git history is the archive for replaced handoffs, old roadmaps, and obsolete
status documents. They are not retained as competing active files.

## Semantic issue names

Use names that state the capability or problem, for example:

```text
official-policy-parity
policy-model-replacement
context-dataset-preparation
clearance-readiness
scripted-social-skill-qualification
concurrent-cognitive-and-embodied-execution
task-event-cursor
hardware-state-readonly-bridge
```

Do not introduce:

- `M<number>` or similar sequence labels;
- numbered implementation stages;
- numbered implementation-step headings;
- project-sequence suffixes in test, module, script, config, or document names;
- generic legacy sequence compatibility fields.

Control-loop steps, gait phase, task lifecycle phase, schema versions, and model
versions remain valid technical concepts when they describe runtime semantics
rather than project sequencing.

## Rules

- Do not add another top-level handoff or project-status document.
- Do not repeat volatile candidate metrics across active documents.
- Do not claim completion from dry-run, preview, plan creation, offline loss, or
  partial evidence.
- Do not copy generated artifacts into `docs/`.
- When behavior changes, update code, tests, the owning contract, and
  `docs/STATUS.md` in one patch.
- Add a repository guard when a class of drift can be checked mechanically.

## Mechanical gate

```bash
python scripts/validate_repository_governance.py
```
