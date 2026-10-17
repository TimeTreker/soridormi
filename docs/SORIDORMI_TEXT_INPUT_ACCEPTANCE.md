# Soridormi text-input acceptance boundary

Raw user text belongs to Chromie. Soridormi should only receive structured body
tasks, skills, or safety calls after Chromie has understood the text and selected
an exact capability.

The fixture is:

```text
task_acceptance_cases/chromie_text_input_acceptance.yaml
```

It covers natural text inputs without test-only `--expect-skill` arguments:

- ordinary English and Chinese body commands;
- compound commands that must preserve every action, such as walk then blink;
- conversational questions that should not submit a body task;
- social replies where a small body response is optional;
- planning requests that Chromie answers in conversation, with no required
  "thinking" gesture or interim acknowledgement;
- object delivery, which is live in simulation through the named skill
  `acquire_and_deliver_resource` (Chromie must not say it is unavailable);
- unsafe physical requests, which Chromie understands and its Value Judgment
  refuses, with no Soridormi body task.

Each case states the observable Chromie outcome (`body_capability`,
`conversation_only`, `social_response`, or `value_judgment_refusal`). It does
not prescribe pre-semantic route labels, an emergency content filter, or stage
names: Chromie's Gateway handles only stop/cancel/emergency/silence, and harmful
requests are refused by Value Judgment after understanding, not by a Soridormi
or Gateway route filter.

Validate it with:

```bash
PYTHONPATH=src pytest -q tests/test_chromie_text_input_acceptance.py
```

This test replays only the structured Soridormi body tasks and plans the
delivery skill. It does not parse language or synthesize speech; those remain
Chromie responsibilities. The task API's `deliver_object` type remains a
contract-only placeholder for non-mocked delivery with a real manipulator; it is
not Chromie's delivery path.
