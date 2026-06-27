# Soridormi text-input acceptance boundary

Raw user text belongs to Chromie. Soridormi should only receive structured body
tasks, skills, or safety calls after Chromie has routed the text.

The fixture is:

```text
task_acceptance_cases/chromie_text_input_acceptance.yaml
```

It covers natural text inputs without test-only `--expect-skill` arguments:

- ordinary English and Chinese body commands;
- compound commands that must preserve every action, such as walk then blink;
- conversational questions that should not submit a body task;
- social replies where a small body response is optional;
- deep-thought requests with an interim attention gesture;
- blocked capabilities such as object delivery;
- unsafe physical requests that should be refused before motion.

Validate it with:

```bash
PYTHONPATH=src pytest -q tests/test_chromie_text_input_acceptance.py
```

This test replays only the structured Soridormi body tasks. It does not parse
language or synthesize speech; those remain Chromie responsibilities.
