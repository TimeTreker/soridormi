#!/usr/bin/env bash
set -euo pipefail

# List and validate the Soridormi structured body-skill interface skill manifest.  This wrapper runs on the
# host and does not require Docker because the manifest is plain JSON.
export PYTHONPATH="${PYTHONPATH:-src}"
python -m soridormi_runtime.skill_manifest "$@"
