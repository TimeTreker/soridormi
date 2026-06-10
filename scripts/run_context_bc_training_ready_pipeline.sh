#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
  cat <<'USAGE'
Usage: ./scripts/run_context_bc_training_ready_pipeline.sh INPUT... [options]

Run the context BC M9 pipeline from prepare through prepared-gate and
training-ready report generation.

Options:
  --output-dir DIR              Prepared dataset output directory
  --prepared-gate-dir DIR       Prepared gate output directory
  --training-ready-dir DIR      Training-ready report output directory
  --scenario-gate PATH          Dataset scenario gate summary JSON path (required)
  --require-scenario SCENARIO   Required scenario id for prepared gate
  --report PATH                 Optional prepared dataset Markdown report path
  --contract PATH               BC contract JSON path
  --profile-name NAME           Candidate profile name for neural BC
  --input-mode MODE             Policy input mode for recommended train commands
  --linear-output-dir DIR       Linear BC output directory for recommended commands
  --neural-output-dir DIR       Neural BC output directory for recommended commands
  --json                        Emit machine-readable JSON from all stages
  -h, --help                    Show this help

Example:
  ./scripts/run_context_bc_training_ready_pipeline.sh \
    /data/training_datasets/context_bc/flat_walk_varied_speed_v1.context.jsonl \
    --scenario-gate artifacts/dataset_coverage/flat_walk_varied_speed_v1_gate/dataset_scenario_gate_summary.json \
    --require-scenario flat_walk_varied_speed_v1 \
    --output-dir /data/training_datasets/context_bc/prepared/flat_walk_varied_speed_v1 \
    --prepared-gate-dir artifacts/training/context_bc/prepared_gate/flat_walk_varied_speed_v1 \
    --training-ready-dir artifacts/training/context_bc/training_ready/flat_walk_varied_speed_v1 \
    --report artifacts/training/context_bc/prepared_flat_walk_varied_speed_v1.md \
    --json | python -m json.tool
USAGE
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

if [ "$#" -eq 0 ]; then
  usage >&2
  exit 2
fi

if [ ! -f .env ]; then
  ./scripts/setup_env.sh >/dev/null
fi

prepared_output_dir=''
prepared_gate_dir=''
training_ready_dir=''
scenario_gate=''
require_scenario=''
report_path=''
contract_path=''
profile_name='context_stage1_candidate'
input_mode='context_stage1_command'
linear_output_dir='/data/training_runs/context_stage1_candidate_linear_bc'
neural_output_dir='/data/training_runs/context_stage1_candidate_neural_bc'
json_flag=''

prepare_args=()
inputs=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --output-dir)
      prepared_output_dir="$2"; shift 2;;
    --prepared-gate-dir)
      prepared_gate_dir="$2"; shift 2;;
    --training-ready-dir)
      training_ready_dir="$2"; shift 2;;
    --scenario-gate)
      scenario_gate="$2"; shift 2;;
    --require-scenario)
      require_scenario="$2"; shift 2;;
    --report)
      report_path="$2"; shift 2;;
    --contract)
      contract_path="$2"; prepare_args+=("--contract" "$2"); shift 2;;
    --profile-name)
      profile_name="$2"; shift 2;;
    --input-mode)
      input_mode="$2"; shift 2;;
    --linear-output-dir)
      linear_output_dir="$2"; shift 2;;
    --neural-output-dir)
      neural_output_dir="$2"; shift 2;;
    --json)
      json_flag='--json'; shift;;
    --help|-h)
      usage
      exit 0;;
    --*)
      prepare_args+=("$1")
      if [[ "$1" != *=* && $# -gt 1 ]]; then
        prepare_args+=("$2")
        shift
      fi
      shift;;
    *)
      inputs+=("$1")
      shift;;
  esac
done

if [ "${#inputs[@]}" -eq 0 ]; then
  echo "error: missing INPUT JSONL paths" >&2
  usage >&2
  exit 2
fi

if [ -z "$scenario_gate" ]; then
  echo "error: --scenario-gate is required" >&2
  usage >&2
  exit 2
fi

if [ -z "$require_scenario" ]; then
  echo "error: --require-scenario is required" >&2
  usage >&2
  exit 2
fi

if [ -z "$prepared_output_dir" ]; then
  echo "error: --output-dir is required" >&2
  usage >&2
  exit 2
fi

prepared_manifest="$prepared_output_dir/prepared_manifest.json"

if [ -z "$prepared_gate_dir" ]; then
  prepared_gate_dir="artifacts/training/context_bc/prepared_gate/$(basename "$prepared_output_dir")"
fi

if [ -z "$training_ready_dir" ]; then
  training_ready_dir="artifacts/training/context_bc/training_ready/$(basename "$prepared_output_dir")"
fi

if [ -z "$report_path" ]; then
  report_path="artifacts/training/context_bc/prepared_$(basename "$prepared_output_dir").md"
fi

mkdir -p "$(dirname "$report_path")"

contract_args=()
if [ -n "$contract_path" ]; then
  contract_args+=("--contract" "$contract_path")
fi

# 1) prepare dataset
./scripts/prepare_context_bc_dataset.sh \
  "${inputs[@]}" \
  --output-dir "$prepared_output_dir" \
  --report "$report_path" \
  --json \
  "${prepare_args[@]}"

# 2) gate prepared dataset
./scripts/gate_context_bc_prepared_dataset.sh \
  "$prepared_manifest" \
  --require-scenario "$require_scenario" \
  --output-dir "$prepared_gate_dir" \
  --json \
  "${contract_args[@]}"

# 3) build training-ready report
./scripts/build_context_bc_training_ready_report.sh \
  "$prepared_manifest" \
  --scenario-gate "$scenario_gate" \
  --prepared-gate "$prepared_gate_dir/prepared_context_gate_report.json" \
  --profile-name "$profile_name" \
  --input-mode "$input_mode" \
  --linear-output-dir "$linear_output_dir" \
  --neural-output-dir "$neural_output_dir" \
  --output-dir "$training_ready_dir" \
  "${contract_args[@]}" \
  $json_flag

echo "Training-ready report written to: $training_ready_dir"
