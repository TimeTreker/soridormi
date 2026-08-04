#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
  cat <<'USAGE'
Usage: ./scripts/evaluate_scenario_suite.sh [options]

Run or plan a batch Soridormi MuJoCo scenario-evaluation suite.  By default the
suite includes registry-ready locomotion scenarios that scenario rollout evaluation can evaluate.  It
continues after individual scenario failures so the final suite report captures
all pass/fail results.

Options:
  --scenario SCENARIO              Scenario id to include; repeat or comma-separate.
  --status STATUS                  Scenario status to include; repeat or comma-separate.
  --family FAMILY                  Scenario family to include; repeat or comma-separate.
  --include-planned                Include planned locomotion scenarios unless --status is used.
  --scenario-manifest PATH         Scenario manifest path.
  --backend mujoco                 Required backend (default: mujoco).
  --profile PROFILE                Policy profile (default: open_duck_forward).
  --duration-s S                   Override every scenario duration.
  --steps N                        Override every scenario rollout step count.
  --control-hz HZ                  Control frequency (default: 50).
  --output-dir DIR                 Suite artifact directory.
  --log-prefix-root PREFIX         Runtime log prefix root (default: scenario_suite).
  --log-dir DIR                    Runtime log directory inside container (default: /data/logs).
  --skip-model-check               Pass through to scenario rollout runner.
  --dry-run-only                   Write suite/scenario run plans without launching runtime.
  --json                           Print suite JSON to stdout. Status goes to stderr.
  -h, --help                       Show this help.

Examples:
  ./scripts/evaluate_scenario_suite.sh \
    --backend mujoco \
    --profile open_duck_forward \
    --output-dir artifacts/scenario_eval/suite

  ./scripts/evaluate_scenario_suite.sh \
    --dry-run-only \
    --json | python -m json.tool
USAGE
}

scenario_manifest="configs/scenarios/open_duck_mini_v2_scenarios.json"
backend="mujoco"
profile="open_duck_forward"
control_hz="50"
output_dir="artifacts/scenario_eval/suite"
log_prefix_root="scenario_suite"
log_dir="/data/logs"
duration_s=""
steps=""
skip_model_check="0"
dry_run_only="0"
json_output="0"
include_planned="0"
scenarios=()
statuses=()
families=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --scenario)
      scenarios+=("${2:?--scenario requires a value}")
      shift 2
      ;;
    --scenario=*)
      scenarios+=("${1#*=}")
      shift
      ;;
    --status)
      statuses+=("${2:?--status requires a value}")
      shift 2
      ;;
    --status=*)
      statuses+=("${1#*=}")
      shift
      ;;
    --family)
      families+=("${2:?--family requires a value}")
      shift 2
      ;;
    --family=*)
      families+=("${1#*=}")
      shift
      ;;
    --include-planned)
      include_planned="1"
      shift
      ;;
    --scenario-manifest)
      scenario_manifest="${2:?--scenario-manifest requires a value}"
      shift 2
      ;;
    --scenario-manifest=*)
      scenario_manifest="${1#*=}"
      shift
      ;;
    --backend)
      backend="${2:?--backend requires a value}"
      shift 2
      ;;
    --backend=*)
      backend="${1#*=}"
      shift
      ;;
    --profile)
      profile="${2:?--profile requires a value}"
      shift 2
      ;;
    --profile=*)
      profile="${1#*=}"
      shift
      ;;
    --duration-s)
      duration_s="${2:?--duration-s requires a value}"
      shift 2
      ;;
    --duration-s=*)
      duration_s="${1#*=}"
      shift
      ;;
    --steps)
      steps="${2:?--steps requires a value}"
      shift 2
      ;;
    --steps=*)
      steps="${1#*=}"
      shift
      ;;
    --control-hz)
      control_hz="${2:?--control-hz requires a value}"
      shift 2
      ;;
    --control-hz=*)
      control_hz="${1#*=}"
      shift
      ;;
    --output-dir)
      output_dir="${2:?--output-dir requires a value}"
      shift 2
      ;;
    --output-dir=*)
      output_dir="${1#*=}"
      shift
      ;;
    --log-prefix-root)
      log_prefix_root="${2:?--log-prefix-root requires a value}"
      shift 2
      ;;
    --log-prefix-root=*)
      log_prefix_root="${1#*=}"
      shift
      ;;
    --log-dir)
      log_dir="${2:?--log-dir requires a value}"
      shift 2
      ;;
    --log-dir=*)
      log_dir="${1#*=}"
      shift
      ;;
    --skip-model-check)
      skip_model_check="1"
      shift
      ;;
    --dry-run-only)
      dry_run_only="1"
      shift
      ;;
    --json)
      json_output="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [ "${backend}" != "mujoco" ]; then
  echo "error: scenario suite evaluation is MuJoCo-only; use --backend mujoco" >&2
  exit 2
fi

export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${output_dir}"

status() {
  printf '%s\n' "$*" >&2
}

plan_args=(
  --scenario-manifest "${scenario_manifest}"
  --profile "${profile}"
  --control-hz "${control_hz}"
  --log-dir "${log_dir}"
  --log-prefix-root "${log_prefix_root}"
  --print-suite-plan
)
for item in "${scenarios[@]}"; do plan_args+=(--scenario "${item}"); done
for item in "${statuses[@]}"; do plan_args+=(--status "${item}"); done
for item in "${families[@]}"; do plan_args+=(--family "${item}"); done
if [ "${include_planned}" = "1" ]; then plan_args+=(--include-planned); fi
if [ -n "${duration_s}" ]; then plan_args+=(--duration-s "${duration_s}"); fi
if [ -n "${steps}" ]; then plan_args+=(--steps "${steps}"); fi

plan_json="$(python -m soridormi_runtime.scenario_suite_eval "${plan_args[@]}")"
printf '%s\n' "${plan_json}" > "${output_dir}/suite_plan.json"
mapfile -t scenario_ids < <(python -c 'import json,sys; print("\n".join(json.load(sys.stdin)["scenario_ids"]))' <<<"${plan_json}")

if [ "${#scenario_ids[@]}" -eq 0 ]; then
  echo "error: scenario suite selection is empty" >&2
  exit 1
fi

if [ "${dry_run_only}" = "1" ]; then
  if [ "${json_output}" = "1" ]; then
    printf '%s\n' "${plan_json}"
  else
    status "Scenario suite dry-run plan: ${output_dir}/suite_plan.json"
    status "Selected scenarios: ${scenario_ids[*]}"
  fi
  exit 0
fi

status "Soridormi scenario suite evaluation"
status "===================================="
status "Scenarios: ${scenario_ids[*]}"
status "Output dir: ${output_dir}"
status "This assumes MuJoCo is already running with:"
status "  ./scripts/run_sim_server.sh --backend mujoco --profile ${profile} --viewer --follow-camera"

report_paths=()
failed_runs=0
for scenario_id in "${scenario_ids[@]}"; do
  scenario_output_dir="${output_dir}/${scenario_id}"
  status ""
  status "--- Scenario: ${scenario_id} ---"
  run_args=(
    --scenario "${scenario_id}"
    --scenario-manifest "${scenario_manifest}"
    --backend "${backend}"
    --profile "${profile}"
    --control-hz "${control_hz}"
    --output-dir "${scenario_output_dir}"
    --log-prefix "${log_prefix_root}_${scenario_id}"
    --log-dir "${log_dir}"
  )
  if [ -n "${duration_s}" ]; then run_args+=(--duration-s "${duration_s}"); fi
  if [ -n "${steps}" ]; then run_args+=(--steps "${steps}"); fi
  if [ "${skip_model_check}" = "1" ]; then run_args+=(--skip-model-check); fi

  if [ "${json_output}" = "1" ]; then
    if ./scripts/evaluate_scenario_rollout.sh "${run_args[@]}" --json > "${scenario_output_dir}.stdout.json" 2>"${scenario_output_dir}.stderr.log"; then
      status "Scenario ${scenario_id}: runner exited 0"
    else
      failed_runs=$((failed_runs + 1))
      status "Scenario ${scenario_id}: runner exited nonzero; preserving report if available"
      cat "${scenario_output_dir}.stderr.log" >&2 || true
    fi
  else
    if ./scripts/evaluate_scenario_rollout.sh "${run_args[@]}"; then
      status "Scenario ${scenario_id}: runner exited 0"
    else
      failed_runs=$((failed_runs + 1))
      status "Scenario ${scenario_id}: runner exited nonzero; preserving report if available"
    fi
  fi
  if [ -f "${scenario_output_dir}/scenario_rollout_report.json" ]; then
    report_paths+=("${scenario_output_dir}/scenario_rollout_report.json")
  else
    status "warning: missing report for ${scenario_id}: ${scenario_output_dir}/scenario_rollout_report.json"
  fi
done

aggregate_args=(
  --reports "${report_paths[@]}"
  --json-output "${output_dir}/suite_summary.json"
  --output "${output_dir}/suite_summary.md"
)
for scenario_id in "${scenario_ids[@]}"; do
  aggregate_args+=(--expected-scenario "${scenario_id}")
done
if [ "${json_output}" = "1" ]; then
  aggregate_args+=(--json)
fi

# The suite aggregator returns nonzero when any scenario failed. Capture it so
# we can still leave JSON/Markdown artifacts for inspection.
aggregate_status=0
python -m soridormi_runtime.scenario_suite_eval "${aggregate_args[@]}" || aggregate_status=$?

status ""
status "Scenario suite JSON: ${output_dir}/suite_summary.json"
status "Scenario suite Markdown: ${output_dir}/suite_summary.md"
if [ "${failed_runs}" -gt 0 ]; then
  status "Scenario runner failures: ${failed_runs}"
fi
exit "${aggregate_status}"
