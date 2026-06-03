#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
  cat <<'USAGE'
Usage: ./scripts/evaluate_scenario_rollout.sh --scenario SCENARIO [options]

Evaluate one Soridormi scenario rollout.  If --log is provided, the command only
analyzes that JSONL file.  If --log is omitted, it derives a deterministic
primary-skill command from the scenario manifest, runs the skill through the
existing MuJoCo policy runtime, then analyzes the newest JSONL log.

Start MuJoCo first for live execution:
  ./scripts/run_sim_server.sh --backend mujoco --profile open_duck_forward --viewer --follow-camera

Options:
  --scenario ID                     Scenario id from configs/scenarios/open_duck_mini_v2_scenarios.json.
  --scenario-manifest PATH          Scenario manifest path (default: configs/scenarios/open_duck_mini_v2_scenarios.json).
  --log PATH                        Existing runtime JSONL log to analyze instead of running a skill.
  --backend mujoco                  Explicit backend selector for user-facing validation (default: mujoco).
  --profile PROFILE                 Policy profile for generated live rollout (default: open_duck_forward).
  --duration-s S                    Override scenario rollout duration.
  --steps N                         Override live rollout control steps.
  --control-hz HZ                   Control frequency for duration->steps and eval fallback (default: 50).
  --output-dir DIR                  Report directory (default: artifacts/scenario_eval/SCENARIO).
  --log-prefix PREFIX               Runtime log prefix (default: scenario_SCENARIO).
  --log-dir DIR                     Runtime log dir inside container (default: /data/logs).
  --skip-model-check                Forwarded to run_skill_in_sim.sh.
  --dry-run-only                    Print derived skill plan without launching MuJoCo runtime.
  --json                            Print JSON report instead of Markdown.
  --min-distance-m M                Progress threshold (default: 0.05).
  --min-mean-forward-speed-mps M    Forward-speed threshold (default: 0.02).
  --max-stuck-sample-ratio R        Stuck threshold (default: 0.40).
  --allow-fallen                    Do not fail when fall telemetry is detected.
  --min-touchdown-count N           Warning threshold (default: 4).
  --min-swing-clearance-m M         Warning threshold (default: 0.015).
  --max-low-clearance-ratio R       Warning threshold (default: 0.35).
  --require-foot-metrics            Fail if foot-position/contact metrics are missing.
  --min-base-z-m M                  Fall threshold (default: 0.12).
  --max-abs-roll-pitch-rad R        Fall threshold (default: 0.90).
  --contact-threshold C             Foot contact threshold (default: 0.5).
  -h, --help                        Show this help.

Examples:
  ./scripts/evaluate_scenario_rollout.sh \
    --scenario flat_walk_varied_speed_v1 \
    --backend mujoco \
    --profile open_duck_forward \
    --output-dir artifacts/scenario_eval/flat_walk_varied_speed_v1

  ./scripts/evaluate_scenario_rollout.sh \
    --scenario flat_walk_varied_speed_v1 \
    --log data/logs/scenario_flat_walk_varied_speed_v1.jsonl \
    --json | python -m json.tool
USAGE
}

scenario=""
scenario_manifest="configs/scenarios/open_duck_mini_v2_scenarios.json"
log_path=""
backend="mujoco"
profile="open_duck_forward"
duration_s=""
steps=""
control_hz="50"
output_dir=""
log_prefix=""
log_dir="/data/logs"
skip_model_check="0"
dry_run_only="0"
json_output="0"
min_distance_m="0.05"
min_mean_forward_speed_mps="0.02"
max_stuck_sample_ratio="0.40"
allow_fallen="0"
min_touchdown_count="4"
min_swing_clearance_m="0.015"
max_low_clearance_ratio="0.35"
require_foot_metrics="0"
min_base_z_m="0.12"
max_abs_roll_pitch_rad="0.90"
contact_threshold="0.5"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --scenario)
      scenario="${2:?--scenario requires a value}"
      shift 2
      ;;
    --scenario=*)
      scenario="${1#*=}"
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
    --log)
      log_path="${2:?--log requires a value}"
      shift 2
      ;;
    --log=*)
      log_path="${1#*=}"
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
    --log-prefix)
      log_prefix="${2:?--log-prefix requires a value}"
      shift 2
      ;;
    --log-prefix=*)
      log_prefix="${1#*=}"
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
    --min-distance-m)
      min_distance_m="${2:?--min-distance-m requires a value}"
      shift 2
      ;;
    --min-mean-forward-speed-mps)
      min_mean_forward_speed_mps="${2:?--min-mean-forward-speed-mps requires a value}"
      shift 2
      ;;
    --max-stuck-sample-ratio)
      max_stuck_sample_ratio="${2:?--max-stuck-sample-ratio requires a value}"
      shift 2
      ;;
    --allow-fallen)
      allow_fallen="1"
      shift
      ;;
    --min-touchdown-count)
      min_touchdown_count="${2:?--min-touchdown-count requires a value}"
      shift 2
      ;;
    --min-swing-clearance-m)
      min_swing_clearance_m="${2:?--min-swing-clearance-m requires a value}"
      shift 2
      ;;
    --max-low-clearance-ratio)
      max_low_clearance_ratio="${2:?--max-low-clearance-ratio requires a value}"
      shift 2
      ;;
    --require-foot-metrics)
      require_foot_metrics="1"
      shift
      ;;
    --min-base-z-m)
      min_base_z_m="${2:?--min-base-z-m requires a value}"
      shift 2
      ;;
    --max-abs-roll-pitch-rad)
      max_abs_roll_pitch_rad="${2:?--max-abs-roll-pitch-rad requires a value}"
      shift 2
      ;;
    --contact-threshold)
      contact_threshold="${2:?--contact-threshold requires a value}"
      shift 2
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

if [ -z "${scenario}" ]; then
  echo "error: --scenario is required" >&2
  usage >&2
  exit 2
fi
if [ "${backend}" != "mujoco" ]; then
  echo "error: M9A scenario rollout evaluation is MuJoCo-only; use --backend mujoco" >&2
  exit 2
fi
if [ -z "${output_dir}" ]; then
  output_dir="artifacts/scenario_eval/${scenario}"
fi
if [ -z "${log_prefix}" ]; then
  log_prefix="scenario_${scenario}"
fi

export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${output_dir}" data/logs

plan_args=(
  --scenario "${scenario}"
  --scenario-manifest "${scenario_manifest}"
  --profile "${profile}"
  --control-hz "${control_hz}"
  --log-prefix "${log_prefix}"
  --log-dir "${log_dir}"
  --print-run-plan
)
if [ -n "${duration_s}" ]; then
  plan_args+=(--duration-s "${duration_s}")
fi
if [ -n "${steps}" ]; then
  plan_args+=(--steps "${steps}")
fi

plan_json="$(python -m soridormi_runtime.scenario_rollout_eval "${plan_args[@]}")"
printf '%s\n' "${plan_json}" > "${output_dir}/scenario_run_plan.json"

skill_id="$(python -c 'import json,sys; print(json.load(sys.stdin)["skill_id"])' <<<"${plan_json}")"
skill_args="$(python -c 'import json,sys; print(json.dumps(json.load(sys.stdin)["args"], sort_keys=True))' <<<"${plan_json}")"
plan_steps="$(python -c 'import json,sys; print(json.load(sys.stdin)["steps"])' <<<"${plan_json}")"

if [ -z "${log_path}" ]; then
  echo "Soridormi scenario rollout evaluation"
  echo "======================================"
  echo "Scenario: ${scenario}"
  echo "Skill: ${skill_id}"
  echo "Skill args: ${skill_args}"
  echo "Profile: ${profile}"
  echo "Steps: ${plan_steps}"
  echo "Backend: ${backend}"
  echo "Run plan: ${output_dir}/scenario_run_plan.json"
  echo "This assumes MuJoCo is already running with:"
  echo "  ./scripts/run_sim_server.sh --backend mujoco --profile ${profile} --viewer --follow-camera"

  if [ "${dry_run_only}" = "1" ]; then
    echo "Dry-run only; not launching runtime."
    exit 0
  fi

  skill_run_args=(
    "${skill_id}"
    --args "${skill_args}"
    --profile "${profile}"
    --steps "${plan_steps}"
    --control-hz "${control_hz}"
    --log-format jsonl
    --log-prefix "${log_prefix}"
    --log-dir "${log_dir}"
  )
  if [ "${skip_model_check}" = "1" ]; then
    skill_run_args+=(--skip-model-check)
  fi
  ./scripts/run_skill_in_sim.sh "${skill_run_args[@]}"

  host_log_dir="data/logs"
  if [[ "${log_dir}" != "/data/logs" ]]; then
    # M9A supports the default container log mount directly.  Custom container
    # paths are still accepted for runtime execution, but callers should pass
    # --log explicitly if they choose a custom mount.
    echo "warning: custom --log-dir was used; falling back to data/logs host search" >&2
  fi
  log_path="$(find "${host_log_dir}" -type f -name "${log_prefix}*.jsonl" | sort | tail -1)"
  if [ -z "${log_path}" ]; then
    echo "error: no JSONL log found for prefix ${log_prefix} under ${host_log_dir}" >&2
    exit 1
  fi
fi

report_args=(
  --scenario "${scenario}"
  --scenario-manifest "${scenario_manifest}"
  --log "${log_path}"
  --fallback-control-hz "${control_hz}"
  --min-distance-m "${min_distance_m}"
  --min-mean-forward-speed-mps "${min_mean_forward_speed_mps}"
  --max-stuck-sample-ratio "${max_stuck_sample_ratio}"
  --min-touchdown-count "${min_touchdown_count}"
  --min-swing-clearance-m "${min_swing_clearance_m}"
  --max-low-clearance-ratio "${max_low_clearance_ratio}"
  --min-base-z-m "${min_base_z_m}"
  --max-abs-roll-pitch-rad "${max_abs_roll_pitch_rad}"
  --contact-threshold "${contact_threshold}"
  --output "${output_dir}/scenario_rollout_report.md"
  --json-output "${output_dir}/scenario_rollout_report.json"
)
if [ "${allow_fallen}" = "1" ]; then
  report_args+=(--allow-fallen)
fi
if [ "${require_foot_metrics}" = "1" ]; then
  report_args+=(--require-foot-metrics)
fi
if [ "${json_output}" = "1" ]; then
  report_args+=(--json)
fi

python -m soridormi_runtime.scenario_rollout_eval "${report_args[@]}"

echo "Scenario report JSON: ${output_dir}/scenario_rollout_report.json" >&2
echo "Scenario report Markdown: ${output_dir}/scenario_rollout_report.md" >&2
