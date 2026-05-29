#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

profile="open_duck_forward"
candidate="neural_bc_teacher_live"
output_root="data/training_pipelines/neural_bc_teacher_live"
episodes="4"
steps_per_episode="1000"
command_x="0.15"
command_y="0.0"
command_yaw="0.0"
command_x_values=""
command_y_values=""
command_yaw_values=""
split_group_field="source_log"
force_profile=""

require_value() {
  if [ "$#" -lt 2 ]; then
    echo "Missing value for $1" >&2
    exit 2
  fi
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --profile) require_value "$@"; profile="$2"; shift 2 ;;
    --profile=*) profile="${1#*=}"; shift ;;
    --candidate) require_value "$@"; candidate="$2"; shift 2 ;;
    --candidate=*) candidate="${1#*=}"; shift ;;
    --output-root) require_value "$@"; output_root="$2"; shift 2 ;;
    --output-root=*) output_root="${1#*=}"; shift ;;
    --episodes) require_value "$@"; episodes="$2"; shift 2 ;;
    --episodes=*) episodes="${1#*=}"; shift ;;
    --steps-per-episode) require_value "$@"; steps_per_episode="$2"; shift 2 ;;
    --steps-per-episode=*) steps_per_episode="${1#*=}"; shift ;;
    --command-x) require_value "$@"; command_x="$2"; shift 2 ;;
    --command-x=*) command_x="${1#*=}"; shift ;;
    --command-y) require_value "$@"; command_y="$2"; shift 2 ;;
    --command-y=*) command_y="${1#*=}"; shift ;;
    --command-yaw) require_value "$@"; command_yaw="$2"; shift 2 ;;
    --command-yaw=*) command_yaw="${1#*=}"; shift ;;
    --command-x-values) require_value "$@"; command_x_values="$2"; shift 2 ;;
    --command-x-values=*) command_x_values="${1#*=}"; shift ;;
    --command-y-values) require_value "$@"; command_y_values="$2"; shift 2 ;;
    --command-y-values=*) command_y_values="${1#*=}"; shift ;;
    --command-yaw-values) require_value "$@"; command_yaw_values="$2"; shift 2 ;;
    --command-yaw-values=*) command_yaw_values="${1#*=}"; shift ;;
    --split-group-field) require_value "$@"; split_group_field="$2"; shift 2 ;;
    --split-group-field=*) split_group_field="${1#*=}"; shift ;;
    --force-profile) force_profile="--force-profile"; shift ;;
    --help|-h)
      cat <<EOF
Usage: $0 [options]

Collect teacher-policy rollouts from MuJoCo, prepare the dataset, summarize it,
train a neural behavior-clone policy, and create a Soridormi runtime profile.

Options:
  --profile NAME              teacher profile (default: open_duck_forward)
  --candidate NAME            generated profile name (default: neural_bc_teacher_live)
  --output-root PATH          artifact root (default: data/training_pipelines/neural_bc_teacher_live)
  --episodes N                teacher rollout episodes (default: 4)
  --steps-per-episode N       rollout length per episode (default: 1000)
  --command-x X               single forward command for collection (default: 0.15)
  --command-y Y               single lateral command for collection (default: 0.0)
  --command-yaw W             single yaw command for collection (default: 0.0)
  --command-x-values CSV      forward command grid, e.g. 0.00,0.05,0.10,0.15
  --command-y-values CSV      lateral command grid, e.g. -0.05,0.00,0.05
  --command-yaw-values CSV    yaw command grid, e.g. -0.20,0.00,0.20
  --split-group-field FIELD   prepare split grouping (default: source_log; use scenario_id for command holdout)
  --force-profile             overwrite configs/policies/<candidate>.yaml
EOF
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

dataset="${output_root}/dataset/teacher_policy_live.jsonl"
prepared="${output_root}/prepared"
train_dir="${output_root}/neural_bc"

mkdir -p "$(dirname "$dataset")" "$prepared" "$train_dir"

collect_args=(
  --profile "$profile"
  --output "$dataset"
  --episodes "$episodes"
  --steps-per-episode "$steps_per_episode"
  --command-x "$command_x"
  --command-y "$command_y"
  --command-yaw "$command_yaw"
)
if [ -n "$command_x_values" ]; then collect_args+=("--command-x-values=$command_x_values"); fi
if [ -n "$command_y_values" ]; then collect_args+=("--command-y-values=$command_y_values"); fi
if [ -n "$command_yaw_values" ]; then collect_args+=("--command-yaw-values=$command_yaw_values"); fi

./scripts/collect_teacher_dataset.sh "${collect_args[@]}"

./scripts/prepare_training_dataset.sh "$dataset" --output-dir "$prepared" --seed 123 --split-group-field "$split_group_field"
./scripts/summarize_training_dataset.sh "$prepared"
./scripts/train_neural_behavior_clone.sh "$prepared" \
  --output-dir "$train_dir" \
  --profile-name "$candidate" \
  --profile-template "$profile" \
  $force_profile

./scripts/check_policy_model.sh --profile "$candidate"

echo "Teacher-policy training pipeline complete. Candidate profile: $candidate"
