#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
  cat <<'EOF'
Usage: ./scripts/train_clearance_residual_policy.sh [options] [-- extra trainer args...]

Run the documented clearance-focused residual policy training recipe. The
simulator server must already be running.

Defaults warm-start from the strongest retained clearance residual candidate,
preserve one flat command, emphasize start/stop plus turning sequences, use
per-step objective normalization, keep worst-case pressure on the weakest
objective, and write a final per-objective score breakdown.

Options:
  --teacher-profile PROFILE       Teacher profile (default: context_command_three_scenario_10ep_e80).
  --profile-name NAME             Generated residual profile name (default: clearance_gap_sequence_s83).
  --output-dir DIR                Training output dir (default: /data/rl_finetune/PROFILE_NAME).
  --initial-checkpoint PATH       Warm-start checkpoint (default: /data/rl_finetune/clearance_command_state_mlp_cem4x14_s79/residual_policy.pt).
  --iterations N                  CEM iterations (default: 4).
  --population N                  CEM population (default: 14).
  --steps-per-episode N           Simulator steps per objective (default: 300).
  --seed N                        Random seed (default: 83).
  --residual-scale S              Runtime residual scale (default: 0.1, matching the retained warm-start checkpoint).
  --worst-case-score-weight W     Weakest-objective blend weight (default: 0.35).
  --episodic-clearance-gap-weight W
                                  Gap penalty for below-target clearance (default: 1.0).
  --episodic-clearance-quantile Q
                                  Lower-tail clearance quantile for quantile gap penalty (default: 0.25).
  --episodic-clearance-quantile-gap-weight W
                                  Gap penalty at the configured clearance quantile (default: 0.0).
  --force-profile                 Overwrite an existing generated profile.
  --dry-run                       Print resolved trainer config without connecting to sim.
  -h, --help                      Show this help.

Anything after "--" is forwarded to train_residual_policy.sh.
EOF
}

teacher_profile="context_command_three_scenario_10ep_e80"
profile_name="clearance_gap_sequence_s83"
output_dir=""
initial_checkpoint="/data/rl_finetune/clearance_command_state_mlp_cem4x14_s79/residual_policy.pt"
iterations="4"
population="14"
steps_per_episode="300"
seed="83"
residual_scale="0.1"
worst_case_score_weight="0.35"
episodic_clearance_gap_weight="1.0"
episodic_clearance_quantile="0.25"
episodic_clearance_quantile_gap_weight="0.0"
force_profile="0"
dry_run="0"
extra_args=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --teacher-profile)
      teacher_profile="${2:?--teacher-profile requires a value}"
      shift 2
      ;;
    --profile-name)
      profile_name="${2:?--profile-name requires a value}"
      shift 2
      ;;
    --output-dir)
      output_dir="${2:?--output-dir requires a value}"
      shift 2
      ;;
    --initial-checkpoint)
      initial_checkpoint="${2:?--initial-checkpoint requires a value}"
      shift 2
      ;;
    --iterations)
      iterations="${2:?--iterations requires a value}"
      shift 2
      ;;
    --population)
      population="${2:?--population requires a value}"
      shift 2
      ;;
    --steps-per-episode)
      steps_per_episode="${2:?--steps-per-episode requires a value}"
      shift 2
      ;;
    --seed)
      seed="${2:?--seed requires a value}"
      shift 2
      ;;
    --residual-scale)
      residual_scale="${2:?--residual-scale requires a value}"
      shift 2
      ;;
    --worst-case-score-weight)
      worst_case_score_weight="${2:?--worst-case-score-weight requires a value}"
      shift 2
      ;;
    --episodic-clearance-gap-weight)
      episodic_clearance_gap_weight="${2:?--episodic-clearance-gap-weight requires a value}"
      shift 2
      ;;
    --episodic-clearance-quantile)
      episodic_clearance_quantile="${2:?--episodic-clearance-quantile requires a value}"
      shift 2
      ;;
    --episodic-clearance-quantile-gap-weight)
      episodic_clearance_quantile_gap_weight="${2:?--episodic-clearance-quantile-gap-weight requires a value}"
      shift 2
      ;;
    --force-profile)
      force_profile="1"
      shift
      ;;
    --dry-run)
      dry_run="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      extra_args+=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [ -z "${output_dir}" ]; then
  output_dir="/data/rl_finetune/${profile_name}"
fi

cmd=(
  ./scripts/train_residual_policy.sh "${teacher_profile}"
  --output-dir "${output_dir}"
  --profile-name "${profile_name}"
  --actor-kind command_state_mlp
  --initial-checkpoint "${initial_checkpoint}"
  --training-command "0.125,0,0,1.0"
  --training-sequence "2.5|0,0,0,50;0.06,0,0,100;0,0,0,50"
  --training-sequence "3.0|0.09,0,0,50;0.09,0,0.12,150;0.09,0,0,100"
  --episodic-clearance-weight 5
  --episodic-low-clearance-penalty-weight 4
  --episodic-clearance-gap-weight "${episodic_clearance_gap_weight}"
  --episodic-clearance-quantile "${episodic_clearance_quantile}"
  --episodic-clearance-quantile-gap-weight "${episodic_clearance_quantile_gap_weight}"
  --worst-case-score-weight "${worst_case_score_weight}"
  --score-normalization per_step
  --iterations "${iterations}"
  --population "${population}"
  --steps-per-episode "${steps_per_episode}"
  --seed "${seed}"
  --residual-scale "${residual_scale}"
  --final-score-breakdown
)

if [ "${force_profile}" = "1" ]; then
  cmd+=(--force-profile)
fi
if [ "${dry_run}" = "1" ]; then
  cmd+=(--dry-run)
fi
cmd+=("${extra_args[@]}")

echo "Soridormi clearance residual training"
echo "====================================="
echo "Teacher profile: ${teacher_profile}"
echo "Profile name:    ${profile_name}"
echo "Output dir:      ${output_dir}"
echo "Warm start:      ${initial_checkpoint}"
echo

exec "${cmd[@]}"
