#!/usr/bin/env bash
# Train and evaluate the paper arms: internal test set, then VinDr-CXR.
#   ./run_arms.sh                       # every arm, in order
#   ./run_arms.sh no_reasoning no_marker  # only these
# Env:
#   PYTHON  python to use                   (default: python)
#   NPROC       GPUs for SFT training (accelerate)  (default: 1; paper: 4)
#   GRPO_NPROC  GPUs for GRPO training              (default: 2, as in the paper)
#   VINDR   1 = also eval on VinDr          (default: 1; needs ../vindr/ data)
# Arms whose adapter or metrics.json already exist are skipped.
set -euo pipefail
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python}
NPROC=${NPROC:-1}
GRPO_NPROC=${GRPO_NPROC:-2}
VINDR=${VINDR:-1}
RUNS=../runs

MARKER_TYPES="[marker_existence,marker_counting,marker_grounding,marker_grounding_discriminative,\
marker_localization,marker_localization_region_conditioned,marker_localization_quadrant_conditioned,\
marker_localization_negative,marker_count_region_conditioned,marker_count_quadrant_conditioned,\
region_conditioned_marker_counting,coordinate_marker_existence,coordinate_to_label,\
coordinate_to_label_negative,category_discrimination,region_to_object]"

# arm | train script | config | reasoning (true/false) | model (sft/grpo/base) | extra --set
ARMS=(
  "baseline||configs/sft.yaml|true|base|"
  "no_reasoning|01|configs/sft.yaml|false|sft|"
  "full|01|configs/sft.yaml|true|sft|"
  "no_marker|01|configs/sft.yaml|false|sft|data.exclude_question_types=$MARKER_TYPES"
  "marker25|01|configs/sft_marker25.yaml|false|sft|"
  "marker50|01|configs/sft_marker50.yaml|false|sft|"
  "grpo_box_no_reasoning|02|configs/grpo.yaml|false|grpo|"
  "grpo_box_base_reasoning|02|configs/grpo_from_base.yaml|true|sft|"
)

launch() {  # launch <n GPUs> <script> ...; multi-GPU training goes through accelerate
  local n=$1; shift
  if [ "$n" -gt 1 ]; then
    "$PYTHON" -m accelerate.commands.launch --multi_gpu --num_processes="$n" --mixed_precision=bf16 "$@"
  else
    "$PYTHON" "$@"
  fi
}

run_arm() {
  IFS='|' read -r name script config reasoning kind extra <<<"$1"
  local run=$RUNS/$name sets=(--set "run_name=$name")
  [ -n "$extra" ] && sets+=(--set "$extra")
  [ "$script" = 01 ] && sets+=(--set "data.use_reasoning_for_boxes=$reasoning"
                                --set "data.use_reasoning_for_coord_reading=$reasoning")

  if [ -n "$script" ] && [ ! -f "$run/adapter/adapter_config.json" ]; then
    echo "== train $name"
    if [ "$script" = 01 ]; then launch "$NPROC" 01_finetune_markercxr.py train --config "$config" "${sets[@]}"
    else launch "$GRPO_NPROC" 02_grpo_markercxr.py train --config "$config" --set "run_name=$name"; fi
  fi

  local model=(--adapter "$run/adapter") iout=$run/eval/finetuned
  case $kind in
    base) model=(--baseline); iout=$RUNS/baseline/eval ;;
    grpo) model=(--set "model.model_id=$run/merged_base" --adapter "$run/adapter") ;;
  esac
  local flags=(--set "data.use_reasoning_for_boxes=$reasoning" --set "data.use_reasoning_for_coord_reading=$reasoning")

  if [ ! -f "$iout/metrics.json" ]; then
    echo "== internal eval $name"
    "$PYTHON" 01_finetune_markercxr.py eval --config configs/sft.yaml "${flags[@]}" "${model[@]}" --out "$iout"
  fi
  if [ "$VINDR" = 1 ] && [ ! -f "$RUNS/vindr/$name/metrics.json" ]; then
    echo "== VinDr eval $name"
    "$PYTHON" 01_finetune_markercxr.py eval --config configs/vindr_eval.yaml "${flags[@]}" "${model[@]}" \
      --out "$RUNS/vindr/$name"
  fi
}

selected=("$@")
[ ${#selected[@]} -eq 0 ] && for a in "${ARMS[@]}"; do selected+=("${a%%|*}"); done
for want in "${selected[@]}"; do
  found=0
  for a in "${ARMS[@]}"; do
    if [ "${a%%|*}" = "$want" ]; then run_arm "$a"; found=1; break; fi
  done
  [ $found = 1 ] || { echo "unknown arm: $want (known: ${ARMS[*]%%|*})" >&2; exit 1; }
done
