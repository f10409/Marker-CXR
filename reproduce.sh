#!/usr/bin/env bash
# Make every result of the paper. See REPRODUCE.md for the steps and the outputs.
#   ./reproduce.sh                 # all steps, in order
#   ./reproduce.sh tables fig2     # only these steps
# Steps:
#   vindr   build the VinDr-CXR test questions (needs VINDR_DIR; skipped if already built)
#   stats   dataset numbers (Sec. 2 and test sets), printed next to the paper values
#   train   train and evaluate all 8 arms (GPU, about 4 days; runs/<arm>/, runs/vindr/<arm>/)
#   tables  Table 1, Fig. 3, Fig. 4 and the numbers in the text (runs/paper/)
#   fig2    Fig. 2, bbox examples (runs/paper/fig_bbox_examples.png)
# Env:
#   MARKER_CXR_DIR  Marker-CXR download (optional: links it to data/marker-cxr if that is missing)
#   VINDR_DIR   VinDr-CXR download, the folder with test/ and annotations/ (step vindr)
#   NPROC       GPUs for SFT                     (default: 4, as in the paper)
#   GRPO_NPROC  GPUs for GRPO                    (default: 2, as in the paper)
#   Other run_arms.sh settings (e.g. NCCL_P2P_DISABLE=1 on PCIe GPUs) pass through.
set -euo pipefail
cd "$(dirname "$0")"
export NPROC=${NPROC:-4}
PYTHON=${PYTHON:-python}
VQA=data/vindr-cxr-vqa
[ -n "${VINDR_DIR:-}" ] && VINDR_DIR=$(realpath "$VINDR_DIR")  # the steps cd into subfolders

step_vindr() {
  # 02_build_vindr_vqa.py writes this file last, so a half-built folder is built again
  if [ -f "$VQA/csvs/single_marker_images.csv" ]; then echo "VinDr questions already in $VQA, skip"; return; fi
  : "${VINDR_DIR:?set VINDR_DIR to the VinDr-CXR folder that holds test/ and annotations/}"
  (cd vindr
   "$PYTHON" 01_convert_vindr_pngs.py --dicom-dir "$VINDR_DIR/test" \
     --out-dir "../$VQA/images/test_png_padded" --meta-csv "../$VQA/csvs/png_meta.csv"
   "$PYTHON" 02_build_vindr_vqa.py --annotations-dir "$VINDR_DIR/annotations" --out-dir "../$VQA")
}

step_stats()  { (cd finetune && "$PYTHON" 00_data_stats.py); }
step_train()  { (cd finetune && ./run_arms.sh); }
step_tables() {  # jupyter is not in uv.lock, so uv adds it for this command only
  (cd finetune && uv run --with jupyter jupyter nbconvert --to notebook --execute 03_paper_results.ipynb \
     --output-dir ../runs/paper --ExecutePreprocessor.timeout=3000)
}
step_fig2()   { (cd finetune && "$PYTHON" 04_fig_examples.py); }

if [ -n "${MARKER_CXR_DIR:-}" ]; then
  MARKER_CXR_DIR=$(realpath "$MARKER_CXR_DIR")
  if [ ! -e data/marker-cxr ]; then
    mkdir -p data && ln -s "$MARKER_CXR_DIR" data/marker-cxr && echo "linked data/marker-cxr -> $MARKER_CXR_DIR"
  elif [ "$(realpath data/marker-cxr)" != "$MARKER_CXR_DIR" ]; then
    echo "data/marker-cxr already points to $(realpath data/marker-cxr), not to MARKER_CXR_DIR" >&2; exit 1
  fi
fi
[ -d data/marker-cxr/vqa ] || { echo "data/marker-cxr not found: set MARKER_CXR_DIR, or see 'Data' in README.md" >&2; exit 1; }
steps=("$@")
[ ${#steps[@]} -eq 0 ] && steps=(vindr stats train tables fig2)
for s in "${steps[@]}"; do
  declare -F "step_$s" >/dev/null || { echo "unknown step: $s (known: vindr stats train tables fig2)" >&2; exit 1; }
  echo "=== $s"
  "step_$s"
done
