# Reproduce the paper

This file shows how to make every number, table and figure in the paper.

## What you need

- The Marker-CXR download from PhysioNet.
- The VinDr-CXR download from PhysioNet, the folder that holds `test/` and `annotations/`.
- `uv sync` done, the environment active, and a Hugging Face login with access to MedGemma
  (see "Setup" in `README.md`).
- For training: 4 GPUs with 48 GB each. The paper used NVIDIA L40S. 7 of the 8 arms train
  (`baseline` is zero-shot). In the paper runs, one SFT arm took 5–18 hours and one GRPO arm took
  6–28 hours. Training takes about 4 days, plus a few hours of evaluation per arm.

## One command

```bash
MARKER_CXR_DIR=/path/to/marker-cxr VINDR_DIR=/path/to/vindr-cxr/1.0.0 NCCL_P2P_DISABLE=1 ./reproduce.sh
```

`MARKER_CXR_DIR` makes the link `data/marker-cxr` to your download, so you need it only once.
Set `NCCL_P2P_DISABLE=1` only on PCIe GPUs without NVLink, such as the L40S. It stops multi-GPU
runs from hanging. All outputs go to `runs/`. You can run single steps in this order:

| Step | Command | Time | Output |
|---|---|---|---|
| 1. VinDr questions | `./reproduce.sh vindr` | minutes | `data/vindr-cxr-vqa/` |
| 2. Dataset numbers | `./reproduce.sh stats` | 1 minute | printed |
| 3. Train and evaluate all 8 arms | `./reproduce.sh train` | about 4 days | `runs/<arm>/`, `runs/vindr/<arm>/` |
| 4. Tables and Figs. 3–4 | `./reproduce.sh tables` | a few minutes | `runs/paper/` |
| 5. Fig. 2 | `./reproduce.sh fig2` | seconds | `runs/paper/fig_bbox_examples.png` |

Step 3 needs step 1, and steps 4 and 5 need step 3. If you stop step 3, start it again: it skips
the arms that are done. The `full` arm (SFT with reasoning) is not in the paper, but step 4 needs it.
Fig. 2 contains X-ray images (PhysioNet credentialed data), so do not share it in public.

## Where each result comes from

All files are in `runs/paper/`, except for step 2, which prints its numbers.

| Paper | Source |
|---|---|
| Sec. 2 and "Test sets": image, patient, split and question counts | step 2 (prints each value next to the paper value). The marker bank numbers come from the marker detector, which is not in this repository. |
| Fig. 2 | `fig_bbox_examples.png` |
| Table 1 | `table1a_internal_findings.csv` + `table1b_internal_markers.csv` (without the `full` row) |
| Fig. 3 | `fig_table3_differences.pdf` (numbers: `table3_differences.csv`) |
| Fig. 4 | `fig_marker_scaling.pdf` (numbers: `table4_scaling.csv`) |
| Results, "Fine-tuning vs zero-shot" | Table 1, `table3_differences.csv`, `table5_yes_no.csv`, `tableS1_all_results.csv` (VinDr existence accuracy) |
| Results, "Marker data helps" | Table 1, `table3_differences.csv` |
| Results, "More marker data" | `table4_scaling.csv`, `table4c_scaling_trend.csv` (trend p), `table4b_scaling_steps.csv` (step p) |
| Results, "GRPO" | Table 1, `table3_differences.csv` |
| Discussion, seen vs unseen VinDr labels | `table2a_vindr_seen.csv`, `table2b_vindr_unseen.csv`, `table5_yes_no.csv` |

Every metric has its 95% CI. `metrics_with_ci.csv` holds the metrics of each arm as plain numbers.
The other files in `runs/paper/` are extra results, not in the paper. Fig. 1 is a drawing, not made
by code.

## How close your numbers will be

- Step 2 gives exactly the paper values.
- With the paper's predictions, steps 4 and 5 give exactly the paper tables and figures. We tested
  this. We do not share these predictions, because they contain MIMIC-CXR data.
- Step 3 retrains the models. GPU training is not fully deterministic, so your numbers can differ a
  little from the paper (each arm was trained once, with one seed).
