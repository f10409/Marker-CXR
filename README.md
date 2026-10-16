# Marker-CXR

Code for the paper **"Pasted markers teach a medical vision-language model where things are in
chest X-rays"** (ISBI 2027, under review).

Marker-CXR is a visual question answering dataset built on MIMIC-CXR. Real radiographic markers
(L/R letters, text, symbols) are cut from MIMIC-CXR films and pasted into other films, so every
marker bounding box (bbox) is known exactly, with no manual labeling. Questions also ask about
finding bboxes (MS-CXR) and anatomical regions (Chest ImaGenome). This repository holds the code to
fine-tune MedGemma-1.5-4B on it (SFT and GRPO), to evaluate on Marker-CXR and VinDr-CXR, and to
rebuild the pasted images.

## What is here

```
finetune/        SFT + GRPO training and evaluation of MedGemma-1.5-4B
  01_finetune_markercxr.py   SFT: data / train / eval / rescore
  02_grpo_markercxr.py       GRPO with a bbox reward (IoU + center distance)
  run_arms.sh                train and evaluate every arm in the paper
  configs/                   one YAML per paper arm, plus vindr_eval.yaml
  src/                       data loading, prompt format, rewards, metrics, statistics
image_prep/      rebuild the Marker-CXR images from MIMIC-CXR DICOMs
  prepare_images.py          paste markers from the bank, pad to a square, save PNG
  marker_paste.py            pasting (size in mm, random scale / rotation / brightness / noise)
  marker_bank_io.py          load the marker bank, read DICOMs
vindr/           build the VinDr-CXR external test questions from the official files
viewer/          review_vqa.html: browse questions and answers overlaid on the images
templates/       question templates (JSON) and their description
demo_data/       fake example of the data layout (placeholder images, made-up boxes)
```

Not included: the marker detector and the question generator.

## Data

All data is credentialed PhysioNet data. None of it is in this repository.

| Data | Where | Needed for |
|---|---|---|
| Marker-CXR (padded PNGs, `vqa/` JSONs, `csvs/`, marker bank, image manifest) | PhysioNet: **TODO link** | training, evaluation |
| MIMIC-CXR (DICOM) | [physionet.org/content/mimic-cxr](https://physionet.org/content/mimic-cxr/) | only to rebuild the images yourself |
| VinDr-CXR | [physionet.org/content/vindr-cxr](https://physionet.org/content/vindr-cxr/) | external test |

Put the Marker-CXR download at `data/marker-cxr/` (or change `paths` in the configs). If it is
somewhere else, link it there from the repository root:

```bash
mkdir -p data
ln -s /path/to/Marker-CXR data/marker-cxr
```

Expected layout:

```
data/marker-cxr/
  vqa/<folder>/<dicom_id>.json      one file per image, all questions for that image
  vqa/review_manifest.json          overlay data for viewer/review_vqa.html
  csvs/                             image tables (split per patient)
  images/
    pasted_images_padded/p10/p10000001/s50000001/<dicom_id>.png
    single_marker_images_padded/...
    marker_free_images_padded/...
  bank/                             marker stencils (only for image_prep/)
```

`demo_data/marker-cxr/` has the same layout with one example per question type. Its images are
gray placeholders and its IDs and box numbers are made up.

### Question format

One JSON per image and folder (see `demo_data/marker-cxr/vqa/`):

```json
{
  "image_path": "pasted_images_padded/p10/p10000001/s50000001/demo-pasted.png",
  "patient_id": 10000001, "study_id": 50000001, "split": "train",
  "questions": [
    {
      "question": "Where is the left upper lung zone? Give its full bounding box, not just a point.",
      "answer": "At [227, 477, 555, 859].",
      "answer_struct": [227, 477, 555, 859],
      "question_type": "anatomical_region_localization",
      "reasoning_frame": "The left upper lung zone is the top third of the left lung, ...",
      "question_id": "localization_bbox_50000001_0"
    }
  ]
}
```

- Images are padded to a square (black border), then boxes are integers on a 0-1000 scale.
- Stored boxes are `[y0, y1, x0, x1]`. `finetune/src/medgemma_format.py` reorders them to MedGemma's
  `[y0, x0, y1, x1]` JSON format for training.
- `reasoning_frame` is a short trace built only from the true labels. It is used only by the
  reasoning arms.
- The paper evaluates existence and localization questions. The other question types (counting,
  grounding, identification) are used in training only.

## Setup

Needs [uv](https://docs.astral.sh/uv/). `uv.lock` pins the exact versions used for the paper
(Python 3.12, PyTorch 2.13 with CUDA 12.9).

```bash
uv sync                   # makes .venv with the locked versions
source .venv/bin/activate
hf auth login             # MedGemma is a gated model: accept its terms on Hugging Face first
```

Check the data loader on the demo data (no GPU; balancing is off because the demo has one row per
type):

```bash
cd finetune
python 01_finetune_markercxr.py data --config configs/sft.yaml \
  --set paths.marker_cxr_dir=../demo_data/marker-cxr \
  --set paths.images_root=../demo_data/marker-cxr/images \
  --set "data.balance_question_types=[]"
```

## Train and evaluate

Every arm in the paper, training then evaluation on the internal test set and VinDr-CXR:

```bash
cd finetune
NPROC=4 ./run_arms.sh                    # all arms (SFT on 4 GPUs, GRPO on 2)
NPROC=4 ./run_arms.sh no_reasoning       # one arm
CUDA_VISIBLE_DEVICES=0,1 ./run_arms.sh grpo_box_no_reasoning   # GRPO on GPUs 0 and 1
```

| Paper arm | run name | config |
|---|---|---|
| Zero-shot | `baseline` | `sft.yaml` (eval only) |
| SFT, no marker data | `no_marker` | `sft.yaml` + marker question types excluded |
| SFT | `no_reasoning` | `sft.yaml` |
| SFT + reasoning | `full` | `sft.yaml` + reasoning on |
| SFT, 25% / 50% marker data | `marker25`, `marker50` | `sft_marker25.yaml`, `sft_marker50.yaml` |
| SFT + GRPO | `grpo_box_no_reasoning` | `grpo.yaml` (needs `no_reasoning` first) |
| GRPO, no SFT | `grpo_box_base_reasoning` | `grpo_from_base.yaml` |

Outputs go to `runs/<run name>/` (adapter, logs, `eval/finetuned/metrics.json`). The paper used
4 NVIDIA L40S GPUs (48 GB) for SFT (effective batch 16) and 2 for GRPO (32 completions per step).
With other GPU counts, change `gradient_accumulation_steps` in the config to keep these batch
sizes (`GRPO_NPROC` sets the GRPO GPU count). Single commands:

```bash
python 01_finetune_markercxr.py train --config configs/sft.yaml
python 01_finetune_markercxr.py eval  --config configs/sft.yaml --adapter ../runs/no_reasoning/adapter
python 02_grpo_markercxr.py train --config configs/grpo.yaml
```

Any config key can be changed with `--set key=value` (the value is read as YAML).

## VinDr-CXR external test

```bash
cd vindr
python 01_convert_vindr_pngs.py --dicom-dir /path/to/vindr-cxr/1.0.0/test \
  --out-dir ../data/vindr-cxr-vqa/images/test_png_padded \
  --meta-csv ../data/vindr-cxr-vqa/csvs/png_meta.csv
python 02_build_vindr_vqa.py --annotations-dir /path/to/vindr-cxr/1.0.0/annotations \
  --out-dir ../data/vindr-cxr-vqa
```

This gives the questions used in the paper: 1,594 bbox questions on 869 images (finding-image pairs
with a single bbox) and 6,910 balanced yes/no questions. `run_arms.sh` then evaluates on them with
`configs/vindr_eval.yaml`.

## Browse the data

`viewer/review_vqa.html` shows each image with its marker, finding and anatomy boxes and every
question and answer. It runs in the browser and only reads local files. Copy it into the
Marker-CXR folder and serve that folder:

```bash
cp viewer/review_vqa.html /path/to/marker-cxr/
cd /path/to/marker-cxr && python3 -m http.server 8000
# open http://localhost:8000/review_vqa.html and click "Load from server"
```

Or open the file directly in Chrome / Edge and pick the `vqa/` and `images/` folders.

## Rebuild the images (optional)

The padded PNGs are on PhysioNet. To make them yourself from MIMIC-CXR DICOMs, use the manifest
shipped with Marker-CXR (image path, cohort, seed, existing marker boxes):

```bash
cd image_prep
python prepare_images.py --manifest /path/to/marker-cxr/manifest.csv \
  --mimic-files /path/to/mimic-cxr/files --bank ../data/marker-cxr/bank \
  --out ../data/marker-cxr/images --workers 8
```

With the shipped seeds, the output matches the released PNGs pixel for pixel. Check it with the
shipped checksums (other numpy / scipy / pydicom versions than in `uv.lock` may change
pixels):

```bash
cd ../data/marker-cxr/images && sha256sum -c /path/to/marker-cxr/SHA256SUMS_images.txt
```

Markers keep their
real size in mm, so this needs the DICOMs (for pixel spacing), not MIMIC-CXR-JPG. Only the ~6k
DICOMs listed in the manifest are read. See `demo_data/manifest_example.csv` for the format.

## Citation

TODO

## License

TODO
