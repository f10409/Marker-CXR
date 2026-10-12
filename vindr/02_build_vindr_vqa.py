#!/usr/bin/env python
"""Build Marker_CXR-style VQA JSONs for VinDr-CXR test (external validation).

Reads the official VinDr-CXR test annotations (annotations_test.csv,
image_labels_test.csv) and the png_meta.csv written by
01_convert_vindr_pngs.py, and writes one JSON per image under
<out_dir>/vqa/{localization_bbox,existence}/ plus the csvs/ split tables
the fine-tuning code expects (every image -> split "test").

Question types reuse the Marker-CXR names so the eval pipeline scores them
unchanged:
  finding_localization_mscxr   "Where is the X? Give its full bounding box,
                               not just a point."  Answer "At [y0,y1,x0,x1]."
                               (0-1000 ints on the padded square canvas).
                               Only (image, label) pairs with exactly ONE box
                               -- several boxes would make the answer
                               ambiguous. reasoning_frame is an EMPTY string:
                               VinDr has no region windows to state, but the
                               key must exist so reasoning-arm evals
                               (use_reasoning_for_boxes=true) get the
                               reasoning prompt and the 1000-token budget
                               instead of silently falling back to the
                               direct prompt.
  specific_label_existence     "Is there evidence of X in this image?"
                               Per label: all positives + the same number of
                               sampled negatives (seed 42), so accuracy is
                               not inflated by the ~2/3 no-finding images.
  finding_existence            "Does this image show a finding?" All
                               no_finding==0 images + the same number of
                               sampled no-finding images.

    python 02_build_vindr_vqa.py \
        --annotations-dir /data/vindr-cxr/1.0.0/annotations \
        --out-dir ../data/vindr-cxr-vqa
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

# Labels asked as existence questions. no_finding is covered by
# finding_existence; other_diseases / other_lesion are catch-alls a model
# cannot answer from the name.
SKIP_LABELS = {"no_finding", "other_diseases", "other_lesion"}

# snake_case column -> phrase used in the question text.
LABEL_PHRASE = {
    "aortic_enlargement": "aortic enlargement",
    "atelectasis": "atelectasis",
    "copd": "COPD",
    "calcification": "calcification",
    "cardiomegaly": "cardiomegaly",
    "clavicle_fracture": "clavicle fracture",
    "consolidation": "consolidation",
    "edema": "edema",
    "emphysema": "emphysema",
    "enlarged_pa": "enlarged pulmonary artery",
    "ild": "interstitial lung disease",
    "infiltration": "infiltration",
    "lung_opacity": "lung opacity",
    "lung_cavity": "lung cavity",
    "lung_cyst": "lung cyst",
    "lung_tumor": "lung tumor",
    "mediastinal_shift": "mediastinal shift",
    "nodule/mass": "nodule or mass",
    "pleural_effusion": "pleural effusion",
    "pleural_thickening": "pleural thickening",
    "pneumonia": "pneumonia",
    "pneumothorax": "pneumothorax",
    "pulmonary_fibrosis": "pulmonary fibrosis",
    "rib_fracture": "rib fracture",
    "tuberculosis": "tuberculosis",
}


def pad_box(box, h, w, pad_top, pad_left):
    """[y0, y1, x0, x1] normalized to h x w -> 0-1000 ints on the padded
    square canvas (same math as image_prep/prepare_images.pad_box)."""
    dim = max(h, w)
    y0, y1, x0, x1 = box
    out = [
        (y0 * h + pad_top) / dim,
        (y1 * h + pad_top) / dim,
        (x0 * w + pad_left) / dim,
        (x1 * w + pad_left) / dim,
    ]
    return [min(1000, max(0, round(v * 1000))) for v in out]


def snake(name: str) -> str:
    """VinDr class name -> snake_case label ("Nodule/Mass" -> "nodule/mass")."""
    return {"Other disease": "other_diseases"}.get(name, name.lower().replace(" ", "_"))


def load_vindr_test(ann_dir: Path, meta: pd.DataFrame) -> pd.DataFrame:
    """One row per test image (image_labels_test.csv order): image-level label
    columns, plus its boxes as [y0, y1, x0, x1] normalized to the DICOM h x w
    and their labels, in annotations_test.csv order."""
    labels = pd.read_csv(ann_dir / "image_labels_test.csv")
    labels = labels.rename(columns={c: snake(c) for c in labels.columns if c != "image_id"})
    ann = pd.read_csv(ann_dir / "annotations_test.csv").dropna(subset=["x_min"])
    by_image = {iid: g for iid, g in ann.groupby("image_id", sort=False)}
    bboxes, bbox_labels = [], []
    for iid in labels["image_id"]:
        g = by_image.get(iid)
        m = meta.loc[iid]
        if g is None:
            bboxes.append([]); bbox_labels.append([])
            continue
        bboxes.append([[y0 / m.h, y1 / m.h, x0 / m.w, x1 / m.w]
                       for x0, y0, x1, y1 in g[["x_min", "y_min", "x_max", "y_max"]].to_numpy()])
        bbox_labels.append([snake(c) for c in g["class_name"]])
    labels["bbox"], labels["bbox_labels"] = bboxes, bbox_labels
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations-dir", required=True,
                    help="VinDr-CXR annotations/ folder (annotations_test.csv, image_labels_test.csv)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out_dir)
    meta = pd.read_csv(out / "csvs" / "png_meta.csv").set_index("image_id")
    df = load_vindr_test(Path(args.annotations_dir), meta)
    assert set(df["image_id"]) <= set(meta.index), "missing PNGs; run convert first"

    rng = random.Random(args.seed)
    questions = defaultdict(lambda: defaultdict(list))  # folder -> image_id -> [q]

    # --- localization: one question per (image, label) with exactly one box
    n_boxes = n_multi = 0
    for row in df.itertuples():
        boxes, labels = row.bbox, row.bbox_labels
        n_boxes += len(boxes)
        by_label = defaultdict(list)
        for b, l in zip(boxes, labels):
            by_label[l].append(b)
        m = meta.loc[row.image_id]
        for label, bl in sorted(by_label.items()):
            if label in SKIP_LABELS or label not in LABEL_PHRASE:
                continue
            if len(bl) != 1:
                n_multi += 1
                continue
            struct = pad_box(bl[0], m.h, m.w, m.pad_top, m.pad_left)
            phrase = LABEL_PHRASE[label]
            qs = questions["localization_bbox"][row.image_id]
            qs.append({
                "question": f"Where is the {phrase}? Give its full bounding box, not just a point.",
                "answer": f"At [{struct[0]}, {struct[1]}, {struct[2]}, {struct[3]}].",
                "answer_struct": struct,
                "question_type": "finding_localization_mscxr",
                "reasoning_frame": "",
                "vindr_label": label,
                "question_id": f"vindr_localization_{row.image_id}_{len(qs)}",
            })

    # --- existence: per label, all positives + equally many sampled negatives
    df_idx = df.set_index("image_id")
    for label in sorted(LABEL_PHRASE):
        pos = df_idx.index[df_idx[label] == 1].tolist()
        neg = df_idx.index[df_idx[label] == 0].tolist()
        neg = rng.sample(neg, min(len(pos), len(neg)))
        for image_id, ans in [(i, True) for i in pos] + [(i, False) for i in neg]:
            qs = questions["existence"][image_id]
            qs.append({
                "question": f"Is there evidence of {LABEL_PHRASE[label]} in this image?",
                "answer": "Yes" if ans else "No",
                "answer_struct": ans,
                "question_type": "specific_label_existence",
                "vindr_label": label,
                "question_id": f"vindr_existence_{image_id}_{len(qs)}",
            })

    # --- finding_existence: all with-finding images + equally many no-finding
    pos = df_idx.index[df_idx["no_finding"] == 0].tolist()
    neg = df_idx.index[df_idx["no_finding"] == 1].tolist()
    neg = rng.sample(neg, min(len(pos), len(neg)))
    for image_id, ans in [(i, True) for i in pos] + [(i, False) for i in neg]:
        qs = questions["existence"][image_id]
        qs.append({
            "question": "Does this image show a finding?",
            "answer": "Yes" if ans else "No",
            "answer_struct": ans,
            "question_type": "finding_existence",
            "question_id": f"vindr_existence_{image_id}_{len(qs)}",
        })

    # --- write JSONs
    for folder, by_image in questions.items():
        d = out / "vqa" / folder
        d.mkdir(parents=True, exist_ok=True)
        for image_id, qs in by_image.items():
            with open(d / f"{image_id}.json", "w") as f:
                json.dump({
                    "image_path": f"test_png_padded/{image_id}.png",
                    "patient_id": image_id,
                    "study_id": image_id,
                    "split": "test",
                    "questions": qs,
                }, f, indent=1)
        n = sum(len(qs) for qs in by_image.values())
        print(f"{folder}: {len(by_image)} images, {n} questions")
    print(f"(localization skipped {n_multi} multi-box (image,label) pairs; "
          f"{n_boxes} raw boxes total)")

    # --- split tables for markercxr_data.load_split_map
    csvs = out / "csvs"
    pd.DataFrame({"patient_id": df["image_id"], "split": "test"}).to_csv(
        csvs / "final_table.csv", index=False)
    empty = pd.DataFrame(columns=["patient_id", "split"])
    empty.to_csv(csvs / "marker_free_images.csv", index=False)
    empty.to_csv(csvs / "single_marker_images.csv", index=False)
    print(f"split tables written to {csvs}")


if __name__ == "__main__":
    main()
