#!/usr/bin/env python
"""Paper Fig. 2: true vs predicted bboxes on 4 test examples (marker, finding, anatomy, VinDr finding).

Each panel uses the example whose SFT IoU is closest to the SFT median IoU for that bbox type
(picked by this rule, not by hand). Needs the `baseline` and `no_reasoning` predictions from run_arms.sh.
The output contains X-ray images (PhysioNet credentialed data). Do not share it in public.

    python 04_fig_examples.py    # writes ../runs/paper/fig_bbox_examples.{pdf,png}
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
from eval_scoring import extract_box_2d  # noqa: E402

RUNS = HERE.parent / "runs"  # run_arms.sh output
VINDR = RUNS / "vindr"
OUT = RUNS / "paper/fig_bbox_examples"
ARMS = {  # name: (internal predictions, VinDr predictions, color, line style)
    "Zero-shot": (RUNS / "baseline/eval/predictions.csv", VINDR / "baseline/predictions.csv", "#d62728", "--"),
    "SFT": (RUNS / "no_reasoning/eval/finetuned/predictions.csv", VINDR / "no_reasoning/predictions.csv", "#1f77b4", "-"),
}
REF_ARM = "SFT"
PANELS = [  # (title, source, question_type)
    ("Marker", "internal", "marker_localization"),
    ("Finding, MS-CXR", "internal", "finding_localization_mscxr"),
    ("Anatomy, ImaGenome", "internal", "anatomical_region_localization"),
    ("Finding, VinDr-CXR", "vindr", "finding_localization_mscxr"),
]
# VinDr labels that are also training findings (the "seen" labels)
SEEN = {"cardiomegaly", "pleural effusion", "atelectasis", "lung opacity", "consolidation", "pneumothorax"}
LABEL_RE = re.compile(r'"label":\s*"([^"]+)"')


def load(source, qtype):
    """One row per question, with true bbox, label, and each arm's bbox and IoU."""
    merged = None
    for arm, (internal, vindr, *_) in ARMS.items():
        d = pd.read_csv(internal if source == "internal" else vindr)
        d = d[d.question_type == qtype].copy()
        d["true_box"] = d.reference.map(lambda t: extract_box_2d(t[t.rfind("```json"):]))
        d["key"] = d.image_path + "|" + d.true_box.astype(str)
        d = d.drop_duplicates("key", keep=False)
        d[f"{arm}_box"] = d.prediction.map(lambda t: extract_box_2d(str(t)))
        d = d.rename(columns={"iou": f"{arm}_iou"})
        if merged is None:
            d["label"] = d.reference.map(lambda t: LABEL_RE.findall(t)[-1])
            merged = d[["key", "image_path", "true_box", "label", f"{arm}_box", f"{arm}_iou"]]
        else:
            merged = merged.merge(d[["key", f"{arm}_box", f"{arm}_iou"]], on="key")
    return merged


def pick(d):
    """Row whose reference-arm IoU is closest to that arm's median; all arms must give a bbox."""
    d = d.dropna(subset=[f"{a}_box" for a in ARMS])
    med = d[f"{REF_ARM}_iou"].median()
    return d.loc[(d[f"{REF_ARM}_iou"] - med).abs().sort_values(kind="stable").index[0]], med


def draw(ax, img, box, color, ls, lw=1.4):
    h, w = img.shape[:2]
    y0, x0, y1, x1 = box
    ax.add_patch(Rectangle((x0 * w, y0 * h), (x1 - x0) * w, (y1 - y0) * h,
                           fill=False, edgecolor=color, linestyle=ls, linewidth=lw))


def draw_all(ax, img, row, scale=1.0):
    draw(ax, img, row.true_box, "#2ca02c", "-", lw=1.6 * scale)
    for arm, (*_, color, ls) in ARMS.items():
        draw(ax, img, row[f"{arm}_box"], color, ls, lw=1.4 * scale)


def main():
    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "pdf.fonttype": 42})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(PANELS), figsize=(7.0, 2.3))
    for ax, (title, source, qtype) in zip(axes, PANELS):
        d = load(source, qtype)
        if source == "vindr":
            d = d[d.label.isin(SEEN - {"cardiomegaly"})]  # cardiomegaly: bbox is just the heart, too easy
        row, med = pick(d)
        img = plt.imread(row.image_path) if row.image_path.endswith(".png") else Image.open(row.image_path)
        ax.imshow(img, cmap="gray")
        draw_all(ax, img, row)
        ious = "  ".join(f"{a} {row[f'{a}_iou']:.2f}" for a in ARMS)
        sub = "pasted marker" if "single_marker" in row.image_path else row.label
        ax.set_title(f"{title}\n{sub}", fontsize=8, pad=2)
        ax.set_xlabel(f"IoU: {ious}", fontsize=7, labelpad=2)
        ax.set_xticks([]), ax.set_yticks([])
        print(f"{title:22s} {row.label:20s} {ious}  (SFT median {med:.2f})  {row.image_path}")
    handles = [plt.Line2D([], [], color="#2ca02c", lw=1.6, label="true box")]
    handles += [plt.Line2D([], [], color=c, ls=ls, lw=1.4, label=a) for a, (*_, c, ls) in ARMS.items()]
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=0.4)
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 0.0))
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", dpi=300, bbox_inches="tight")
    print("saved", OUT)


if __name__ == "__main__":
    main()
