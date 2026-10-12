#!/usr/bin/env python
"""Make the Marker-CXR PNGs from MIMIC-CXR DICOMs: paste markers, pad to a
square, save 8-bit.

Input is a manifest CSV (shipped with Marker-CXR on PhysioNet), one row per
output image:

    image_path        DICOM path relative to MIMIC-CXR files/, e.g.
                      p10/p10000001/s50000001/<dicom_id>.dcm
    cohort            pasted | single_marker | marker_free
    seed              paste seed (pasted, single_marker); same seed = same image
    real_marker_bbox  JSON list of markers already on the film,
                      [[x0, y0, x1, y1], ...] normalized (pasted only, else [])

Cohorts:
    pasted         1-3 stencils pasted next to the film's own markers
    single_marker  exactly 1 stencil pasted on a marker-free film
    marker_free    no paste, only pad + save

Output, one PNG per row plus a box table:

    <out>/pasted_images_padded/<image_path>.png
    <out>/single_marker_images_padded/<image_path>.png
    <out>/marker_free_images_padded/<image_path>.png
    <out>/prepared_images.csv   frame (orig size, padding) and pasted boxes
                                as [y0, y1, x0, x1] normalized to the padded
                                square, the same format as the dataset tables

    python prepare_images.py --manifest manifest.csv --mimic-files /data/mimic-cxr/files \\
        --bank /data/marker-cxr/bank --out /data/marker-cxr/images --workers 8
"""

from __future__ import annotations

import os

# One BLAS thread per worker process; many workers x many threads stalls the CPU.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import marker_bank_io as mb
import marker_paste

COHORT_DIRS = {
    "pasted": "pasted_images_padded",
    "single_marker": "single_marker_images_padded",
    "marker_free": "marker_free_images_padded",
}
N_RANGE = {"pasted": (1, 3), "single_marker": (1, 1)}


def pad_to_square(arr: np.ndarray, fill: float) -> tuple[np.ndarray, int, int]:
    """Center arr on a max(H, W) square filled with `fill`.
    Returns (padded, pad_top, pad_left)."""
    h, w = arr.shape
    dim = max(h, w)
    pad_top, pad_left = (dim - h) // 2, (dim - w) // 2
    padded = np.full((dim, dim), fill, dtype=arr.dtype)
    padded[pad_top : pad_top + h, pad_left : pad_left + w] = arr
    return padded, pad_top, pad_left


def pad_box(box, h: int, w: int, pad_top: int, pad_left: int) -> list[float]:
    """[y0, y1, x0, x1] normalized to the h x w original -> normalized to the
    padded square."""
    dim = max(h, w)
    y0, y1, x0, x1 = box
    return [
        (y0 * h + pad_top) / dim,
        (y1 * h + pad_top) / dim,
        (x0 * w + pad_left) / dim,
        (x1 * w + pad_left) / dim,
    ]


def save_png(img: np.ndarray, out_path: Path) -> tuple[int, int, int, int]:
    """Pad with the image minimum (black after rescale), rescale min/max to
    0-255, save 8-bit. Returns (h, w, pad_top, pad_left) of the original."""
    h, w = img.shape
    lo, hi = float(img.min()), float(img.max())
    padded, pad_top, pad_left = pad_to_square(img, lo)
    img8 = np.clip(np.round((padded - lo) / max(hi - lo, 1e-6) * 255.0), 0, 255).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img8, mode="L").save(out_path)
    return h, w, pad_top, pad_left


_PATS = None
_ARGS = None


def _init(bank_dir, mimic_files, out_dir):
    """Load the bank once per worker."""
    global _PATS, _ARGS
    _PATS = mb.load_bank(bank_dir) if bank_dir else None
    _ARGS = (Path(mimic_files), Path(out_dir))


def process(row: dict) -> dict | None:
    mimic_files, out_dir = _ARGS
    cohort = row["cohort"]
    out_path = out_dir / COHORT_DIRS[cohort] / Path(row["image_path"]).with_suffix(".png")
    try:
        donor = mb.Donor(mimic_files / row["image_path"])
        if cohort == "marker_free":
            img, boxes = donor.screen_array(max_side=None)[0], []
        else:
            real = [tuple(b) for b in json.loads(row.get("real_marker_bbox") or "[]")]
            img, boxes = marker_paste.paste_random(
                donor, real, _PATS, n_range=N_RANGE[cohort], rng=int(row["seed"])
            )
            if cohort == "single_marker" and len(boxes) != 1:
                return None
    except Exception as e:
        print(f"failed {row['image_path']}: {e}", flush=True)
        return None
    h, w, pad_top, pad_left = save_png(img, out_path)
    # (x0, y0, x1, y1) original -> [y0, y1, x0, x1] on the padded square
    padded = [pad_box([y0, y1, x0, x1], h, w, pad_top, pad_left) for x0, y0, x1, y1 in boxes]
    return {
        "image_path": str(out_path.relative_to(out_dir)),
        "cohort": cohort,
        "orig_image_height": h,
        "orig_image_width": w,
        "pad_top": pad_top,
        "pad_left": pad_left,
        "image_size": max(h, w),
        "pasted_marker_bbox": json.dumps([[round(v, 6) for v in b] for b in padded]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--mimic-files", required=True, help="MIMIC-CXR (DICOM) files/ folder")
    ap.add_argument("--bank", help="marker bank folder (needed for pasted / single_marker)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    rows = pd.read_csv(args.manifest).to_dict("records")
    unknown = {r["cohort"] for r in rows} - set(COHORT_DIRS)
    if unknown:
        raise SystemExit(f"unknown cohort(s): {unknown}")
    if args.bank is None and any(r["cohort"] != "marker_free" for r in rows):
        raise SystemExit("--bank is required for pasted / single_marker rows")

    results = []
    with mp.get_context("fork").Pool(
        max(1, min(args.workers, len(rows))), initializer=_init,
        initargs=(args.bank, args.mimic_files, args.out),
    ) as pool:
        for k, res in enumerate(pool.imap(process, rows), 1):
            if res is not None:
                results.append(res)
            if k % 200 == 0:
                print(f"{k}/{len(rows)}", flush=True)

    out_csv = Path(args.out) / "prepared_images.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"{len(results)}/{len(rows)} images saved; table -> {out_csv}")


if __name__ == "__main__":
    main()
