#!/usr/bin/env python
"""VinDr-CXR test DICOMs -> 8-bit PNGs, same recipe as the Marker-CXR images:
min-max rescale to uint8, center pad to square with black.

    python 01_convert_vindr_pngs.py \
        --dicom-dir /data/vindr-cxr/1.0.0/test \
        --out-dir   ../data/vindr-cxr-vqa/images/test_png_padded \
        --meta-csv  ../data/vindr-cxr-vqa/csvs/png_meta.csv

Writes <image_id>.png per DICOM and a meta CSV (image_id, h, w, pad_top,
pad_left, dim) used by 02_build_vindr_vqa.py to move the boxes
(normalized to the original h x w) onto the padded square canvas.
"""

import argparse
import csv
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image
from pydicom.pixel_data_handlers.util import apply_voi_lut


def pad_to_square(arr: np.ndarray, fill) -> tuple:
    h, w = arr.shape
    dim = max(h, w)
    pad_top, pad_left = (dim - h) // 2, (dim - w) // 2
    padded = np.full((dim, dim), fill, dtype=arr.dtype)
    padded[pad_top : pad_top + h, pad_left : pad_left + w] = arr
    return padded, pad_top, pad_left


def convert_one(task):
    dicom_path, out_path = task
    try:
        ds = pydicom.dcmread(dicom_path)
        arr = apply_voi_lut(ds.pixel_array, ds).astype(np.float32)
        if ds.get("PhotometricInterpretation", "") == "MONOCHROME1":
            arr = arr.max() - arr
        h, w = arr.shape
        lo, hi = float(arr.min()), float(arr.max())
        arr, pad_top, pad_left = pad_to_square(arr, lo)  # lo -> 0: black border
        img8 = np.clip(np.round((arr - lo) / max(hi - lo, 1e-6) * 255.0), 0, 255).astype(np.uint8)
        Image.fromarray(img8, mode="L").save(out_path)
        return dicom_path.stem, h, w, pad_top, pad_left, max(h, w)
    except Exception as e:
        print(f"FAILED {dicom_path.name}: {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dicom-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--meta-csv", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.meta_csv).parent.mkdir(parents=True, exist_ok=True)

    dicoms = sorted(Path(args.dicom_dir).glob("*.dicom"))
    if args.limit:
        dicoms = dicoms[: args.limit]
    tasks = [(p, out_dir / f"{p.stem}.png") for p in dicoms]
    tasks = [(p, o) for p, o in tasks if not o.exists()]  # resumable
    print(f"{len(dicoms)} dicoms, {len(tasks)} to convert")

    rows = []
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(convert_one, tasks, chunksize=8)):
            if r is not None:
                rows.append(r)
            if (i + 1) % 200 == 0:
                print(f"{i + 1}/{len(tasks)}")

    # Merge with any rows from a previous (partial) run.
    meta = Path(args.meta_csv)
    seen = {}
    if meta.exists():
        with open(meta) as f:
            for row in csv.DictReader(f):
                seen[row["image_id"]] = row
    for image_id, h, w, pad_top, pad_left, dim in rows:
        seen[image_id] = dict(image_id=image_id, h=h, w=w, pad_top=pad_top, pad_left=pad_left, dim=dim)
    with open(meta, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["image_id", "h", "w", "pad_top", "pad_left", "dim"])
        wr.writeheader()
        for row in sorted(seen.values(), key=lambda r: r["image_id"]):
            wr.writerow(row)
    print(f"{len(seen)} rows in {meta}")


if __name__ == "__main__":
    main()
