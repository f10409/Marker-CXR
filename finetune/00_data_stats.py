#!/usr/bin/env python
"""Print the dataset numbers of the paper (Sec. 2 and test sets), next to the paper values.

Reads the Marker-CXR CSVs and questions (configs/sft.yaml paths) and the VinDr-CXR
questions (configs/vindr_eval.yaml paths). CPU only, no model.

    python 00_data_stats.py
    python 00_data_stats.py --set paths.marker_cxr_dir=/path/to/marker-cxr
"""

import argparse
import ast
import importlib

import pandas as pd

ft = importlib.import_module("01_finetune_markercxr")  # name starts with a digit
from markercxr_data import BOX_PRODUCING_TYPES  # noqa: E402  (src/ is on sys.path after the import above)


def has_items(col):
    """True where a CSV list column (e.g. "['effusion']") is not empty."""
    return col.map(lambda x: isinstance(x, str) and len(ast.literal_eval(x)) > 0)


def images_table(marker_cxr_dir):
    """One row per unique radiograph. The single-marker cohort uses the marker-free images."""
    csvs = marker_cxr_dir / "csvs"
    pasted = pd.read_csv(csvs / "final_table.csv")
    free = pd.read_csv(csvs / "marker_free_images.csv")
    for t in (pasted, free):
        t["dicom_id"] = t["image_path"].str.rsplit("/", n=1).str[-1]
    both = pd.concat([pasted, free]).drop_duplicates("dicom_id").reset_index(drop=True)
    return pasted, free, both


def show(rows):
    print("\n" + pd.DataFrame(rows, columns=["number", "computed", "paper"]).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ft.HERE / "configs/sft.yaml"), help="Marker-CXR config")
    ap.add_argument("--vindr-config", default=str(ft.HERE / "configs/vindr_eval.yaml"), help="VinDr-CXR config")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a --config key")
    ap.add_argument("--vindr-set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a --vindr-config key")
    args = ap.parse_args()
    cfg = ft.load_config(args.config, args.set)
    rows = []

    def add(name, value, paper):
        rows.append((name, f"{value:,}" if isinstance(value, int) else value, paper))

    pasted, free, both = images_table(ft.resolve_path(cfg["paths"]["marker_cxr_dir"]))
    add("images", len(both), "6,032")
    add("patients", both["patient_id"].nunique(), "5,063")
    add("MS-CXR images", int(both["is_ms_cxr"].sum()), "1,032")
    gold = both["imagenome_split"] == "gold"
    add("ImaGenome gold images", int(gold.sum()), "1,000")
    add("ImaGenome silver images (not MS-CXR)", int((~gold & ~both["is_ms_cxr"]).sum()), "4,009")
    add("ImaGenome gold images also in MS-CXR", int((gold & both["is_ms_cxr"]).sum()), "9")
    add("pasted cohort images", len(pasted), "3,023")
    add("marker-free cohort images", len(free), "3,023")
    add("images in both cohorts", len(set(pasted["dicom_id"]) & set(free["dicom_id"])), "14")
    for split, paper_pat, paper_img in [("train", "3,972", "4,805"), ("valid", "561", "636"), ("test", "530", "591")]:
        s = both[both["split"] == split]
        add(f"{split} patients", s["patient_id"].nunique(), paper_pat)
        add(f"{split} images", len(s), paper_img)
    add("pasted images with >= 1 ImaGenome finding", f"{has_items(pasted['imagenome_finding_labels']).mean():.1%}", "44%")

    train = ft.training_records(cfg)
    add("questions (all splits)", len(train), "156,169")
    test = ft.test_records(cfg)
    add("internal test questions", len(test), "15,518")
    # same radiograph in 2-3 cohort folders: count by file name
    add("internal test images", len({r["image_path"].rsplit("/", 1)[-1] for r in test}), "591")
    box = [r for r in test if r["question_type"] in BOX_PRODUCING_TYPES]
    add("internal test bbox questions", len(box), "2,325")
    add("internal test finding bboxes", sum(r["question_type"].startswith("finding_localization") for r in box), "124")

    vcfg = ft.load_config(args.vindr_config, args.vindr_set)
    if not (ft.resolve_path(vcfg["paths"]["marker_cxr_dir"]) / "vqa").is_dir():
        print("\nVinDr-CXR questions not built yet (see ../vindr/), VinDr numbers skipped")
        show(rows)
        return
    vtest = ft.test_records(vcfg)
    vbox = [r for r in vtest if r["question_type"] in BOX_PRODUCING_TYPES]
    add("VinDr bbox questions", len(vbox), "1,594")
    add("VinDr bbox images", len({r["image_path"] for r in vbox}), "869")
    add("VinDr yes/no questions", len(vtest) - len(vbox), "6,910")

    show(rows)


if __name__ == "__main__":
    main()
