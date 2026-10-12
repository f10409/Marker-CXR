"""Result-table helpers: yes/no metrics and tables colored by
the rank of each arm (red = best arm, purple = worst)."""

import numpy as np
import pandas as pd

# Which direction is better per column; anything not listed: higher is better.
#   low = lower, one = closest to 1 (log scale, so 2x too big = 2x too small),
#   half = closest to 0.5
BETTER = {"miss": "low", "miss (iou=0)": "low", "unparsed": "low", "yes_rate": "half",
          "area_ratio": "one", "area_ratio (median)": "one",
          "internal_mscxr_area_ratio": "one", "vindr_area_ratio": "one"}
SCORE = {"high": lambda s: s, "low": lambda s: -s,
         "one": lambda s: -np.log(s).abs(), "half": lambda s: -(s - 0.5).abs()}


def existence_metrics(sub: pd.DataFrame) -> dict:
    """Accuracy, F1, TPR, TNR of yes/no rows (positive = "yes"; an answer
    that is neither yes nor no counts as "no" and is counted in unparsed)."""
    ref = sub["reference_normalized"] == "yes"
    pred = sub["prediction_normalized"] == "yes"
    tp, tn = int((ref & pred).sum()), int((~ref & ~pred).sum())
    fp, fn = int((~ref & pred).sum()), int((ref & ~pred).sum())
    return {
        "n": len(sub),
        "acc": (tp + tn) / max(len(sub), 1),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "tpr": tp / max(tp + fn, 1),
        "tnr": tn / max(tn + fp, 1),
        "unparsed": int((~sub["prediction_normalized"].isin(["yes", "no"])).sum()),
    }


def box_metrics(g: pd.DataFrame) -> dict:
    """Box metrics of box rows (iou, giou, area_ratio, pred_box_parsed columns)."""
    return {"n": len(g), "mean_iou": g["iou"].mean(), "iou>=0.5": (g["iou"] >= 0.5).mean(),
            "mean_giou": g["giou"].mean(), "miss (iou=0)": (g["iou"] == 0).mean(),
            "area_ratio (median)": g["area_ratio"].median(), "parsed": g["pred_box_parsed"].mean()}


def color_rank(t: pd.DataFrame, arms: list, by="col", better=None, fmt="{:.3f}", group=None):
    """Style t with rainbow colors by rank among arms: red = best, purple = worst.
    by="col": arms are rows, rank down each column (direction from BETTER).
    by="row": arms are columns, rank across each row (direction = better).
    group: index level to rank within, e.g. "group" for seen/unseen."""
    if by == "row":
        cols = [a for a in arms if a in t.columns]
        rank = SCORE[better or "high"](t[cols].astype(float)).rank(axis=1)
    else:
        cols = [c for c in t.columns if c != "n" and pd.api.types.is_numeric_dtype(t[c])]
        s = pd.DataFrame({c: SCORE[BETTER.get(c, "high")](t[c].astype(float)) for c in cols})
        rank = s.groupby(level=group).rank() if group else s.rank()
    return (t.style.background_gradient(cmap="rainbow", subset=cols, axis=None, gmap=rank)
            .format(fmt, subset=cols, na_rep=""))


def type_mean(g: pd.DataFrame, fn, by="question_type") -> float:
    """fn(rows of one group) -> number, then the plain mean over groups, so
    each group counts the same however many rows it has. Groups are question
    types by default; by="label" for per-finding groups (VinDr).
    Example: type_mean(box_rows, lambda s: box_metrics(s)["mean_iou"])."""
    return g.groupby(by).apply(fn, include_groups=False).mean()
