"""Box-geometry metrics for scoring MedGemma's coordinate predictions:
parsing a prediction into (y0, x0, y1, x1), IoU, and center distance.
"""

import re

import pandas as pd

# Plain string for .str.contains row-selection; compiled version below
# with capture groups for parse_box() to pull the 4 numbers out.
BRACKET_PATTERN = (
    r"\[\s*-?\d*\.?\d+\s*,\s*-?\d*\.?\d+\s*,\s*-?\d*\.?\d+\s*,\s*-?\d*\.?\d+\s*\]"
)
BRACKET_RE = re.compile(
    r"\[\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*\]"
)
JSON_XY_RE = re.compile(
    r'"?x1"?\s*:\s*(-?\d*\.?\d+).*?"?y1"?\s*:\s*(-?\d*\.?\d+).*?'
    r'"?x2"?\s*:\s*(-?\d*\.?\d+).*?"?y2"?\s*:\s*(-?\d*\.?\d+)',
    re.DOTALL,
)


def parse_box(text: str, image_width: float, image_height: float):
    """Best-effort (y0, x0, y1, x1) box normalized to [0, 1] from free-form
    text: either a [y0, x0, y1, x1] bracket, or a pixel-space
    {"x1","y1","x2","y2"} JSON blob. None if neither is found."""
    m = BRACKET_RE.search(text)
    if m:
        y0, x0, y1, x1 = (float(g) for g in m.groups())
        if max(y0, x0, y1, x1) > 1.5:
            y0, x0, y1, x1 = (v / 1000.0 for v in (y0, x0, y1, x1))
        return (y0, x0, y1, x1)

    m = JSON_XY_RE.search(text)
    if m:
        x0, y0, x1, y1 = (float(g) for g in m.groups())
        return (
            y0 / image_height,
            x0 / image_width,
            y1 / image_height,
            x1 / image_width,
        )

    return None


def box_iou(box_a, box_b) -> float:
    ay0, ax0, ay1, ax1 = box_a
    by0, bx0, by1, bx1 = box_b
    inter_y0, inter_x0 = max(ay0, by0), max(ax0, bx0)
    inter_y1, inter_x1 = min(ay1, by1), min(ax1, bx1)
    inter = max(0.0, inter_y1 - inter_y0) * max(0.0, inter_x1 - inter_x0)
    area_a = max(0.0, ay1 - ay0) * max(0.0, ax1 - ax0)
    area_b = max(0.0, by1 - by0) * max(0.0, bx1 - bx0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def box_giou(box_a, box_b) -> float:
    """Generalized IoU in [-1, 1]: IoU minus the share of the smallest
    enclosing box that neither box covers. Unlike IoU, still tells apart a
    near miss from a far miss when the boxes don't overlap."""
    ay0, ax0, ay1, ax1 = box_a
    by0, bx0, by1, bx1 = box_b
    inter = max(0.0, min(ay1, by1) - max(ay0, by0)) * max(0.0, min(ax1, bx1) - max(ax0, bx0))
    area_a = max(0.0, ay1 - ay0) * max(0.0, ax1 - ax0)
    area_b = max(0.0, by1 - by0) * max(0.0, bx1 - bx0)
    union = area_a + area_b - inter
    hull = (max(ay1, by1) - min(ay0, by0)) * (max(ax1, bx1) - min(ax0, bx0))
    if union <= 0 or hull <= 0:
        return -1.0
    return inter / union - (hull - union) / hull


def box_area_ratio(pred, ref) -> float:
    """Predicted box area / true box area. 1 = right size, 10 = ten times
    too big, 0.1 = ten times too small. NaN if the true box has no area."""
    area = lambda b: max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return area(pred) / area(ref) if area(ref) > 0 else float("nan")


def box_center(box):
    y0, x0, y1, x1 = box
    return ((y0 + y1) / 2, (x0 + x1) / 2)


def center_distance(box_a, box_b) -> float:
    ay, ax = box_center(box_a)
    by, bx = box_center(box_b)
    return ((ay - by) ** 2 + (ax - bx) ** 2) ** 0.5


def score_row(row):
    """Row-wise .apply target over predicted_box/reference_box columns.
    Unparsed prediction scores IoU 0, no center distance."""
    if row["predicted_box"] is None:
        return pd.Series({"iou": 0.0, "center_distance": float("nan")})
    return pd.Series(
        {
            "iou": box_iou(row["predicted_box"], row["reference_box"]),
            "center_distance": center_distance(
                row["predicted_box"], row["reference_box"]
            ),
        }
    )
