"""Shared MedGemma-1.5 evaluation helpers: parsing a raw generation down to
its scored answer, batched two-pass generation, and exact-match/macro-F1
scoring.
"""

import json
import re

import pandas as pd
from sklearn.metrics import f1_score
from tqdm.auto import tqdm

from bbox_metrics import box_area_ratio, box_giou, box_iou, center_distance


def strip_thinking(text: str) -> str:
    # MedGemma 1.5's reasoning trace sits between <unused94>/<unused95>;
    # only the closing marker matters for parsing the final answer.
    if "<unused95>" in text:
        return text.split("<unused95>", 1)[1].lstrip()
    return text


_JSON_BOX_RE = re.compile(
    r'"box_2d"\s*:\s*\[\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*\]'
)
_BRACKET_RE = re.compile(
    r"\[\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*\]"
)


def extract_box_2d(text: str):
    """Best-effort (y0, x0, y1, x1) off a box-producing response, normalized
    to [0, 1] regardless of scale or schema. None if nothing parseable is
    found."""
    text = strip_thinking(text)
    box = None
    json_start, json_end = text.find("```json"), text.rfind("```")
    if json_start != -1 and json_end > json_start:
        try:
            parsed = json.loads(text[json_start + len("```json") : json_end].strip())
            if isinstance(parsed, list) and parsed:
                candidate = parsed[0].get("box_2d")
                if isinstance(candidate, list) and len(candidate) == 4:
                    box = tuple(float(v) for v in candidate)
        except (json.JSONDecodeError, KeyError, ValueError, TypeError, IndexError):
            pass
    if box is None:
        m = _JSON_BOX_RE.search(text)
        box = tuple(float(g) for g in m.groups()) if m else None
    if box is None:
        m = _BRACKET_RE.search(text)  # plain bracket, no JSON key at all
        box = tuple(float(g) for g in m.groups()) if m else None
    if box is None:
        return None
    if max(box) > 1.5:  # anything past 1.5 is the native 0-1000 scale
        box = tuple(v / 1000.0 for v in box)
    return box


def box_prediction_to_sentence(text: str) -> str:
    box = extract_box_2d(text)
    if box is None:
        return strip_thinking(text)  # leave unparsed text visible for debugging
    y0, x0, y1, x1 = box
    return f"At [{round(y0 * 1000)}, {round(x0 * 1000)}, {round(y1 * 1000)}, {round(x1 * 1000)}]."  # 0-1000 ints, like the references


def format_prompt(example: dict) -> dict:
    example["prompt_messages"] = [
        {
            "role": "user",
            "content": [
                {"type": "image", "path": example["image_path"]},
                {"type": "text", "text": example["question"]},
            ],
        },
    ]
    return example


def generate_with_reasoning_split(
    pipe,
    prompt_messages: list,
    reasoning_flags: list,
    regular_tokens: int = 100,
    regular_batch: int = 16,
    reasoning_tokens: int = 1000,
    reasoning_batch: int = 8,
) -> list:
    """Two-pass batched generation split on reasoning_flags: True rows get
    reasoning_tokens/reasoning_batch (room for a trace, smaller batch to
    avoid OOM); everything else gets the cheap regular pass. Chunks
    manually so tqdm can show progress per batch."""

    def run(indices: list, max_new_tokens: int, batch_size: int) -> dict:
        out = {}
        desc = f"Generating ({max_new_tokens} tok)"
        for start in tqdm(range(0, len(indices), batch_size), desc=desc):
            batch_idx = indices[start : start + batch_size]
            batch = [prompt_messages[i] for i in batch_idx]
            outputs = pipe(
                text=batch,
                max_new_tokens=max_new_tokens,
                batch_size=len(batch),
                return_full_text=False,
            )
            for i, out_i in zip(batch_idx, outputs):
                out[i] = out_i[0]["generated_text"].strip()
        return out

    regular_idx = [i for i, r in enumerate(reasoning_flags) if not r]
    reasoning_idx = [i for i, r in enumerate(reasoning_flags) if r]
    raw = {}
    raw.update(run(regular_idx, regular_tokens, regular_batch))
    raw.update(run(reasoning_idx, reasoning_tokens, reasoning_batch))
    return [raw[i] for i in range(len(prompt_messages))]


_FINAL_ANSWER_WORD_RE = re.compile(r"final answer:\s*(\w+)", re.IGNORECASE)
# DOTALL + optional space before the colon -- takes everything to end of string.
_FINAL_ANSWER_LINE_RE = re.compile(r"final answer\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)


def extract_final_answer(text: str, whole_line: bool = False) -> str:
    """Text after the last 'Final Answer:' anchor. whole_line=True takes
    everything after it; otherwise just the first word. Unchanged if no
    anchor is present."""
    if not isinstance(text, str):
        return text
    pattern = _FINAL_ANSWER_LINE_RE if whole_line else _FINAL_ANSWER_WORD_RE
    matches = pattern.findall(text)
    return matches[-1].strip() if matches else text


def normalize_answer(s: pd.Series, whole_line=False) -> pd.Series:
    """Extract + strip/lowercase/de-period an answer Series. whole_line is
    one bool for the whole Series, or a same-length Series/list of per-row
    bools for a frame mixing anchor- and phrase-style answers."""
    if isinstance(whole_line, bool):
        extracted = s.apply(lambda t: extract_final_answer(t, whole_line=whole_line))
    else:
        extracted = pd.Series(
            [extract_final_answer(t, whole_line=wl) for t, wl in zip(s, whole_line)],
            index=s.index,
            dtype="object",
        )
    return extracted.str.strip().str.lower().str.rstrip(".")


def macro_f1(group: pd.DataFrame, whole_line_types=frozenset()) -> float:
    """Macro F1 over one groupby("question_type") group. whole_line_types
    is a bool for every group, or a set of question_types to check against
    group.name (not a column -- include_groups=False strips it)."""
    whole_line = (
        whole_line_types if isinstance(whole_line_types, bool) else group.name in whole_line_types
    )
    return f1_score(
        normalize_answer(group["reference"], whole_line),
        normalize_answer(group["prediction"], whole_line),
        average="macro",
        zero_division=0,
    )


def score_rows(
    predictions,
    references,
    question_types,
    whole_line_types: frozenset = frozenset(),
) -> pd.DataFrame:
    """One row per prediction with its normalized answers and `correct`
    flag. whole_line_types get whole-line Final Answer: extraction;
    everything else, single-word."""
    results = pd.DataFrame(
        {
            "prediction": list(predictions),
            "reference": list(references),
            "question_type": list(question_types),
        }
    )
    whole_line_mask = results["question_type"].isin(whole_line_types)
    results["prediction_normalized"] = normalize_answer(results["prediction"], whole_line_mask)
    results["reference_normalized"] = normalize_answer(results["reference"], whole_line_mask)
    results["correct"] = results["prediction_normalized"] == results["reference_normalized"]
    return results


def score(
    predictions,
    references,
    question_types,
    whole_line_types: frozenset = frozenset(),
) -> tuple:
    """Exact-match accuracy, overall and by question_type (see score_rows)."""
    results = score_rows(predictions, references, question_types, whole_line_types)
    overall = results["correct"].mean()
    by_type = results.groupby("question_type")["correct"].mean().sort_values()
    return overall, by_type


# Types whose answer lists several labels/locations, e.g. "the markers are
# located at the left clavicle, the left lung, and the left mid lung zone".
# Scored by set F1 over those items, so one wrong item isn't a full miss.
SET_F1_TYPES = frozenset({"marker_grounding", "finding_grounding_imagenome", "coordinate_to_label"})

_LOCATED_RE = re.compile(r"^(.*?),?\s+(?:is |are )?located (?:at|in|near)\s+(.*)$")
_ITEM_SPLIT_RE = re.compile(r",\s*(?:and\s+|in\s+)?|\s+and\s+")


def answer_items(text: str) -> set:
    """Normalized answer -> set of items: "label:<x>" for each label before
    "located", plus each location after it. "there is no ..." -> empty set.
    Labels are kept only for the "marker, located at ..." style (no leading
    "the"), since in "the markers are located ..." the label is given."""
    if not isinstance(text, str) or text.startswith("there is no"):
        return set()
    m = _LOCATED_RE.match(text)
    if not m:
        # Bare label, e.g. coordinate_to_label answering just "marker".
        return {f"label:{x}" for x in _ITEM_SPLIT_RE.split(text) if x}
    head, locs = m.groups()
    items = {x.strip() for x in _ITEM_SPLIT_RE.split(locs) if x.strip()}
    if not head.startswith("the "):
        items |= {f"label:{x}" for x in _ITEM_SPLIT_RE.split(head) if x}
    return items


def set_f1(pred: str, ref: str) -> float:
    """F1 between the item sets of two normalized answers. Both empty
    (both say "there is no ...") -> 1.0."""
    p, r = answer_items(pred), answer_items(ref)
    if not p and not r:
        return 1.0
    tp = len(p & r)
    return 2 * tp / (len(p) + len(r))


def box_geometry(predictions, references, question_types, box_producing_types) -> tuple:
    """Per-row IoU and center distance against `box_producing_types` only
    (NaN elsewhere -- score()'s exact match already covers those). An
    unparsed prediction or reference scores IoU 0 and no center distance."""
    ious, dists = [], []
    for pred, ref, qt in zip(predictions, references, question_types):
        if qt not in box_producing_types:
            ious.append(float("nan"))
            dists.append(float("nan"))
            continue
        pred_box, ref_box = extract_box_2d(pred), extract_box_2d(ref)
        if pred_box is None or ref_box is None:
            ious.append(0.0)
            dists.append(float("nan"))
            continue
        ious.append(box_iou(pred_box, ref_box))
        dists.append(center_distance(pred_box, ref_box))
    return ious, dists


def box_gious(predictions, references, question_types, box_producing_types) -> list:
    """Per-row GIoU, same rows as box_geometry (NaN for non-box types). An
    unparsed prediction or reference scores -1, the GIoU minimum (like IoU 0)."""
    out = []
    for pred, ref, qt in zip(predictions, references, question_types):
        if qt not in box_producing_types:
            out.append(float("nan"))
            continue
        pred_box, ref_box = extract_box_2d(pred), extract_box_2d(ref)
        out.append(-1.0 if pred_box is None or ref_box is None else box_giou(pred_box, ref_box))
    return out


def box_area_ratios(predictions, references, question_types, box_producing_types) -> list:
    """Per-row predicted/true box area ratio (NaN for non-box types and
    unparsed boxes)."""
    out = []
    for pred, ref, qt in zip(predictions, references, question_types):
        pred_box = extract_box_2d(pred) if qt in box_producing_types else None
        ref_box = extract_box_2d(ref) if pred_box is not None else None
        out.append(float("nan") if ref_box is None else box_area_ratio(pred_box, ref_box))
    return out
