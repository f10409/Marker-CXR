"""Verifiable rewards for GRPO on the Marker_CXR VQA tasks.

One reward function scores every question_type, dispatching on it:
  box-producing types   parse bonus + IoU + closeness of box centers
  set-answer types      set F1 (partial credit per listed item)
  everything else       exact match after the same normalization as eval

All parsing/normalization is the SAME code the eval uses (eval_scoring),
so the reward optimizes exactly the metric that metrics.json reports.
"""

import re

from bbox_metrics import box_giou, box_iou, center_distance
from eval_scoring import SET_F1_TYPES, extract_box_2d, extract_final_answer, set_f1
from markercxr_data import BOX_PRODUCING_TYPES, COORD_READING_TYPES

_BOX_TYPES = set(BOX_PRODUCING_TYPES)

# The v2 reasoning frame states the region's window as "... spans
# y = A to B and x = C to D" (0-1000 ints). Present in the reference of
# reasoning-arm box rows (except marker_localization_quadrant_conditioned,
# whose markers sit outside every region) and, if the policy keeps the
# sentence, in its generated trace.
_WINDOW_RE = re.compile(r"spans y = (\d+) to (\d+) and x = (\d+) to (\d+)")


def extract_window(text: str):
    """(y0, x0, y1, x1) in [0, 1] from the first window sentence, or None."""
    m = _WINDOW_RE.search(text)
    if m is None:
        return None
    y0, y1, x0, x1 = (int(g) / 1000.0 for g in m.groups())
    return (y0, x0, y1, x1)


def box_fraction_inside(box, window) -> float:
    """Fraction of box's area that lies inside window (0 to 1)."""
    y0 = max(box[0], window[0])
    x0 = max(box[1], window[1])
    y1 = min(box[2], window[2])
    x1 = min(box[3], window[3])
    inter = max(0.0, y1 - y0) * max(0.0, x1 - x0)
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    return inter / area if area > 0 else 0.0


def completion_text(completion) -> str:
    """The generated text of one completion. trl returns conversational
    completions as [{"role": "assistant", "content": <str or blocks>}]."""
    if isinstance(completion, str):
        return completion
    content = completion[0]["content"]
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict))


def _normalize(text: str, whole_line: bool) -> str:
    return extract_final_answer(text, whole_line=whole_line).strip().lower().rstrip(".")


def make_reward(
    parse_bonus: float = 0.2,
    iou_weight: float = 1.0,
    center_weight: float = 0.3,
    giou_weight: float = 0.0,
    window_parse_bonus: float = 0.0,
    window_weight: float = 0.0,
    window_giou_weight: float = 0.0,
    box_in_window_weight: float = 0.0,
):
    """Reward function closure over the weights (config-driven).

    Box reward = parse_bonus (box found in the output at all)
               + iou_weight * IoU(pred, ref)
               + center_weight * (1 - center distance), floored at 0.
               + giou_weight * (1 + GIoU) / 2
    GIoU is in [-1, 1]; rescaled to [0, 1] so a parsed far-miss box never
    scores below an unparseable answer (0).
    An unparseable prediction scores 0; an unparseable reference returns
    None (row dropped from the group by trl).

    Window terms (reasoning arm only, all weights default 0), active on
    rows whose reference frame has a "spans y = A to B and x = C to D"
    window:
      window_parse_bonus + window_weight * IoU(claimed, true window)
        + window_giou_weight * (1 + GIoU(claimed, true window)) / 2
        -- rewards stating the region's window correctly in the trace.
        Without its own parse bonus the policy can dodge the term by
        dropping the sentence.
      box_in_window_weight * fraction of the PREDICTED box inside the
        TRUE window -- "at least you're in the right region". Gives a
        gradient when IoU with a small target box is 0. True window only:
        overlap with the model's own claimed window would be gameable
        (claim the whole image, or copy the box into the sentence)."""

    def marker_cxr_reward(completions, answer, question_type, **kwargs) -> list:
        rewards = []
        for comp, ref, qt in zip(completions, answer, question_type):
            text = completion_text(comp)
            if qt in _BOX_TYPES:
                ref_box = extract_box_2d(ref)
                if ref_box is None:
                    rewards.append(None)
                    continue
                pred_box = extract_box_2d(text)
                if pred_box is None:
                    rewards.append(0.0)
                    continue
                r = (
                    parse_bonus
                    + iou_weight * box_iou(pred_box, ref_box)
                    + center_weight * max(0.0, 1.0 - center_distance(pred_box, ref_box))
                    + giou_weight * (1.0 + box_giou(pred_box, ref_box)) / 2
                )
                if window_parse_bonus or window_weight or window_giou_weight or box_in_window_weight:
                    ref_win = extract_window(ref)
                    if ref_win is not None:
                        pred_win = extract_window(text)
                        if pred_win is not None:
                            r += (
                                window_parse_bonus
                                + window_weight * box_iou(pred_win, ref_win)
                                + window_giou_weight * (1.0 + box_giou(pred_win, ref_win)) / 2
                            )
                        if box_in_window_weight:
                            r += box_in_window_weight * box_fraction_inside(pred_box, ref_win)
            else:
                whole_line = qt in COORD_READING_TYPES
                pred_n = _normalize(text, whole_line)
                ref_n = _normalize(ref, whole_line)
                if qt in SET_F1_TYPES:
                    r = set_f1(pred_n, ref_n)
                else:
                    r = 1.0 if pred_n == ref_n else 0.0
            rewards.append(r)
        return rewards

    return marker_cxr_reward
