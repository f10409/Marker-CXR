"""Build the Marker_CXR VQA table that `01_finetune_markercxr.py` trains and
evaluates on: loader, prompt/target formatting, drop/balance.

One row per question: (image_path, split, stage, question_type, question,
answer, used_reasoning). `stage` here is the vqa/ folder name.
"""

import json
import random
import re
from pathlib import Path

import pandas as pd

from medgemma_format import (
    REASONING_START_TOKEN,
    build_coord_reading_direct_prompt,
    build_coord_reading_plain_target,
    build_coord_reading_reasoning_prompt,
    build_coord_reading_reasoning_target,
    build_direct_prompt,
    build_plain_box_target,
    build_reasoning_prompt,
    build_reasoning_target,
    label_for_box_question,
    to_medgemma_bbox_order,
)

# --- question_type groups -------------------------------------------------

# Stage-4 types whose reasoning_frame decodes the box given in the question.
COORD_READING_TYPES = frozenset(
    {"coordinate_to_label", "coordinate_to_anatomy", "coordinate_to_label_negative"}
)

# The 6 box-producing localization_bbox types: scored by IoU, not exact match.
BOX_PRODUCING_TYPES = [
    "anatomical_region_localization",
    "finding_localization_mscxr",
    "finding_localization_mscxr_region_conditioned",
    "marker_localization",
    "marker_localization_region_conditioned",
    "marker_localization_quadrant_conditioned",
]

# Free-text answers with no small fixed answer set (grounding, negatives,
# coordinate reading) -- skipped by the answer-distribution check.
OPEN_QUESTION_TYPES = [
    "finding_grounding_imagenome",
    "marker_grounding",
    "region_to_object",
    "finding_localization_mscxr_negative",
    "marker_localization_negative",
    "coordinate_to_label",
    "coordinate_to_anatomy",
    "coordinate_to_label_negative",
]

# --- instructions appended to the question --------------------------------

YES_NO_ANCHOR_INSTRUCTION = (
    " You MUST end your response with either 'Final Answer: Yes' or "
    "'Final Answer: No'."
)
MARKER_FINDING_ANCHOR_INSTRUCTION = (
    " You MUST end your response with either 'Final Answer: Marker' or "
    "'Final Answer: Finding'."
)
# Closed-form types: fixed anchor instead of the generic short-answer one.
ANCHOR_INSTRUCTION_BY_TYPE = {
    "marker_existence": YES_NO_ANCHOR_INSTRUCTION,
    "finding_existence": YES_NO_ANCHOR_INSTRUCTION,
    "specific_label_existence": YES_NO_ANCHOR_INSTRUCTION,
    "marker_grounding_discriminative": YES_NO_ANCHOR_INSTRUCTION,
    "category_discrimination": MARKER_FINDING_ANCHOR_INSTRUCTION,
    # single_marker-only; no reasoning_frame -- nothing geometric to derive.
    "coordinate_marker_existence": YES_NO_ANCHOR_INSTRUCTION,
}
INTEGER_ANSWER_INSTRUCTION = (
    " Answer with a single integer only -- just the number, nothing else."
)
SHORT_ANSWER_INSTRUCTION = (
    " Answer in as few words as possible -- a single word or short phrase "
    "only, with no explanation."
)
SENTENCE_ANSWER_INSTRUCTION = (
    " Answer in one plain, complete sentence -- no bullet points, headings, "
    "or additional explanation."
)

_BBOX_TEXT_RE = re.compile(
    r"\[(-?\d+(?:\.\d+)?), (-?\d+(?:\.\d+)?), (-?\d+(?:\.\d+)?), (-?\d+(?:\.\d+)?)\]"
)

# Each cohort table carries its own patient_id -> split column. Overlapping
# patients were checked to get the identical split in every table, so the
# merged map cannot move a patient across train/valid/test.
SPLIT_TABLES = [
    "csvs/final_table.csv",
    "csvs/marker_free_images.csv",
    "csvs/single_marker_images.csv",
]


def load_split_map(marker_cxr_dir: Path) -> dict:
    """patient_id -> split, merged over every cohort table."""
    pid_to_split = {}
    for rel in SPLIT_TABLES:
        table = pd.read_csv(marker_cxr_dir / rel, usecols=["patient_id", "split"])
        pid_to_split.update(zip(table["patient_id"], table["split"]))
    return pid_to_split


def _format_question_answer(q: dict, is_bbox_stage: bool, use_reasoning_for_boxes: bool,
                            use_reasoning_for_coord_reading: bool) -> tuple:
    """Question with its instruction appended, and the SFT target."""
    question, answer = q["question"], q["answer"]
    expects_box_answer = is_bbox_stage and bool(_BBOX_TEXT_RE.search(answer))
    if is_bbox_stage:
        question = to_medgemma_bbox_order(question)
        answer = to_medgemma_bbox_order(answer)
    anchor_instruction = ANCHOR_INSTRUCTION_BY_TYPE.get(q["question_type"])
    reasoning_frame = q.get("reasoning_frame")
    if anchor_instruction is not None:
        question += anchor_instruction
    elif expects_box_answer:
        label = label_for_box_question(q["question_type"], q["question"])
        if use_reasoning_for_boxes and reasoning_frame is not None:
            question = build_reasoning_prompt(question)
            answer = build_reasoning_target(reasoning_frame, q["answer_struct"], label)
        else:
            question = build_direct_prompt(question)
            answer = build_plain_box_target(q["answer_struct"], label)
    elif q["question_type"] in COORD_READING_TYPES:
        if use_reasoning_for_coord_reading and reasoning_frame is not None:
            question = build_coord_reading_reasoning_prompt(question)
            answer = build_coord_reading_reasoning_target(reasoning_frame, answer)
        else:
            question = build_coord_reading_direct_prompt(question)
            answer = build_coord_reading_plain_target(answer)
    elif answer.strip().isdigit():
        question += INTEGER_ANSWER_INSTRUCTION
    elif len(answer.split()) <= 4:
        question += SHORT_ANSWER_INSTRUCTION
    else:
        question += SENTENCE_ANSWER_INSTRUCTION
    return question, answer


def build_records(
    marker_cxr_dir: Path,
    folders: list,
    use_reasoning_for_boxes: bool = True,
    use_reasoning_for_coord_reading: bool = True,
    images_root: Path | None = None,
    exclude_question_types=(),
) -> list:
    """Flatten every question in `vqa/<folder>/*.json` for the given folders.

    Each item's `image_path` is relative (<cohort dir>/<patient prefix>/
    <patient>/<study>/<file>.png); images_root is the folder that
    <cohort dir> sits under, so it can be resolved on any machine.
    """
    vqa_dir = marker_cxr_dir / "vqa"
    for folder in folders:
        if not (vqa_dir / folder).is_dir():
            raise FileNotFoundError(f"no vqa folder {vqa_dir / folder}")
    pid_to_split = load_split_map(marker_cxr_dir)
    exclude = set(exclude_question_types)
    records = []
    for stage in folders:
        is_bbox_stage = stage.endswith("_bbox")
        for fp in sorted((vqa_dir / stage).glob("*.json")):
            with open(fp) as f:
                item = json.load(f)
            split = pid_to_split[item["patient_id"]]
            image_path = item["image_path"]
            if images_root is not None:
                image_path = str(images_root / image_path)
            for q in item["questions"]:
                if q["question_type"] in exclude:
                    continue
                question, answer = _format_question_answer(
                    q, is_bbox_stage, use_reasoning_for_boxes, use_reasoning_for_coord_reading
                )
                records.append(
                    {
                        "image_path": image_path,
                        "split": split,
                        "stage": stage,
                        "question_type": q["question_type"],
                        "question": question,
                        "answer": answer,
                        # Rows whose target opens a reasoning span get the
                        # long generation pass at eval time.
                        "used_reasoning": answer.startswith(REASONING_START_TOKEN),
                    }
                )
    return records


def downsample_to_balance(records: list, question_type: str, seed: int = 42, log=print) -> list:
    """Downsample a 2-answer question_type to 1:1 by randomly dropping rows
    from the majority answer down to the minority answer's count. A type
    with no rows (its stage is switched off) is left alone."""
    others = [r for r in records if r["question_type"] != question_type]
    matched = [r for r in records if r["question_type"] == question_type]
    if not matched:
        log(f"{question_type}: no rows loaded, nothing to balance")
        return records
    by_answer = {}
    for r in matched:
        by_answer.setdefault(r["answer"], []).append(r)
    assert len(by_answer) == 2, (
        f"{question_type} has {len(by_answer)} distinct answers, expected 2"
    )
    (minority_answer, minority_rows), (majority_answer, majority_rows) = sorted(
        by_answer.items(), key=lambda kv: len(kv[1])
    )
    rng = random.Random(seed)
    balanced = minority_rows + rng.sample(majority_rows, len(minority_rows))
    log(
        f"{question_type}: {len(majority_rows)} {majority_answer!r} / "
        f"{len(minority_rows)} {minority_answer!r} -> "
        f"{len(minority_rows)} / {len(minority_rows)}"
    )
    return others + balanced


def balance_records(records: list, question_types: list, seed: int = 42, log=print) -> list:
    pre = len(records)
    for qt in question_types:
        records = downsample_to_balance(records, qt, seed=seed, log=log)
    log(f"{pre} -> {len(records)} records after balancing")
    return records


def count_by_type_and_split(records: list) -> pd.DataFrame:
    """Rows per (question_type, split) with a total column, largest first."""
    df = pd.DataFrame(records)
    table = df.pivot_table(
        index="question_type", columns="split", values="question", aggfunc="size", fill_value=0
    )
    for split in ["train", "valid", "test"]:
        if split not in table.columns:
            table[split] = 0
    table = table[["train", "valid", "test"]]
    table["total"] = table.sum(axis=1)
    return table.sort_values("total", ascending=False)
