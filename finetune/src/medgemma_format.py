"""MedGemma-1.5-specific formatting: prompt builders (question side) and
target builders (answer side). The VQA JSONs themselves stay
model-agnostic; this module turns them into MedGemma's format.
"""

import re

# Our stored bbox text is axis-grouped [y0, y1, x0, x1]; MedGemma's is
# interleaved [y0, x0, y1, x1]. Reordering is all that's needed since both
# use the same 0-1000 scale.
_BBOX_TEXT_RE = re.compile(
    r"\[(-?\d+(?:\.\d+)?), (-?\d+(?:\.\d+)?), (-?\d+(?:\.\d+)?), (-?\d+(?:\.\d+)?)\]"
)


def to_medgemma_bbox_order(text: str) -> str:
    return _BBOX_TEXT_RE.sub(
        lambda m: f"[{m.group(1)}, {m.group(3)}, {m.group(2)}, {m.group(4)}]", text
    )


# MedGemma 1.5's spontaneous reasoning sits between these two literal
# tokens -- empirically observed, still generic reserved vocab slots
# (ids 100/101), not a documented API.
REASONING_START_TOKEN = "<unused94>"
REASONING_END_TOKEN = "<unused95>"


# Adapted from Google's cxr_anatomy_localization_with_hugging_face.ipynb,
# with our own per-type question text slotted in directly.
def build_reasoning_prompt(question: str) -> str:
    return f"""Instructions:
The following user query will require outputting bounding boxes. The format of bounding boxes coordinates is [y0, x0, y1, x1] where (y0, x0) must be top-left corner and (y1, x1) the bottom-right corner. This implies that x0 < x1 and y0 < y1. Always normalize the x and y coordinates the range [0, 1000], meaning that a bounding box starting at 15% of the image width would be associated with an x coordinate of 150. You MUST output a single parseable json list of objects enclosed into ```json...``` brackets, for instance ```json[{{"box_2d": [800, 3, 840, 471], "label": "car"}}, {{"box_2d": [400, 22, 600, 73], "label": "dog"}}]``` is a valid output. Now answer to the user query.

Remember "left" refers to the patient's left side where the heart is and sometimes underneath an L in the upper right corner of the image.

Query:
{question} Don't give a final answer without reasoning. Output the final answer in the format "Final Answer: X" where X is a JSON list of objects. The object needs a "box_2d" and "label" key. Answer:"""


# Same schema as build_reasoning_prompt, minus the reasoning invitation --
# isolates the reasoning trace as the one difference in the
# USE_REASONING_FOR_BOXES ablation.
def build_direct_prompt(question: str) -> str:
    return f"""Instructions:
The following user query will require outputting bounding boxes. The format of bounding boxes coordinates is [y0, x0, y1, x1] where (y0, x0) must be top-left corner and (y1, x1) the bottom-right corner. This implies that x0 < x1 and y0 < y1. Always normalize the x and y coordinates the range [0, 1000], meaning that a bounding box starting at 15% of the image width would be associated with an x coordinate of 150. You MUST output a single parseable json list of objects enclosed into ```json...``` brackets, for instance ```json[{{"box_2d": [800, 3, 840, 471], "label": "car"}}, {{"box_2d": [400, 22, 600, 73], "label": "dog"}}]``` is a valid output. Now answer to the user query.

Remember "left" refers to the patient's left side where the heart is and sometimes underneath an L in the upper right corner of the image.

Query:
{question} Answer:"""


# Stage-4 siblings of the pair above: the box sits in the question, not the
# answer, so this explains how to read a box rather than how to write one.
_COORD_READING_INSTRUCTIONS = """Instructions:
The following user query refers to a location in the image given as a bounding box. The format of bounding box coordinates is [y0, x0, y1, x1] where (y0, x0) is the top-left corner and (y1, x1) the bottom-right corner. The x and y coordinates are normalized to the range [0, 1000], meaning that a bounding box starting at 15% of the image width has an x0 coordinate of 150. Look at that location in the image and answer the question about it. Now answer to the user query.

Remember "left" refers to the patient's left side where the heart is and sometimes underneath an L in the upper right corner of the image.

Query:
"""


def build_coord_reading_reasoning_prompt(question: str) -> str:
    return (
        f"{_COORD_READING_INSTRUCTIONS}{question} Don't give a final answer without "
        'reasoning. Output the final answer in the format "Final Answer: X" where X is '
        "a short phrase or a single sentence. Answer:"
    )


def build_coord_reading_direct_prompt(question: str) -> str:
    return f"{_COORD_READING_INSTRUCTIONS}{question} Answer:"


# --- Target builders (answer side) ---
# Format the SFT target from ground truth only (answer_struct,
# generator-written reasoning_frame, or the dataset's own answer string) --
# never from a model prediction.

_LABEL_RE = re.compile(r"^Where is the (.+?)(?: at the .+)?\?")


def label_for_box_question(question_type: str, question: str) -> str:
    """Object name for a box question's Final Answer JSON label."""
    if question_type.startswith("marker_localization"):
        return "marker"
    m = _LABEL_RE.match(question)
    return m.group(1) if m else "object"


def scaled_box_2d(answer_struct) -> tuple[int, int, int, int]:
    """answer_struct ([y0, y1, x0, x1], 0-1000 ints) reordered to
    MedGemma's [y0, x0, y1, x1]."""
    assert all(isinstance(v, int) for v in answer_struct), answer_struct
    y0, y1, x0, x1 = answer_struct
    return (y0, x0, y1, x1)


def box_2d_final_answer(answer_struct, label: str) -> str:
    y0, x0, y1, x1 = scaled_box_2d(answer_struct)
    return f'```json[{{"box_2d": [{y0}, {x0}, {y1}, {x1}], "label": "{label}"}}]```'


def build_reasoning_target(reasoning_frame: str, answer_struct, label: str) -> str:
    return (
        f"{REASONING_START_TOKEN}thought\n{reasoning_frame}"
        f"{REASONING_END_TOKEN}Final Answer: {box_2d_final_answer(answer_struct, label)}"
    )


def build_plain_box_target(answer_struct, label: str) -> str:
    """USE_REASONING_FOR_BOXES = False arm: same final line, no reasoning span."""
    return f"Final Answer: {box_2d_final_answer(answer_struct, label)}"


def build_coord_reading_reasoning_target(reasoning_frame: str, answer: str) -> str:
    """Mirror of build_reasoning_target: the frame decodes the question's
    own box; the final answer is the dataset's answer string verbatim."""
    return (
        f"{REASONING_START_TOKEN}thought\n{reasoning_frame}"
        f"{REASONING_END_TOKEN}Final Answer: {answer}"
    )


def build_coord_reading_plain_target(answer: str) -> str:
    """USE_REASONING_FOR_COORD_READING = False arm."""
    return f"Final Answer: {answer}"
