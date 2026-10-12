# VQA Question Templates by Stage and Type

Question templates (the JSON files in this folder hold the exact strings). Stages 3-4 each come in a
point-format stage (plain name) and a bbox-format sibling (`_bbox` suffix);
`question_type` strings are identical across the pair. Several Stage 3
templates share their exact question text with a Stage 2 grounding question
that wants a location name instead of a coordinate (e.g. "Where is the
marker?"), so a stage's plain text alone doesn't reveal what's expected —
every Stage 3 coordinate question appends `BBOX_INSTRUCTION` ("Give its full
bounding box, not just a point.") in the bbox stage or `POINT_INSTRUCTION`
("Give a point, not a location name.") in the point stage.

## Stage 1 — Existence

| `question_type` | Question | Answer |
|---|---|---|
| `marker_existence` | "Is there a marker visible in this image?" | Yes/No |
| `finding_existence` | "Does this image show a finding?" (Chest ImaGenome only) | Yes/No |
| `specific_label_existence` | "Is there evidence of {label} in this image?" — one per label in the closed Imagenome vocab | Yes/No |
| `marker_counting` | "How many markers are in this image?" | count |
| `region_conditioned_marker_counting` | "How many markers are near the {region}?" — per region with ≥1 marker | count |

"No" means the report does not mention the label, not that it was verified absent. `lung opacity` is the parent of atelectasis, consolidation, edema and pneumonia, and Chest ImaGenome lists it on every image that has one of them; the reverse is not implied (an unexplained opacity still answers "No" for each child).

## Stage 2 — Grounding

| `question_type` | Question | Answer |
|---|---|---|
| `finding_grounding_imagenome` | "Where is the {label}?" | "The {label} is located at {region(s)}." (all distinct regions the label maps to) |
| `marker_grounding` | "Where is the marker?" (1 marker) / "Where are the markers?" (>1 marker) | 1 marker: region name if the containment gate passed; else the nearest labeled region if within `NEAREST_REGION_MAX_GAP` (0.1, edge-to-edge gap) — "near the {region}, in the patient's {quadrant}"; else quadrant alone. >1: every distinct region/quadrant across all markers, joined (still quadrant-only per marker) |
| `marker_grounding_discriminative` | "Is there a marker at the {region}?" — one true per occupied region + one sampled false region | Yes/No |

## Stage 3 — Localization

(`localization` = point format, `localization_bbox` = bbox format; every
question below gets `POINT_INSTRUCTION`/`BBOX_INSTRUCTION` appended per its
stage — see the note above.)

| `question_type` | Question | Answer |
|---|---|---|
| `anatomical_region_localization` | "Where is the {region}?" — bbox-only (no point form), subsampled to 4/image | point/bbox coordinate |
| `finding_localization_mscxr` | "Where is the {label}?" — MS-CXR finding, unique in image | point/bbox |
| `finding_localization_mscxr_region_conditioned` | "Where is the {label} at the {region}?" — unique within region | point/bbox |
| `marker_localization` | "Where is the marker?" — single-marker case | point/bbox |
| `marker_localization_region_conditioned` | "Where is the marker at the {region}?" | point/bbox |
| `marker_count_region_conditioned` | "How many markers are near the {region}?" | count |
| `marker_localization_quadrant_conditioned` | "Where is the marker in the patient's {quadrant}?" — region gate failed, one-level-down fallback; only when it is the one marker in that quadrant, counting markers that passed the gate too (09/25; before, 38% had other markers there) | point/bbox |
| `marker_count_quadrant_conditioned` | "How many markers are in the patient's {quadrant}?" — 2+ markers in the quadrant; the count covers every marker there (09/25; before, 41% undercounted by leaving out gate-passing markers). 0/1 answers come from the marker_free/single_marker cohorts below | count |
| `finding_localization_mscxr_negative` | "Where is the {label}?" — a real label (appears elsewhere in the dataset) sampled from those reported by neither MS-CXR nor Chest ImaGenome on this image (09/19; before, MS-CXR alone, and 16% denied a finding ImaGenome lists); the inverse of Stage 4's `coordinate_to_label_negative` | "There is no {label} in this image." (`answer_struct: null`) |
| `marker_localization_negative` | "Where is the marker at the {region}?" — a region with no marker, sampled only when the image has markers elsewhere | "There is no marker at the {region}." (`answer_struct: null`) |

## Stage 4 — Identification

(`identification` = point format, `identification_bbox` = bbox format; the
coordinate is embedded directly in the question text and is self-describing
— a 2-number point vs. a 4-number box — so no extra instruction text is
needed.)

| `question_type` | Question | Answer |
|---|---|---|
| `region_to_object` | "What is at the {region}?" | joined label set of everything sharing that region |
| `coordinate_to_label` | "What do you see at {coord}?" | co-located marker/finding labels + ", located at the {region}" clause |
| `coordinate_to_anatomy` | "What anatomical region is at {coord}?" | region name, or "Outside the labeled anatomical regions, in the {quadrant}..." |
| `category_discrimination` | "At {coord}, is it a marker or a finding?" | "Marker."/"Finding." |
| `coordinate_to_label_negative` | "What do you see at {coord}?" — sampled box with no overlap on record | "There is no marker or finding there." |

`coordinate_to_label`/`coordinate_to_label_negative` draw only from markers
and MS-CXR findings (`collect_objects(..., include_anatomical=False)`) —
anatomical regions are never a possible answer, even though the question
itself is phrased open-endedly ("what do you see").

## Marker-free & single-marker cohort additions

Two extra image cohorts, using the same templates as above on a different
image set.

| Cohort | Images | What's generated |
|---|---|---|
| `marker_free` | 3,023 real images confirmed to have no marker at all | `marker_existence` "No" (`vqa/existence_negatives/`); `marker_localization_negative` and `marker_count_quadrant_conditioned` "0" (a random quadrant), one each per image (`vqa/marker_free_localization(_bbox)/`) |
| `single_marker` | The same 3,023 images, each with exactly one pasted marker | `marker_localization`/`_bbox`; `marker_count_quadrant_conditioned` "1" (the marker's own quadrant); `marker_counting`/`region_conditioned_marker_counting`; `marker_grounding`/`marker_grounding_discriminative`; `coordinate_to_label`/`coordinate_to_anatomy`/`coordinate_marker_existence` |

| `question_type` | Question | Answer | Notes |
|---|---|---|---|
| `marker_localization_negative` | "Where is the marker at the {region}?" — one region per image, sampled from its own anatomical labels | "There is no marker at the {region}." (`answer_struct: null`) | Only on `marker_free` images: a genuinely clean image has no marker anywhere, so the claim holds at any region. Not asked on `single_marker` images — they do have a marker, just not necessarily at the sampled region. |
| `coordinate_marker_existence` | "Is there a marker at {coord}?" | Yes at the marker's own coordinate; No at a sampled coordinate overlapping no marker/MS-CXR finding (`include_anatomical=False`, so it can still land on real anatomy) | Only on `single_marker` images, `identification`/`identification_bbox`. Replaces `coordinate_to_label_negative` there: finding columns are empty placeholders on this cohort, not "no findings," so that type's "There is no marker or finding there." can't be honestly claimed. One Yes + one No per image, so it can't teach an always-"No" shortcut. |

Left out on the `single_marker` cohort, and why: finding-based types (finding
columns are empty placeholders); `marker_existence` "Yes" (`vqa/existence/`
already has all 3,023); `category_discrimination` (would always answer
"Marker." with no findings on record); `region_to_object` (would always
answer "marker." for the same reason).

## Shared disambiguation chain

Stage 3's finding/marker templates (and Stage 2's marker grounding) resolve
a repeated/ambiguous label the same way: unique-in-image -> plain question;
unique-in-region -> region-conditioned question; still tied -> a counting
question for markers, no question for findings. Markers additionally fall
back to quadrant when the region gate fails (`region is None`).

Findings get no counting question (`finding_count_mscxr_region_conditioned`
was dropped 09/25): 99% of its rows were a "0" added once per image,
repeating `finding_localization_mscxr_negative`, and its few real counts
could undercount, since MS-CXR boxes only some of a report's findings.
