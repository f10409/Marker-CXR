#!/usr/bin/env python
"""Fine-tune / evaluate MedGemma 1.5 on the Marker_CXR VQA data from a config
file.

    python 01_finetune_markercxr.py data  --config configs/sft.yaml
    python 01_finetune_markercxr.py train --config configs/sft.yaml
    python 01_finetune_markercxr.py eval  --config configs/sft.yaml
    python 01_finetune_markercxr.py eval  --config configs/sft.yaml --baseline

`data` only builds the tables and prints row counts (no GPU). Every command
accepts `--set key.path=value` overrides, e.g. `--set run_name=no_grounding
--set stages.grounding.enabled=false`. See configs/sft.yaml for every key.
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from markercxr_data import (  # noqa: E402
    BOX_PRODUCING_TYPES,
    COORD_READING_TYPES,
    balance_records,
    build_records,
    count_by_type_and_split,
)


def dist_info() -> tuple:
    """(local_rank, rank, world_size). `accelerate launch`/`torchrun` set
    these in every worker's environment; a plain `python ...` run has none
    of them and gets (0, 0, 1)."""
    return (
        int(os.environ.get("LOCAL_RANK", 0)),
        int(os.environ.get("RANK", 0)),
        int(os.environ.get("WORLD_SIZE", 1)),
    )


def log(*args):
    """Printed by rank 0 only -- under DDP every worker runs this same code
    top to bottom, so unguarded prints come out 8x interleaved."""
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, flush=True)


# --- config ----------------------------------------------------------------


def parse_override(text: str) -> tuple:
    if "=" not in text:
        raise SystemExit(f"--set expects key.path=value, got {text!r}")
    key, value = text.split("=", 1)
    return key.strip(), yaml.safe_load(value)


def apply_override(cfg: dict, key: str, value):
    """Set a dotted key that already exists in the config (typo guard)."""
    node = cfg
    parts = key.split(".")
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            raise SystemExit(f"--set {key}: no such config key {part!r}")
        node = node[part]
    if not isinstance(node, dict) or parts[-1] not in node:
        raise SystemExit(f"--set {key}: no such config key {parts[-1]!r}")
    node[parts[-1]] = value


def load_config(path: str, overrides: list) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for text in overrides:
        apply_override(cfg, *parse_override(text))
    return cfg


def resolve_path(p: str) -> Path:
    """Relative paths are taken from this script's folder, not the cwd."""
    p = Path(p).expanduser()
    return p.resolve() if p.is_absolute() else (HERE / p).resolve()


def run_dir(cfg: dict) -> Path:
    return resolve_path(cfg["output_root"]) / cfg["run_name"]


def stage_folders(cfg: dict, which: str) -> list:
    """Folders of the enabled stages ("enabled") or of every stage ("all")."""
    folders = []
    for name, stage in cfg["stages"].items():
        if which == "all" or stage["enabled"]:
            folders.extend(stage["folders"])
    if not folders:
        raise SystemExit("no stage is enabled -- turn at least one on")
    return folders


def enabled_stage_names(cfg: dict) -> list:
    return [n for n, s in cfg["stages"].items() if s["enabled"]]


# --- data -----------------------------------------------------------------


def build_table(cfg: dict, folders: list, exclude_question_types=()) -> list:
    """Records for the given folders, with drop/balance applied."""
    d = cfg["data"]
    images_root = cfg["paths"]["images_root"]
    records = build_records(
        resolve_path(cfg["paths"]["marker_cxr_dir"]),
        folders,
        use_reasoning_for_boxes=d["use_reasoning_for_boxes"],
        use_reasoning_for_coord_reading=d["use_reasoning_for_coord_reading"],
        images_root=resolve_path(images_root) if images_root else None,
        exclude_question_types=exclude_question_types,
    )
    log(f"{len(records)} records from {len(folders)} folders: {folders}")
    return balance_records(records, d["balance_question_types"], seed=d["seed"], log=log)


def training_records(cfg: dict) -> list:
    d = cfg["data"]
    records = build_table(cfg, stage_folders(cfg, "enabled"), d["exclude_question_types"])
    if d.get("subsample_fraction") is not None:
        records = subsample_by_image(records, d["subsample_question_types"],
                                     d["subsample_fraction"], d["seed"])
    return records


def subsample_by_image(records: list, question_types: list, fraction: float, seed: int) -> list:
    """Keep the train rows of `question_types` for only a random `fraction`
    of the images that have them; all other rows (other types, valid, test)
    stay. One seeded shuffle, so a smaller fraction's images are a subset of
    a larger one's (25% is inside 50%)."""
    import random

    types = set(question_types)
    hit = lambda r: r["split"] == "train" and r["question_type"] in types
    images = sorted({r["image_path"] for r in records if hit(r)})
    random.Random(seed).shuffle(images)
    keep = set(images[: round(fraction * len(images))])
    out = [r for r in records if not hit(r) or r["image_path"] in keep]
    log(f"subsample {fraction:.0%} of {len(images)} images for {len(types)} types: "
        f"{sum(map(hit, records))} -> {sum(map(hit, out))} train rows")
    return out


def test_records(cfg: dict) -> list:
    """Test rows for eval. eval.stages="all" scores every stage, so each
    ablation arm is scored on the same rows."""
    e = cfg["eval"]
    records = build_table(cfg, stage_folders(cfg, e["stages"]))
    records = [r for r in records if r["split"] == "test"]
    if e["question_types"]:
        keep = set(e["question_types"])
        records = [r for r in records if r["question_type"] in keep]
    return records


def to_dataset_dict(records: list, cfg: dict):
    from datasets import Dataset, DatasetDict

    full = Dataset.from_list(records)
    # `s=s` pins the loop variable per closure (late-binding bug otherwise).
    data = DatasetDict(
        {
            s: full.filter(lambda r, s=s: r["split"] == s, desc=f"split={s}")
            for s in ["train", "valid", "test"]
        }
    )
    d = cfg["data"]
    if d["max_train_examples"] is not None and len(data["train"]) > d["max_train_examples"]:
        data["train"] = data["train"].shuffle(seed=d["seed"]).select(range(d["max_train_examples"]))
    return data


def format_data(example: dict) -> dict:
    example["messages"] = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": example["question"]}],
        },
        {"role": "assistant", "content": [{"type": "text", "text": example["answer"]}]},
    ]
    return example


def cmd_data(cfg: dict, args):
    log("== training table (enabled stages:", enabled_stage_names(cfg), ")")
    train_recs = training_records(cfg)
    log(count_by_type_and_split(train_recs).to_string())
    log()
    log(f"== test table for eval (eval.stages={cfg['eval']['stages']})")
    test_recs = test_records(cfg)
    log(count_by_type_and_split(test_recs)[["test"]].to_string())
    log(f"\n{len(test_recs)} test rows")


# --- model -----------------------------------------------------------------


def require_hf_token():
    from huggingface_hub import get_token

    if get_token() is None:
        raise SystemExit(
            "No Hugging Face token found. Run `huggingface-cli login` once, or "
            "export HF_TOKEN=... in the job script (MedGemma is a gated model)."
        )


def load_processor(model_id: str, padding_side: str):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = padding_side
    return processor


def make_collate_fn(processor):
    from PIL import Image

    def collate_fn(examples: list):
        texts, prompt_texts, images = [], [], []
        for example in examples:
            images.append([Image.open(example["image_path"]).convert("RGB")])
            texts.append(
                processor.apply_chat_template(
                    example["messages"], add_generation_prompt=False, tokenize=False
                ).strip()
            )
            # Prompt-only rendering, tokenized below just to count how many
            # leading tokens to mask out of the loss. Not stripped: the
            # newline after "<start_of_turn>model" is part of the prompt.
            prompt_texts.append(
                processor.apply_chat_template(
                    example["messages"][:1], add_generation_prompt=True, tokenize=False
                )
            )
        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

        # Loss only on the answer tokens: mask padding, image tokens and the
        # whole prompt (otherwise the model also learns to copy its input).
        labels = batch["input_ids"].clone()
        image_token_id = [
            processor.tokenizer.convert_tokens_to_ids(
                processor.tokenizer.special_tokens_map["boi_token"]
            )
        ]
        labels[labels == processor.tokenizer.pad_token_id] = -100
        labels[labels == image_token_id] = -100
        labels[labels == 262144] = -100
        # Right padding keeps each row's real tokens at the front, so the
        # first prompt_len positions are exactly the prompt.
        for i, (prompt_text, image) in enumerate(zip(prompt_texts, images)):
            prompt_len = processor(
                text=[prompt_text], images=[image], return_tensors="pt"
            )["input_ids"].shape[1]
            labels[i, :prompt_len] = -100
        batch["labels"] = labels
        return batch

    return collate_fn


def cmd_train(cfg: dict, args):
    import math

    import torch
    from peft import LoraConfig
    from transformers import AutoModelForImageTextToText, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    require_hf_token()
    local_rank, rank, world_size = dist_info()
    # Asked about this rank's own GPU, not the default index 0.
    if torch.cuda.get_device_capability(local_rank)[0] < 8:
        raise SystemExit("GPU does not support bfloat16; use an Ampere or newer GPU.")

    out = run_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        with open(out / "config.yaml", "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

    # Every rank builds the same table (sorted globs + seeded downsampling,
    # so the row order is identical) -- DDP's sampler shards it by index,
    # which only lines up if all ranks hold the same rows in the same order.
    records = training_records(cfg)
    summary = count_by_type_and_split(records)
    if rank == 0:
        summary.to_csv(out / "data_summary.csv")
    log(summary.to_string())
    data = to_dataset_dict(records, cfg).map(format_data)
    log(data)

    model_id = cfg["model"]["model_id"]
    t = cfg["train"]
    # Single process: "auto" lets accelerate place the model (one GPU here,
    # since a 4-bit 4B fits). Under DDP each rank holds a full copy on its
    # own GPU -- a multi-device device_map would be model parallelism, which
    # Trainer refuses to combine with distributed training.
    device_map = {"": local_rank} if world_size > 1 else "auto"
    model_kwargs = dict(
        attn_implementation="eager", dtype=torch.bfloat16, device_map=device_map
    )
    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16,
    )
    model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
    processor = load_processor(model_id, padding_side="right")

    peft_config = LoraConfig(
        lora_alpha=t["lora"]["alpha"],
        lora_dropout=t["lora"]["dropout"],
        r=t["lora"]["r"],
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        modules_to_save=["lm_head", "embed_tokens"],
    )

    # Steps counted after gradient accumulation and across ranks, like
    # Trainer's own warmup: one optimizer step consumes
    # per_device_batch x grad_accum x world_size rows.
    effective_batch = (
        t["per_device_train_batch_size"] * t["gradient_accumulation_steps"] * world_size
    )
    steps_per_epoch = math.ceil(len(data["train"]) / effective_batch)
    warmup_steps = math.ceil(t["warmup_ratio"] * steps_per_epoch * t["num_train_epochs"])
    log(
        f"world_size={world_size}  effective batch={effective_batch} "
        f"({t['per_device_train_batch_size']} x {t['gradient_accumulation_steps']} accum "
        f"x {world_size} ranks)  steps/epoch={steps_per_epoch}  warmup={warmup_steps}"
    )
    if world_size > 1:
        # Arms trained with different GPU counts are not comparable unless
        # the effective batch matches; scale per_device batch or grad_accum
        # down by world_size to keep it.
        log(
            f"note: on 1 GPU this config is effective batch "
            f"{effective_batch // world_size}; arms trained at different "
            f"effective batches are not directly comparable"
        )

    sft_args = SFTConfig(
        output_dir=str(out / "adapter"),
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        logging_steps=t["logging_steps"],
        save_strategy=t["save_strategy"],
        save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"],
        eval_strategy="steps",
        eval_steps=t["eval_steps"],
        learning_rate=float(t["learning_rate"]),
        bf16=True,
        max_grad_norm=t["max_grad_norm"],
        warmup_steps=warmup_steps,
        lr_scheduler_type="linear",
        push_to_hub=False,  # MIMIC-CXR-derived weights stay local
        report_to="tensorboard",
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        label_names=["labels"],
        # Every LoRA parameter gets a gradient on every step, so DDP does not
        # need the extra backward-graph traversal that looks for unused ones.
        ddp_find_unused_parameters=False,
    )

    n_valid = min(cfg["data"]["max_valid_examples"], len(data["valid"]))
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=data["train"],
        eval_dataset=data["valid"].shuffle(seed=cfg["data"]["seed"]).select(range(n_valid)),
        peft_config=peft_config,
        processing_class=processor,
        data_collator=make_collate_fn(processor),
    )
    # Resume from the newest adapter/checkpoint-N if an earlier job was
    # killed (e.g. by the time limit) before training finished.
    from transformers.trainer_utils import get_last_checkpoint

    last_ckpt = get_last_checkpoint(sft_args.output_dir) if Path(sft_args.output_dir).is_dir() else None
    if last_ckpt:
        log(f"resuming from {last_ckpt}")
    trainer.train(resume_from_checkpoint=last_ckpt)
    trainer.save_model()  # Trainer writes from the main process only
    if rank == 0:
        with open(out / "train_log.json", "w") as f:
            json.dump(trainer.state.log_history, f, indent=1)
    log(f"adapter saved to {out / 'adapter'}")


# --- eval ------------------------------------------------------------------


def load_pipeline(model_id: str, adapter_dir: Path | None):
    import torch
    from transformers import AutoModelForImageTextToText, pipeline

    # Left padding for batched generation: generate() continues from the
    # last column, which must be each row's own last real token.
    processor = load_processor(model_id, padding_side="left")
    # One card, not "auto". The model is 4B in bf16 (~8GB) and fits on any
    # GPU that can train it; "auto" with several cards visible splits it
    # across them, so each generated token hops card to card and only one
    # computes at a time -- measured 621 s per batch of 16 on 4 L40S.
    # Index 0 of CUDA_VISIBLE_DEVICES, so the shell still picks which card.
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map={"": 0}
    )
    if adapter_dir is not None:
        # PeftModel.from_pretrained renames the fully fine-tuned
        # modules_to_save (lm_head, embed_tokens) correctly;
        # pipeline(model=<adapter dir>) silently skips them.
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_dir))
    pipe = pipeline("image-text-to-text", model=model, processor=processor)
    pipe.model.generation_config.do_sample = False
    pipe.model.generation_config.pad_token_id = processor.tokenizer.eos_token_id
    pipe.model.generation_config.max_length = None  # max_new_tokens is passed per call
    return pipe


def metrics_by_type(rows) -> "pd.DataFrame":
    """Mean metrics per question_type plus an OVERALL row. IoU columns are
    only defined for the box-producing types (NaN elsewhere)."""
    import pandas as pd

    def summarize(group):
        box = group["iou"].notna()
        return pd.Series(
            {
                "n": len(group),
                "exact_match": group["correct"].mean(),
                "mean_set_f1": group["set_f1"].mean(),  # NaN outside SET_F1_TYPES
                "mean_iou": group.loc[box, "iou"].mean() if box.any() else float("nan"),
                "frac_iou_ge_0.5": (group.loc[box, "iou"] >= 0.5).mean() if box.any() else float("nan"),
                "mean_giou": group.loc[box, "giou"].mean() if box.any() else float("nan"),
                # pred/true box area; 1 = right size. Median: ratios are skewed.
                "median_area_ratio": group.loc[box, "area_ratio"].median() if box.any() else float("nan"),
                "median_center_dist": group.loc[box, "center_dist"].median() if box.any() else float("nan"),
                "box_parsed_frac": group.loc[box, "pred_box_parsed"].mean() if box.any() else float("nan"),
            }
        )

    by_type = rows.groupby("question_type").apply(summarize, include_groups=False)
    overall = summarize(rows).to_frame("OVERALL").T
    table = pd.concat([by_type.sort_index(), overall])
    table["n"] = table["n"].astype(int)
    return table


def row_to_json(row) -> dict:
    import pandas as pd

    return {k: (None if pd.isna(v) else (int(v) if k == "n" else float(v))) for k, v in row.items()}


def add_set_f1(rows) -> None:
    """Adds a set_f1 column: per-row set F1 for SET_F1_TYPES, NaN elsewhere."""
    from eval_scoring import SET_F1_TYPES, set_f1

    rows["set_f1"] = [
        set_f1(p, r) if qt in SET_F1_TYPES else float("nan")
        for p, r, qt in zip(rows["prediction_normalized"], rows["reference_normalized"], rows["question_type"])
    ]


def write_metrics(out: Path, rows, info: dict) -> None:
    """Writes metrics_by_type.csv and metrics.json (info + per-type table)."""
    table = metrics_by_type(rows)
    table.to_csv(out / "metrics_by_type.csv", index_label="question_type")
    metrics = {
        **info,
        "n_test": int(len(rows)),
        "overall": row_to_json(table.loc["OVERALL"]),
        "by_type": {qt: row_to_json(r) for qt, r in table.drop(index="OVERALL").iterrows()},
    }
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=1)
    log(table.round(3).to_string())


def cmd_eval(cfg: dict, args):
    import pandas as pd
    from datasets import Dataset

    from eval_scoring import (
        box_geometry,
        box_area_ratios,
        box_gious,
        extract_box_2d,
        format_prompt,
        generate_with_reasoning_split,
        score_rows,
    )

    require_hf_token()
    e = cfg["eval"]
    model_id = cfg["model"]["model_id"]

    if args.baseline:
        adapter_dir = None
        out = Path(args.out) if args.out else resolve_path(cfg["output_root"]) / "baseline" / "eval"
    else:
        adapter_dir = Path(args.adapter) if args.adapter else run_dir(cfg) / "adapter"
        if not (adapter_dir / "adapter_config.json").exists():
            raise SystemExit(f"no adapter at {adapter_dir} -- run `train` first or pass --adapter")
        out = Path(args.out) if args.out else run_dir(cfg) / "eval" / "finetuned"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    records = test_records(cfg)
    test_data = Dataset.from_list(records)
    if e["test_sample_size"] is not None and len(test_data) > e["test_sample_size"]:
        test_data = test_data.shuffle(seed=cfg["data"]["seed"]).select(range(e["test_sample_size"]))
    test_data = test_data.map(format_prompt)
    log(f"{len(test_data)} test rows; adapter={adapter_dir or 'none (baseline)'}")

    pipe = load_pipeline(model_id, adapter_dir)
    predictions = generate_with_reasoning_split(
        pipe,
        list(test_data["prompt_messages"]),
        list(test_data["used_reasoning"]),
        regular_tokens=e["regular_tokens"],
        regular_batch=e["regular_batch"],
        reasoning_tokens=e["reasoning_tokens"],
        reasoning_batch=e["reasoning_batch"],
    )

    references = list(test_data["answer"])
    question_types = list(test_data["question_type"])
    scored = score_rows(predictions, references, question_types, whole_line_types=COORD_READING_TYPES)
    ious, dists = box_geometry(predictions, references, question_types, BOX_PRODUCING_TYPES)
    gious = box_gious(predictions, references, question_types, BOX_PRODUCING_TYPES)
    area_ratios = box_area_ratios(predictions, references, question_types, BOX_PRODUCING_TYPES)

    rows = pd.DataFrame(
        {
            "image_path": test_data["image_path"],
            "stage": test_data["stage"],
            "question_type": question_types,
            "question": test_data["question"],
            "reference": references,
            "prediction": predictions,
            "used_reasoning": test_data["used_reasoning"],
            "prediction_normalized": scored["prediction_normalized"],
            "reference_normalized": scored["reference_normalized"],
            "correct": scored["correct"],
            "iou": ious,
            "giou": gious,
            "area_ratio": area_ratios,
            "center_dist": dists,
        }
    )
    is_box = rows["question_type"].isin(BOX_PRODUCING_TYPES)
    rows["pred_box_parsed"] = [
        (extract_box_2d(p) is not None) if b else None for p, b in zip(predictions, is_box)
    ]
    add_set_f1(rows)

    # Per-row predictions: one file with everything (for bootstrapping
    # confidence intervals), plus one file per question_type.
    rows.to_csv(out / "predictions.csv", index=False)
    by_type_dir = out / "predictions_by_type"
    by_type_dir.mkdir(exist_ok=True)
    for qt, sub in rows.groupby("question_type"):
        sub.to_csv(by_type_dir / f"{qt}.csv", index=False)

    write_metrics(
        out,
        rows,
        {
            "model_id": model_id,
            "adapter": str(adapter_dir) if adapter_dir else None,
            "run_name": None if args.baseline else cfg["run_name"],
            "trained_stages": None if args.baseline else enabled_stage_names(cfg),
        },
    )
    log(f"wrote {out}/{{predictions.csv, predictions_by_type/, metrics_by_type.csv, metrics.json}}")


def cmd_rescore(cfg: dict, args):
    """Recomputes set_f1 and the metrics files from an existing
    predictions.csv, without generating again (no GPU needed)."""
    import pandas as pd

    out = Path(args.out)
    rows = pd.read_csv(out / "predictions.csv")
    add_set_f1(rows)
    rows.to_csv(out / "predictions.csv", index=False)
    for qt, sub in rows.groupby("question_type"):
        sub.to_csv(out / "predictions_by_type" / f"{qt}.csv", index=False)
    with open(out / "metrics.json") as f:
        old = json.load(f)
    info = {k: old[k] for k in ("model_id", "adapter", "run_name", "trained_stages") if k in old}
    write_metrics(out, rows, info)
    log(f"rescored {out}")


# --- main ------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn in [("data", cmd_data), ("train", cmd_train), ("eval", cmd_eval), ("rescore", cmd_rescore)]:
        p = sub.add_parser(name)
        p.add_argument("--config", required=True, help="YAML config, see configs/sft.yaml")
        p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                       help="override a config key, e.g. stages.grounding.enabled=false")
        p.set_defaults(fn=fn)
        if name == "eval":
            p.add_argument("--baseline", action="store_true",
                           help="score the base model with no adapter (writes to <output_root>/baseline/eval)")
            p.add_argument("--adapter", help="adapter folder (default: <run dir>/adapter)")
            p.add_argument("--out", help="output folder (default: <run dir>/eval/finetuned)")
        if name == "rescore":
            p.add_argument("--out", required=True, help="eval folder holding predictions.csv and metrics.json")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    # Must happen before the first CUDA call so device_map="auto" only sees
    # the chosen GPU on a shared box. Under `accelerate launch`/`torchrun`
    # the launcher owns GPU assignment (LOCAL_RANK indexes into whatever
    # CUDA_VISIBLE_DEVICES it inherited), so pinning to one index here would
    # put every rank on the same card -- restrict the set with the launcher's
    # own --gpu_ids, or CUDA_VISIBLE_DEVICES in the surrounding shell.
    _, _, world_size = dist_info()
    if cfg["model"]["gpu_id"] is not None:
        if world_size > 1:
            log(f"ignoring model.gpu_id={cfg['model']['gpu_id']} under a {world_size}-process launch")
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["model"]["gpu_id"])
    # Lets the allocator grow blocks instead of leaving reserved-but-unused
    # gaps; avoids OOM from fragmentation on a GPU shared with other jobs.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    args.fn(cfg, args)


if __name__ == "__main__":
    main()
