#!/usr/bin/env python
"""GRPO (RL with verifiable rewards) on the Marker_CXR VQA data. Same config
style, data pipeline and output layout as `01_finetune_markercxr.py` (SFT).

    python 02_grpo_markercxr.py data  --config configs/grpo.yaml
    python 02_grpo_markercxr.py train --config configs/grpo.yaml

The model samples several answers per question and is pushed toward the ones
with higher reward (IoU + center closeness + parse bonus for boxes) -- no
target text is imitated, only the final answer is scored. Training starts
from `model.start_from_adapter` (an SFT LoRA, merged into the base model
once, saved under <run dir>/merged_base) or from the raw base if null.

Evaluate with the SFT script so metrics land in the same format:

    python 01_finetune_markercxr.py eval --config configs/sft.yaml \\
        --set model.model_id=<run dir>/merged_base \\
        --set data.use_reasoning_for_boxes=false \\
        --set data.use_reasoning_for_coord_reading=false \\
        --adapter <run dir>/adapter --out <run dir>/eval/finetuned

(the two --set flags make eval prompts match this run's direct prompts;
drop them if data.use_reasoning_for_boxes is true here.)
"""

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE))

ft = importlib.import_module("01_finetune_markercxr")  # leading digit -- not `import`-able
build_table, dist_info, load_config, log = ft.build_table, ft.dist_info, ft.load_config, ft.log
require_hf_token, resolve_path, run_dir = ft.require_hf_token, ft.resolve_path, ft.run_dir
stage_folders = ft.stage_folders

from grpo_rewards import make_reward  # noqa: E402
from markercxr_data import count_by_type_and_split  # noqa: E402


# --- data -------------------------------------------------------------------


def grpo_records(cfg: dict) -> list:
    """Train-split rows of the enabled stages, restricted to
    data.question_types (null = all)."""
    records = build_table(cfg, stage_folders(cfg, "enabled"))
    keep = cfg["data"]["question_types"]
    if keep:
        found = {r["question_type"] for r in records}
        missing = set(keep) - found
        if missing:
            raise SystemExit(f"data.question_types not in the enabled stages: {sorted(missing)}")
        records = [r for r in records if r["question_type"] in set(keep)]
    return [r for r in records if r["split"] == "train"]


def to_grpo_dataset(cfg: dict, records: list):
    """trl's GRPOTrainer wants a `prompt` column (chat messages) plus an
    `image` column; every other column (answer, question_type) is passed
    through to the reward function as kwargs."""
    from datasets import Dataset, Image

    rows = [
        {
            "prompt": [{"role": "user", "content": r["question"]}],
            "image": r["image_path"],  # decoded lazily by the Image feature
            "answer": r["answer"],
            "question_type": r["question_type"],
        }
        for r in records
    ]
    ds = Dataset.from_list(rows).cast_column("image", Image())
    d = cfg["data"]
    if d["max_train_examples"] is not None and len(ds) > d["max_train_examples"]:
        ds = ds.shuffle(seed=d["seed"]).select(range(d["max_train_examples"]))
    return ds


def cmd_data(cfg: dict, args):
    records = grpo_records(cfg)
    log(count_by_type_and_split(records)[["train"]].to_string())
    log(f"\n{len(records)} GRPO prompts (train split)")


# --- model ------------------------------------------------------------------


def prepare_start_model(cfg: dict, out: Path) -> str:
    """Path of the model GRPO starts from. With start_from_adapter set, the
    SFT LoRA (incl. its fully tuned lm_head/embed_tokens) is merged into the
    base in bf16 and saved once under <run dir>/merged_base -- LoRA can't be
    merged into 4-bit weights, and the ref policy / eval both need the merged
    model as a plain loadable checkpoint. ~9GB, built on CPU by rank 0."""
    m = cfg["model"]
    if not m["start_from_adapter"]:
        return m["model_id"]
    merged = out / "merged_base"
    done = merged / ".done"
    _, rank, world_size = dist_info()
    if rank == 0 and not done.exists():
        import torch
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor

        adapter = resolve_path(m["start_from_adapter"])
        if not (adapter / "adapter_config.json").exists():
            raise SystemExit(f"no adapter at {adapter}")
        log(f"merging {adapter} into {m['model_id']} -> {merged}")
        base = AutoModelForImageTextToText.from_pretrained(
            m["model_id"], dtype=torch.bfloat16, device_map="cpu"
        )
        model = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
        model.save_pretrained(merged)
        AutoProcessor.from_pretrained(m["model_id"]).save_pretrained(merged)
        del model, base
        done.touch()
    while not done.exists():  # other ranks wait for rank 0 (shared filesystem)
        time.sleep(5)
    return str(merged)


def cmd_train(cfg: dict, args):
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from trl import GRPOConfig, GRPOTrainer

    require_hf_token()
    local_rank, rank, world_size = dist_info()
    if torch.cuda.get_device_capability(local_rank)[0] < 8:
        raise SystemExit("GPU does not support bfloat16; use an Ampere or newer GPU.")

    out = run_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        with open(out / "config.yaml", "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

    # Same reasoning as SFT: every rank builds the identical table.
    records = grpo_records(cfg)
    summary = count_by_type_and_split(records)
    if rank == 0:
        summary.to_csv(out / "data_summary.csv")
    log(summary[["train"]].to_string())
    dataset = to_grpo_dataset(cfg, records)
    log(dataset)

    start = prepare_start_model(cfg, out)
    log(f"starting policy: {start}")
    # bf16, not 4-bit: the start model is already a merged full checkpoint
    # (see prepare_start_model) and a 4B model in bf16 fits the L40S easily.
    device_map = {"": local_rank} if world_size > 1 else "auto"
    model = AutoModelForImageTextToText.from_pretrained(
        start, dtype=torch.bfloat16, device_map=device_map, attn_implementation="eager"
    )
    processor = AutoProcessor.from_pretrained(start)
    # The SFT model ends every answer with the chat template's <end_of_turn>,
    # not <eos> -- but trl stops generation at the tokenizer's eos_token only.
    # Without this, no rollout ever terminates: every completion runs to
    # max_completion_length (clipped_ratio 1.0) with garbage after the answer.
    processor.tokenizer.eos_token = "<end_of_turn>"

    t = cfg["train"]
    peft_config = LoraConfig(
        lora_alpha=t["lora"]["alpha"],
        lora_dropout=t["lora"]["dropout"],
        r=t["lora"]["r"],
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    # In GRPO, per_device_train_batch_size counts COMPLETIONS. One reward
    # group = num_generations completions of the same prompt, so:
    completions_per_step = (
        t["per_device_train_batch_size"] * t["gradient_accumulation_steps"] * world_size
    )
    if completions_per_step % t["num_generations"] != 0:
        raise SystemExit(
            f"per_device_batch x grad_accum x ranks = {completions_per_step} "
            f"must be divisible by num_generations = {t['num_generations']}"
        )
    log(
        f"world_size={world_size}  {completions_per_step} completions/step = "
        f"{completions_per_step // t['num_generations']} prompts x "
        f"{t['num_generations']} generations"
    )

    grpo_args = GRPOConfig(
        output_dir=str(out / "adapter"),
        max_steps=t["max_steps"],
        learning_rate=float(t["learning_rate"]),
        lr_scheduler_type="constant",  # RL: no decay schedule to a fixed step count
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        num_generations=t["num_generations"],
        generation_batch_size=t["generation_batch_size"],
        max_completion_length=t["max_completion_length"],
        temperature=t["temperature"],
        beta=t["beta"],
        logging_steps=t["logging_steps"],
        save_strategy="steps",
        save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"],
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        log_completions=t["log_completions"],
        push_to_hub=False,  # MIMIC-CXR-derived weights stay local
        report_to="tensorboard",
        remove_unused_columns=False,  # answer/question_type go to the reward fn
        ddp_find_unused_parameters=False,
    )

    r = t["reward"]
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[
            make_reward(
                parse_bonus=r["parse_bonus"],
                iou_weight=r["iou_weight"],
                center_weight=r["center_weight"],
                giou_weight=r.get("giou_weight", 0.0),
                # Window terms for the reasoning arm; absent in older configs.
                window_parse_bonus=r.get("window_parse_bonus", 0.0),
                window_weight=r.get("window_weight", 0.0),
                window_giou_weight=r.get("window_giou_weight", 0.0),
                box_in_window_weight=r.get("box_in_window_weight", 0.0),
            )
        ],
        args=grpo_args,
        train_dataset=dataset,
        processing_class=processor,
        peft_config=peft_config,
    )
    from transformers.trainer_utils import get_last_checkpoint

    last_ckpt = get_last_checkpoint(grpo_args.output_dir) if Path(grpo_args.output_dir).is_dir() else None
    if last_ckpt:
        log(f"resuming from {last_ckpt}")
    trainer.train(resume_from_checkpoint=last_ckpt)
    trainer.save_model()
    if rank == 0:
        with open(out / "train_log.json", "w") as f:
            json.dump(trainer.state.log_history, f, indent=1)
    log(f"adapter saved to {out / 'adapter'}")
    log(
        "evaluate with:\n"
        f"  python 01_finetune_markercxr.py eval --config configs/sft.yaml \\\n"
        f"    --set model.model_id={start} \\\n"
        f"    --set data.use_reasoning_for_boxes={str(cfg['data']['use_reasoning_for_boxes']).lower()} \\\n"
        f"    --set data.use_reasoning_for_coord_reading={str(cfg['data']['use_reasoning_for_coord_reading']).lower()} \\\n"
        f"    --adapter {out / 'adapter'} --out {out / 'eval' / 'finetuned'}"
    )


# --- main -------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn in [("data", cmd_data), ("train", cmd_train)]:
        p = sub.add_parser(name)
        p.add_argument("--config", required=True, help="YAML config, see configs/grpo.yaml")
        p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                       help="override a config key, e.g. train.max_steps=50")
        p.set_defaults(fn=fn)
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    _, _, world_size = dist_info()
    if cfg["model"]["gpu_id"] is not None:
        if world_size > 1:
            log(f"ignoring model.gpu_id={cfg['model']['gpu_id']} under a {world_size}-process launch")
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["model"]["gpu_id"])
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    args.fn(cfg, args)


if __name__ == "__main__":
    main()
