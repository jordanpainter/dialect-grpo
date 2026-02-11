import argparse
import json
import logging
import os
import random

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer

from rewards import COMETReward, DialectRewardStub
from src.formatting import build_chat_prompt


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("grpo")


def maybe_setup_wandb(cfg, logger) -> bool:
    wb = cfg.get("wandb", {})
    if not wb.get("project"):
        return False
    os.environ["WANDB_PROJECT"] = wb["project"]
    if wb.get("api_key"):
        import wandb
        wandb.login(key=wb["api_key"])
    return True


def load_policy_and_tokenizer(cfg):
    mcfg = cfg["model"]
    model_id = mcfg["model_id"]
    tok_id = mcfg.get("tokenizer_id", model_id)

    tokenizer = AutoTokenizer.from_pretrained(tok_id, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_cfg = None
    if mcfg.get("load_in_4bit", False):
        compute_dtype = torch.bfloat16 if mcfg.get("bnb_4bit_compute_dtype", "bfloat16") == "bfloat16" else torch.float16
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=mcfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=bool(mcfg.get("bnb_4bit_use_double_quant", True)),
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        quantization_config=quant_cfg,
        return_dict=True,
    )
    return model, tokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", "-c", required=True)
    p.add_argument("--resume", "-r", action="store_true")
    args = p.parse_args()

    logger = setup_logging()

    with open(args.config) as f:
        cfg = json.load(f)

    # seeds
    seed = int(cfg["data"].get("seed", 42))
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    use_wandb = maybe_setup_wandb(cfg, logger)
    if not use_wandb:
        cfg["trainer"]["report_to"] = "none"

    model, tokenizer = load_policy_and_tokenizer(cfg)

    # Load dataset from HF hub (your case)
    dcfg = cfg["data"]
    ds = load_dataset(dcfg["dataset_id"], split=dcfg.get("dataset_split", "train"))

    # sanity columns
    for col in ["prompt", "chosen", "rejected"]:
        if col not in ds.column_names:
            raise ValueError(f"Dataset missing required column: {col}. Found: {ds.column_names}")

    # split
    ds = ds.train_test_split(test_size=float(dcfg.get("test_size", 0.02)), seed=seed)
    train_ds, eval_ds = ds["train"], ds["test"]

    # smoke subset (optional)
    if "smoke_subset_train" in dcfg and dcfg["smoke_subset_train"]:
        n = min(int(dcfg["smoke_subset_train"]), len(train_ds))
        train_ds = train_ds.select(range(n))
        logger.info("Using smoke subset train: %d rows", n)
    if "smoke_subset_eval" in dcfg and dcfg["smoke_subset_eval"]:
        n = min(int(dcfg["smoke_subset_eval"]), len(eval_ds))
        eval_ds = eval_ds.select(range(n))
        logger.info("Using smoke subset eval: %d rows", n)

    system_prompt = dcfg.get("system_prompt", "")

    # map: keep raw prompt for COMET, wrap prompt for policy generation
    def map_fn(ex):
        raw = ex["prompt"]
        ex["prompt_raw"] = raw
        ex["prompt"] = build_chat_prompt(tokenizer, system_prompt, raw)
        return ex

    train_ds = train_ds.map(map_fn)
    eval_ds = eval_ds.map(map_fn)

    # LoRA
    pcfg = cfg["peft"]
    lora_cfg = LoraConfig(
        r=int(pcfg.get("r", 8)),
        lora_alpha=int(pcfg.get("lora_alpha", 16)),
        lora_dropout=float(pcfg.get("lora_dropout", 0.05)),
        target_modules=list(pcfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])),
        bias=str(pcfg.get("bias", "none")),
        task_type=str(pcfg.get("task_type", "CAUSAL_LM")),
    )

    # Rewards (composable)
    rcfg = cfg["rewards"]
    reward_funcs = [
        COMETReward(
            checkpoint=rcfg.get("comet_checkpoint", "Unbabel/wmt22-comet-da"),
            device=rcfg.get("comet_device", "cpu"),
            batch_size=int(rcfg.get("comet_batch_size", 4)),
            use_prompt_raw=True,
        ),
        DialectRewardStub(),
    ]
    reward_weights = [float(rcfg.get("w_comet", 1.0)), float(rcfg.get("w_dialect", 0.0))]

    # TRL GRPO: processing_class is the tokenizer in newer TRL. :contentReference[oaicite:3]{index=3}
    grpo_args = GRPOConfig(**cfg["trainer"])
    trainer = GRPOTrainer(
        model=model,
        args=grpo_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=lora_cfg,
        reward_funcs=reward_funcs,
        reward_weights=reward_weights,
    )

    ckpt = None
    if args.resume:
        ckpt = get_last_checkpoint(trainer.args.output_dir)
        if ckpt:
            logger.info("Resuming from checkpoint: %s", ckpt)

    trainer.train(resume_from_checkpoint=ckpt)

    out_dir = trainer.args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    logger.info("Saved to %s", out_dir)


if __name__ == "__main__":
    main()
