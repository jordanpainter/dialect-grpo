import argparse
import json
import logging
import os
import random

import numpy as np
import torch
from datasets import load_from_disk
from huggingface_hub import snapshot_download, login as hf_login
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from trl import GRPOConfig, GRPOTrainer

from rewards import DialectRewardStub
from src.formatting import build_chat_prompt


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("grpo")


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


def load_dataset_from_hub_snapshot(dataset_id: str, split: str, logger: logging.Logger):
    """
    Dataset is stored on HF Hub as a `save_to_disk()` snapshot.
    We snapshot_download the repo and then load_from_disk.
    """
    local_path = snapshot_download(dataset_id, repo_type="dataset")
    logger.info("Downloaded dataset snapshot to: %s", local_path)

    ds_any = load_from_disk(local_path)

    # ds_any can be Dataset or DatasetDict
    if hasattr(ds_any, "column_names"):
        ds = ds_any
        logger.info("Loaded dataset (Dataset) with columns: %s", ds.column_names)
        return ds

    # DatasetDict
    if split in ds_any:
        ds = ds_any[split]
    else:
        first_split = list(ds_any.keys())[0]
        logger.warning("Requested split '%s' not found. Using first split '%s'.", split, first_split)
        ds = ds_any[first_split]

    logger.info("Loaded dataset split '%s' with columns: %s", split, ds.column_names)
    return ds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", "-c", required=True)
    args = p.parse_args()

    logger = setup_logging()

    with open(args.config) as f:
        cfg = json.load(f)

    # Optional HF token login (helps with rate limits)
    hf_token = cfg.get("hf_token") or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if hf_token:
        hf_login(token=hf_token)
        logger.info("Logged into Hugging Face Hub via token")

    # Seeds
    seed = int(cfg["data"].get("seed", 42))
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model, tokenizer = load_policy_and_tokenizer(cfg)

    dcfg = cfg["data"]
    ds = load_dataset_from_hub_snapshot(
        dataset_id=dcfg["dataset_id"],
        split=dcfg.get("dataset_split", "train"),
        logger=logger,
    )

    # Validate columns
    required = ["prompt", "chosen", "rejected"]
    for col in required:
        if col not in ds.column_names:
            raise ValueError(f"Dataset missing required column: {col}. Found: {ds.column_names}")

    # Split
    ds = ds.train_test_split(test_size=float(dcfg.get("test_size", 0.02)), seed=seed)
    train_ds, eval_ds = ds["train"], ds["test"]
    logger.info("Train/Eval sizes: %d / %d", len(train_ds), len(eval_ds))

    # Smoke subset (optional)
    n_tr = int(dcfg.get("smoke_subset_train", 0) or 0)
    n_ev = int(dcfg.get("smoke_subset_eval", 0) or 0)
    if n_tr > 0:
        train_ds = train_ds.select(range(min(n_tr, len(train_ds))))
        logger.info("Smoke subset train: %d rows", len(train_ds))
    if n_ev > 0:
        eval_ds = eval_ds.select(range(min(n_ev, len(eval_ds))))
        logger.info("Smoke subset eval: %d rows", len(eval_ds))

    system_prompt = dcfg.get("system_prompt", "")

    # Wrap prompt for the policy; keep prompt_raw for future reward models (COMET)
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

    # Rewards (smoke test): stub only
    reward_funcs = [DialectRewardStub()]

    import inspect
    # Filter config keys to match the installed TRL version's GRPOConfig signature
    sig = inspect.signature(GRPOConfig.__init__)
    allowed = set(sig.parameters.keys())
    allowed.discard("self")

    raw_args = cfg["trainer"]
    filtered_args = {k: v for k, v in raw_args.items() if k in allowed}

    dropped = sorted(set(raw_args.keys()) - set(filtered_args.keys()))
    if dropped:
        logger.warning("Dropping unsupported GRPOConfig args for this TRL version: %s", dropped)

    grpo_args = GRPOConfig(**filtered_args)

    trainer = GRPOTrainer(
        model=model,
        args=grpo_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=lora_cfg,
        reward_funcs=reward_funcs,
    )

    trainer.train()

    out_dir = trainer.args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    logger.info("Saved to %s", out_dir)


if __name__ == "__main__":
    main()
