import argparse
import json
import logging
import os
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets import load_from_disk
from huggingface_hub import snapshot_download, login as hf_login
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from trl import GRPOConfig, GRPOTrainer

# keep your reward
from rewards import cometkiwi_reward, embedding_margin_reward, dialect_reward_stub


# keep your formatter as fallback
from src.formatting import build_chat_prompt


# -----------------------------
# Logging / utilities
# -----------------------------
def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("grpo")


def get_local_rank() -> int:
    # accelerate sets LOCAL_RANK
    return int(os.environ.get("LOCAL_RANK", "0"))


def hard_trim_completion(text: str, stop_strings: List[str]) -> str:
    """Trim at the earliest stop string occurrence."""
    if not text:
        return text
    cut = None
    for s in stop_strings:
        if not s:
            continue
        idx = text.find(s)
        if idx != -1:
            cut = idx if cut is None else min(cut, idx)
    return text[:cut].rstrip() if cut is not None else text


def ensure_pad_token(tokenizer):
    if tokenizer.pad_token is None:
        # common safe default for causal LMs
        tokenizer.pad_token = tokenizer.eos_token


# -----------------------------
# Model + tokenizer
# -----------------------------
def load_policy_and_tokenizer(cfg: Dict[str, Any], logger: logging.Logger):
    mcfg = cfg["model"]
    model_id = mcfg["model_id"]
    tok_id = mcfg.get("tokenizer_id", model_id)

    tokenizer = AutoTokenizer.from_pretrained(tok_id, use_fast=True)
    tokenizer.padding_side = "left"
    ensure_pad_token(tokenizer)

    quant_cfg = None
    if mcfg.get("load_in_4bit", False):
        compute_dtype = (
            torch.bfloat16
            if mcfg.get("bnb_4bit_compute_dtype", "bfloat16") == "bfloat16"
            else torch.float16
        )
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=mcfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=bool(mcfg.get("bnb_4bit_use_double_quant", True)),
        )

    # IMPORTANT for accelerate/DDP:
    # - don't use device_map="auto"
    # - pin to local rank if quantized
    local_rank = get_local_rank()
    device_map = {"": local_rank} if quant_cfg is not None else None
    logger.info("LOCAL_RANK=%s | device_map=%s", local_rank, device_map)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if cfg["trainer"].get("bf16", False) else None,
        low_cpu_mem_usage=True,
        quantization_config=quant_cfg,
        device_map=device_map,
        return_dict=True,
    )

    # Some tiny chat models behave better with these set
    model.config.use_cache = not bool(cfg["trainer"].get("gradient_checkpointing", False))

    return model, tokenizer


# -----------------------------
# Dataset loading
# -----------------------------
def load_dataset_from_hub_snapshot(dataset_id: str, split: str, logger: logging.Logger):
    """
    Dataset is stored on HF Hub as a `save_to_disk()` snapshot.
    We snapshot_download the repo and then load_from_disk.
    """
    local_path = snapshot_download(dataset_id, repo_type="dataset")
    logger.info("Downloaded dataset snapshot to: %s", local_path)

    ds_any = load_from_disk(local_path)

    # Dataset
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


def build_prompt(
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    prefer_chat_template: bool = True,
) -> str:
    """
    Prefer tokenizer.apply_chat_template(add_generation_prompt=True) when available,
    otherwise fall back to your build_chat_prompt().
    """
    if prefer_chat_template and hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    # fallback
    return build_chat_prompt(tokenizer, system_prompt, user_prompt)


# -----------------------------
# Main
# -----------------------------
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

    # model/tokenizer
    model, tokenizer = load_policy_and_tokenizer(cfg, logger)

    # data
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

    system_prompt = dcfg.get("system_prompt", "") or ""

    # Stop strings to prevent "starting a conversation" in the completion
    # (we'll trim for reward evaluation and for quick sanity sampling)
    stop_strings = [
        "\nUser:",
        "\nAssistant:",
        "\n### User:",
        "\n### Assistant:",
        "\n<|user|>",
        "\n<|assistant|>",
    ]

    # Map: build the chat-ready prompt text
    def map_fn(ex):
        raw = ex["prompt"]
        ex["prompt_raw"] = raw
        ex["prompt"] = build_prompt(tokenizer, system_prompt, raw, prefer_chat_template=True)
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

    # Reward function:
    # keep your existing one, but (important) ensure we don't score the "continued chat" tail.
    # We wrap it lightly so signature stays function-like and __name__ exists.
    def trim_wrapper(reward_fn):
        def _wrapped(prompts, completions, **kwargs):
            trimmed = [hard_trim_completion(c, stop_strings) for c in completions]
            return reward_fn(prompts, trimmed, **kwargs)
        _wrapped.__name__ = getattr(reward_fn, "__name__", "reward_fn")
        return _wrapped

    reward_funcs = [trim_wrapper(embedding_margin_reward)]


    # Build GRPOConfig safely across TRL versions
    import inspect

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

    # Save: for LoRA this saves the adapter (good, lightweight).
    out_dir = trainer.args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    logger.info("Saved adapter+tokenizer to %s", out_dir)

    # Quick sanity generation (rank 0 only)
    if get_local_rank() == 0:
        try:
            logger.info("Running quick sanity generation...")
            test_prompt = "Write a short friendly reply in British English about making a cup of tea."
            chat = build_prompt(tokenizer, system_prompt, test_prompt, prefer_chat_template=True)
            inputs = tokenizer(chat, return_tensors="pt").to(trainer.model.device)

            with torch.no_grad():
                gen = trainer.model.generate(
                    **inputs,
                    max_new_tokens=min(int(cfg["trainer"].get("max_completion_length", 64)), 128),
                    do_sample=True,
                    temperature=float(cfg["trainer"].get("temperature", 0.9)),
                    top_p=float(cfg["trainer"].get("top_p", 0.95)),
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            decoded = tokenizer.decode(gen[0], skip_special_tokens=True)
            # keep only completion portion if possible
            # (for most chat templates, prompt is a prefix)
            completion = decoded[len(chat):] if decoded.startswith(chat) else decoded
            completion = hard_trim_completion(completion, stop_strings)
            logger.info("SANITY COMPLETION:\n%s", completion.strip())
        except Exception as e:
            logger.warning("Sanity generation failed (non-fatal): %s", e)


if __name__ == "__main__":
    main()
