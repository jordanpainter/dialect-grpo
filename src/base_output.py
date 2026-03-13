#!/usr/bin/env python3
"""
Add deterministic base-model generations to a saved-to-disk HF dataset snapshot,
then optionally upload the augmented snapshot back to the Hub as a dataset repo.

Designed for this project, where GRPO currently loads datasets via:
  snapshot_download(..., repo_type="dataset") + load_from_disk(...)

Example:
  python -m src.base_output \
    --dataset_id jordanpainter/dialect-preferences \
    --dataset_split train \
    --model_id Qwen/Qwen3-8B \
    --output_dir data/dialect-preferences-with-base \
    --push_repo_id jordanpainter/dialect-preferences-with-base-qwen

Optional dialect-density caching:
  python -m src.base_output \
    --dataset_id jordanpainter/dialect-preferences \
    --dataset_split train \
    --model_id Qwen/Qwen3-8B \
    --output_dir data/dialect-preferences-with-base \
    --dialect_model_path srirag/feature-identifier \
    --push_repo_id jordanpainter/dialect-preferences-with-base-qwen
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any, Dict, Optional, Tuple

import torch
from datasets import Dataset, DatasetDict, load_from_disk
from huggingface_hub import HfApi, create_repo, login as hf_login, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from rewards.dialect_reward_model import RewardModel
from src.formatting import build_chat_prompt


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("base_output")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def ensure_pad_token(tokenizer) -> None:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def resolve_model_max_length(tokenizer, model, fallback: int = 2048) -> int:
    tmax = getattr(tokenizer, "model_max_length", None)
    if isinstance(tmax, int) and 64 < tmax < 100_000:
        return int(tmax)

    mmax = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    if isinstance(mmax, int) and 64 < mmax < 100_000:
        return int(mmax)

    return int(fallback)


def truncate_prompt_to_max_tokens(tokenizer, text: str, max_tokens: int) -> str:
    enc = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        truncation=True,
        max_length=max_tokens,
    )
    ids = enc["input_ids"]
    return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def build_prompt(
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    prefer_chat_template: bool = True,
) -> str:
    if prefer_chat_template and hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return build_chat_prompt(tokenizer, system_prompt, user_prompt)


def infer_eos_ids(tokenizer) -> list[int]:
    eos_ids = []
    if tokenizer.eos_token_id is not None:
        eos_ids.append(int(tokenizer.eos_token_id))

    candidates = ["<|eot_id|>", "<|end_of_turn|>", "<end_of_turn>"]
    vocab = tokenizer.get_vocab()
    for tok in candidates:
        if tok in vocab:
            eos_ids.append(int(tokenizer.convert_tokens_to_ids(tok)))

    seen = set()
    deduped = []
    for item in eos_ids:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def load_dataset_snapshot(dataset_id: str, split: str, logger: logging.Logger) -> Tuple[Dataset | DatasetDict, Optional[str]]:
    """
    Load a dataset repo that contains a save_to_disk() snapshot.

    Returns:
      (dataset_or_datasetdict, active_split_name)

    If the repo contains a plain Dataset, active_split_name will be None.
    If it contains a DatasetDict, active_split_name is the split we selected.
    """
    local_path = snapshot_download(repo_id=dataset_id, repo_type="dataset")
    logger.info("Downloaded dataset snapshot to %s", local_path)

    ds_any = load_from_disk(local_path)

    if isinstance(ds_any, Dataset):
        logger.info("Loaded Dataset with columns: %s", ds_any.column_names)
        return ds_any, None

    if not isinstance(ds_any, DatasetDict):
        raise TypeError(f"Unsupported dataset object: {type(ds_any)}")

    if split in ds_any:
        logger.info("Loaded DatasetDict split '%s' with columns: %s", split, ds_any[split].column_names)
        return ds_any, split

    first_split = next(iter(ds_any.keys()))
    logger.warning("Requested split '%s' not found. Falling back to '%s'.", split, first_split)
    return ds_any, first_split


# -----------------------------------------------------------------------------
# Core augmentation
# -----------------------------------------------------------------------------

def load_generation_model(model_id: str, tokenizer_id: Optional[str], bf16: bool, logger: logging.Logger):
    tok_id = tokenizer_id or model_id
    tokenizer = AutoTokenizer.from_pretrained(tok_id, use_fast=True)
    logger.info("Loading tokenizer: %s", tok_id)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    ensure_pad_token(tokenizer)

    dtype = torch.bfloat16 if (bf16 and torch.cuda.is_available()) else None
    model_kwargs: Dict[str, Any] = dict(low_cpu_mem_usage=True, device_map="auto")
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype

    logger.info("Loading generation model: %s", model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.eval()
    model.config.use_cache = True

    model.config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        model.config.eos_token_id = tokenizer.eos_token_id

    return model, tokenizer

from torch.utils.data import DataLoader
import time


def add_base_outputs(
    ds: Dataset,
    model,
    tokenizer,
    system_prompt: str,
    max_new_tokens: int,
    prompt_max_len: Optional[int],
    batch_size: int,
    output_col: str,
    use_chat_template: bool,
    logger: logging.Logger,
) -> Dataset:
    eos_ids = infer_eos_ids(tokenizer)
    logger.info("Using eos ids for generation: %s", eos_ids)

    if prompt_max_len is None:
        prompt_max_len = 1024

    logger.info("Prompt max length set to %d tokens", prompt_max_len)

    prompts = ds["prompt"]
    total = len(prompts)
    results = []

    loader = DataLoader(prompts, batch_size=batch_size, shuffle=False)

    start_time = time.time()
    last_log_time = start_time

    logger.info(
        "Starting base generation for %d prompts with batch_size=%d, max_new_tokens=%d",
        total,
        batch_size,
        max_new_tokens,
    )

    for step, batch in enumerate(loader, start=1):
        if use_chat_template:
            built_prompts = [
                truncate_prompt_to_max_tokens(
                    tokenizer,
                    build_prompt(tokenizer, system_prompt, p, prefer_chat_template=True),
                    prompt_max_len,
                )
                for p in batch
            ]
        else:
            built_prompts = [
                truncate_prompt_to_max_tokens(
                    tokenizer,
                    p,
                    prompt_max_len,
                )
                for p in batch
            ]

        inputs = tokenizer(
            built_prompts,
            padding=True,
            truncation=True,
            max_length=prompt_max_len,
            return_tensors="pt",
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        if step == 1:
            logger.info("Running first generation batch...")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                return_dict_in_generate=False,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_ids,
        )

        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = outputs[:, prompt_len:]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        decoded = [text.strip() for text in decoded]

        results.extend(decoded)

        processed = len(results)

        if step == 1 or step % 10 == 0 or processed == total:
            now = time.time()
            elapsed = now - start_time
            since_last = now - last_log_time
            ex_per_sec = processed / elapsed if elapsed > 0 else 0.0
            pct = 100.0 * processed / total

            logger.info(
                "Progress: %d/%d examples (%.2f%%) | step=%d | elapsed=%.1fs | rate=%.2f ex/s | last_batch_time=%.1fs",
                processed,
                total,
                pct,
                step,
                elapsed,
                ex_per_sec,
                since_last,
            )

            if decoded:
                preview = decoded[0].replace("\n", " ")[:160]
                logger.info("Sample output preview: %s", preview)

            last_log_time = now

    logger.info("Generated %d base outputs total", len(results))
    return ds.add_column(output_col, results)


def add_base_dialect_density(
    ds: Dataset,
    dialect_model_path: str,
    batch_size: int,
    input_col: str,
    output_col: str,
    logger: logging.Logger,
) -> Dataset:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading dialect reward model from %s on %s", dialect_model_path, device)
    reward_model = RewardModel(model_path=dialect_model_path, device=device)

    def map_fn(batch: Dict[str, list[str]]) -> Dict[str, list[float]]:
        rewards = reward_model.reward(batch[input_col])
        if isinstance(rewards, torch.Tensor):
            rewards = rewards.detach().cpu().tolist()
        return {output_col: [float(x) for x in rewards]}

    logger.info("Scoring '%s' into column '%s' ...", input_col, output_col)
    return ds.map(map_fn, batched=True, batch_size=batch_size, desc=f"Scoring {output_col}")


# -----------------------------------------------------------------------------
# Hub upload
# -----------------------------------------------------------------------------

def upload_saved_snapshot(folder_path: str, repo_id: str, private: bool, logger: logging.Logger) -> None:
    logger.info("Uploading saved snapshot from %s to dataset repo %s", folder_path, repo_id)
    create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    api = HfApi()
    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add cached base outputs for GRPO",
    )
    logger.info("Upload complete: %s", repo_id)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", default="jordanpainter/dialect-preferences")
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--model_id", default="Qwen/Qwen3-8B")
    parser.add_argument("--tokenizer_id", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_col", default="base_output_qwen")
    parser.add_argument("--system_prompt", default="You are a helpful assistant.")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--prompt_max_len", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--dialect_model_path", default=None)
    parser.add_argument("--dialect_output_col", default="base_dialect_density")
    parser.add_argument("--push_repo_id", default=None)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--use_chat_template", action="store_true")
    args = parser.parse_args()

    logger = setup_logging()

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if hf_token:
        hf_login(token=hf_token)
        logger.info("Logged into Hugging Face Hub via token")

    ds_any, active_split = load_dataset_snapshot(args.dataset_id, args.dataset_split, logger)

    model, tokenizer = load_generation_model(
        model_id=args.model_id,
        tokenizer_id=args.tokenizer_id,
        bf16=args.bf16,
        logger=logger,
    )

    if isinstance(ds_any, Dataset):
        work_ds = ds_any
    else:
        work_ds = ds_any[active_split]

    required_cols = ["prompt", "chosen", "rejected"]
    missing = [c for c in required_cols if c not in work_ds.column_names]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}. Found: {work_ds.column_names}")

    work_ds = add_base_outputs(
        ds=work_ds,
        model=model,
        tokenizer=tokenizer,
        system_prompt=args.system_prompt,
        max_new_tokens=args.max_new_tokens,
        prompt_max_len=args.prompt_max_len,
        batch_size=args.batch_size,
        output_col=args.output_col,
        use_chat_template=args.use_chat_template,
        logger=logger,
    )

    if args.dialect_model_path:
        work_ds = add_base_dialect_density(
            ds=work_ds,
            dialect_model_path=args.dialect_model_path,
            batch_size=max(1, args.batch_size * 4),
            input_col=args.output_col,
            output_col=args.dialect_output_col,
            logger=logger,
        )

    if isinstance(ds_any, DatasetDict):
        ds_any[active_split] = work_ds
        save_obj = ds_any
    else:
        save_obj = work_ds

    os.makedirs(os.path.dirname(args.output_dir) or ".", exist_ok=True)
    logger.info("Saving augmented dataset snapshot to %s", args.output_dir)
    save_obj.save_to_disk(args.output_dir)

    if args.push_repo_id:
        upload_saved_snapshot(args.output_dir, args.push_repo_id, args.private, logger)

    logger.info("Done. Final columns: %s", work_ds.column_names)


if __name__ == "__main__":
    main()
