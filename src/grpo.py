"""
src/grpo.py

GRPO (TRL GRPOTrainer) training script for dialectal English preference learning.

Key features:
- Loads a HF `save_to_disk()` dataset snapshot via snapshot_download + load_from_disk.
- Converts raw prompts into chat-format prompts using tokenizer.apply_chat_template when available.
- Trains a small chat policy with LoRA (optionally with 4-bit quantization).
- Uses a *single combined reward* made of:
    (1) dialect reward on the generated completion
    (2) COMET similarity (chosen vs generated)
    (3) embedding cosine similarity (chosen vs generated)
  then normalizes each component (batch z-score or running z-score) and combines with weights.
- Adds practical “stop token id list” support so generations can terminate properly.
- Trims completions by stop-strings *for reward scoring* to avoid scoring “continued chat” tails.

Run:
  accelerate launch --num_processes=1 -m src.grpo -c configs/gemma.json
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from datasets import load_from_disk
from huggingface_hub import login as hf_login
from huggingface_hub import snapshot_download
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from trl import GRPOConfig, GRPOTrainer

# --- Reward components (your project modules) ---
from rewards.comet_reward import comet_reward_with_ref
from rewards.dialect_reward import dialect_reward
from rewards.sim_reward import embedding_similarity_reward

# --- Prompt formatting fallback (your project module) ---
from src.formatting import build_chat_prompt


# =============================================================================
# Logging and small utilities
# =============================================================================

def setup_logging() -> logging.Logger:
    """Configure a simple console logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("grpo")


def get_local_rank() -> int:
    """Accelerate sets LOCAL_RANK for each process."""
    return int(os.environ.get("LOCAL_RANK", "0"))


def ensure_pad_token(tokenizer) -> None:
    """Ensure the tokenizer has a pad token. Safe default: use EOS as pad."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def hard_trim_completion(text: str, stop_strings: Sequence[str]) -> str:
    """
    Trim the completion at the earliest occurrence of any stop string.
    This prevents scoring “continued chat” where the model starts generating User/Assistant tags.
    """
    if not text:
        return text

    cut: Optional[int] = None
    for s in stop_strings:
        if not s:
            continue
        idx = text.find(s)
        if idx != -1:
            cut = idx if cut is None else min(cut, idx)

    return text[:cut].rstrip() if cut is not None else text


class RunningZScore:
    """
    Exponential moving average (EMA) running mean/variance for z-score normalization.

    Useful once training stabilizes: reduces per-batch noise vs batch z-score.
    """
    def __init__(self, beta: float = 0.99, eps: float = 1e-6):
        self.beta = float(beta)
        self.eps = float(eps)
        self.mu: Optional[float] = None
        self.var: Optional[float] = None
        self.steps: int = 0

    def update(self, x: np.ndarray) -> None:
        mu = float(x.mean())
        var = float(x.var())
        if self.mu is None:
            self.mu, self.var = mu, var
        else:
            b = self.beta
            self.mu = b * self.mu + (1 - b) * mu
            self.var = b * self.var + (1 - b) * var
        self.steps += 1

    def normalize(self, x: np.ndarray) -> np.ndarray:
        assert self.mu is not None and self.var is not None
        denom = np.sqrt(self.var + self.eps)
        return (x - self.mu) / denom


# =============================================================================
# Model + tokenizer loading
# =============================================================================

def _build_quant_config(mcfg: Dict[str, Any]) -> Optional[BitsAndBytesConfig]:
    """Create a 4-bit quant config if requested."""
    if not mcfg.get("load_in_4bit", False):
        return None

    compute_dtype = torch.bfloat16 if mcfg.get("bnb_4bit_compute_dtype", "bfloat16") == "bfloat16" else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=mcfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=bool(mcfg.get("bnb_4bit_use_double_quant", True)),
    )


def load_policy_and_tokenizer(cfg: Dict[str, Any], logger: logging.Logger) -> Tuple[torch.nn.Module, Any]:
    """
    Load the policy model + tokenizer.

    Important notes for accelerate/DDP:
    - Avoid device_map="auto".
    - If quantized (4bit), pin the model to local_rank device using device_map={"": local_rank}.
    - Do NOT pass torch_dtype=None; only pass torch_dtype when it’s set.
    """
    mcfg = cfg["model"]
    model_id = mcfg["model_id"]
    tok_id = mcfg.get("tokenizer_id", model_id)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tok_id, use_fast=True)
    tokenizer.padding_side = "left"         # generation-friendly
    tokenizer.truncation_side = "left"      # keep most recent context if truncating
    ensure_pad_token(tokenizer)

    # Quant config (optional)
    quant_cfg = _build_quant_config(mcfg)

    # Device map (important for quantized models + multi-GPU)
    local_rank = get_local_rank()
    device_map = {"": local_rank}
    logger.info("LOCAL_RANK=%s | device_map=%s", local_rank, device_map)

    # dtype (only include if not None)
    dtype = torch.bfloat16 if cfg.get("trainer", {}).get("bf16", False) else None

    from_pretrained_kwargs: Dict[str, Any] = dict(
        low_cpu_mem_usage=True,
        quantization_config=quant_cfg,
        device_map=device_map,
        return_dict=True,
    )
    if dtype is not None:
        from_pretrained_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if cfg["trainer"].get("bf16", False) else None,
        low_cpu_mem_usage=True,
        quantization_config=quant_cfg,   # can be None
        device_map=device_map,
        return_dict=True,
    )

    # Make sure generation sees correct special tokens
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # If you enable gradient checkpointing, you generally want use_cache=False
    model.config.use_cache = not bool(cfg.get("trainer", {}).get("gradient_checkpointing", False))

    return model, tokenizer


# =============================================================================
# Dataset loading and prompt formatting
# =============================================================================

def load_dataset_from_hub_snapshot(dataset_id: str, split: str, logger: logging.Logger):
    """
    Dataset is saved on HF Hub via `save_to_disk()`.
    We:
      1) snapshot_download(..., repo_type="dataset")
      2) load_from_disk(local_path)

    Handles both Dataset and DatasetDict.
    """
    local_path = snapshot_download(dataset_id, repo_type="dataset")
    logger.info("Downloaded dataset snapshot to: %s", local_path)

    ds_any = load_from_disk(local_path)

    # Case 1: already a Dataset
    if hasattr(ds_any, "column_names"):
        logger.info("Loaded dataset (Dataset) with columns: %s", ds_any.column_names)
        return ds_any

    # Case 2: DatasetDict, pick split
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
    Convert (system, user) into a model-ready prompt string.

    Preference order:
    1) tokenizer.apply_chat_template(..., add_generation_prompt=True) if available
    2) project fallback build_chat_prompt()
    """
    if prefer_chat_template and hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return build_chat_prompt(tokenizer, system_prompt, user_prompt)


def infer_stop_token_ids(tokenizer) -> List[int]:
    """
    Build a list of token ids that should end generation.
    Many chat templates end on an "end of turn" token rather than EOS alone.
    """
    eos_ids = []
    if tokenizer.eos_token_id is not None:
        eos_ids.append(int(tokenizer.eos_token_id))

    # Common special tokens across popular chat templates
    candidates = ["<|eot_id|>", "<|end_of_turn|>", "<end_of_turn>"]
    vocab = tokenizer.get_vocab()
    for tok in candidates:
        if tok in vocab:
            eos_ids.append(int(tokenizer.convert_tokens_to_ids(tok)))

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for i in eos_ids:
        if i not in seen:
            deduped.append(i)
            seen.add(i)
    return deduped


# =============================================================================
# Reward construction (single combined reward)
# =============================================================================

def make_trim_wrapper(stop_strings: Sequence[str]):
    """Return a wrapper that trims completions before passing them to a reward function."""
    def trim_wrapper(reward_fn):
        def _wrapped(prompts, completions, **kwargs):
            trimmed = [hard_trim_completion(c, stop_strings) for c in completions]
            return reward_fn(prompts, trimmed, **kwargs)
        _wrapped.__name__ = getattr(reward_fn, "__name__", "reward_fn")
        return _wrapped
    return trim_wrapper


def make_combined_reward(cfg: Dict[str, Any], logger: logging.Logger):
    """
    Create a single reward function for TRL that:
      - computes raw dialect/comet/cosine rewards
      - normalizes each component
      - combines with configurable weights

    Returns:
      callable(prompts, completions, **kw) -> List[float]
    """
    rcfg = cfg.get("rewards", {})

    # Weights
    weights = rcfg.get("weights", {})
    w_dialect = float(weights.get("dialect", 0.50))
    w_comet = float(weights.get("comet", 0.25))
    w_cosine = float(weights.get("cosine", 0.25))

    # Normalization config
    ncfg = rcfg.get("normalization", {})
    method = str(ncfg.get("method", "batch_zscore"))  # "batch_zscore" | "running_zscore" | "none"
    clip_z = float(ncfg.get("clip_z", 5.0))
    eps = float(ncfg.get("eps", 1e-6))
    beta = float(ncfg.get("beta", 0.99))
    warmup_steps = int(ncfg.get("warmup_steps", 0))

    # Running stats (used only for running_zscore)
    rz_dialect = RunningZScore(beta=beta, eps=eps)
    rz_comet = RunningZScore(beta=beta, eps=eps)
    rz_cosine = RunningZScore(beta=beta, eps=eps)

    def zscore_batch(x: np.ndarray) -> np.ndarray:
        mu = x.mean()
        sd = x.std()
        return (x - mu) / (sd + eps)

    def clip(x: np.ndarray) -> np.ndarray:
        if clip_z and clip_z > 0:
            return np.clip(x, -clip_z, clip_z)
        return x

    def combined_reward(prompts, completions, **kw):
        # TRL passes dataset columns via kwargs (confirmed by your logs).
        chosen = kw.get("chosen")
        if chosen is None:
            raise ValueError("Expected 'chosen' in kwargs for COMET/cosine rewards.")

        prompt_raw = kw.get("prompt_raw")

        # --- raw component rewards (vectors of shape [batch]) ---
        r_d = np.array(dialect_reward(prompts, completions), dtype=np.float32)

        r_c = np.array(
            comet_reward_with_ref(
                prompts,
                completions,
                chosen=chosen,
                prompt_raw=prompt_raw,
                model_name=rcfg.get("comet_model_name", "Unbabel/wmt22-comet-da"),
                batch_size=int(rcfg.get("comet_batch_size", 8)),
                force_cpu=bool(rcfg.get("comet_force_cpu", True)),
            ),
            dtype=np.float32,
        )

        r_s = np.array(
            embedding_similarity_reward(
                completions=completions,
                chosen=chosen,
                sim_model_name=rcfg.get("sim_model_name", "sentence-transformers/all-MiniLM-L6-v2"),
            ),
            dtype=np.float32,
        )

        # --- normalize each component ---
        if method == "none":
            z_d, z_c, z_s = r_d, r_c, r_s

        elif method == "batch_zscore":
            z_d = clip(zscore_batch(r_d))
            z_c = clip(zscore_batch(r_c))
            z_s = clip(zscore_batch(r_s))

        elif method == "running_zscore":
            rz_dialect.update(r_d)
            rz_comet.update(r_c)
            rz_cosine.update(r_s)

            if rz_dialect.steps <= warmup_steps:
                z_d = clip(zscore_batch(r_d))
                z_c = clip(zscore_batch(r_c))
                z_s = clip(zscore_batch(r_s))
            else:
                z_d = clip(rz_dialect.normalize(r_d))
                z_c = clip(rz_comet.normalize(r_c))
                z_s = clip(rz_cosine.normalize(r_s))
        else:
            raise ValueError(f"Unknown rewards.normalization.method: {method}")

        total = (w_dialect * z_d) + (w_comet * z_c) + (w_cosine * z_s)

        # Log kwargs keys once (helps debug TRL version differences)
        if not hasattr(combined_reward, "_logged_keys"):
            logger.info("combined_reward kwargs keys: %s", sorted(list(kw.keys())))
            combined_reward._logged_keys = True

        # Log component means (raw) and normalized total mean
        logger.info(
            "reward raw mean | dialect=%.4f comet=%.4f cosine=%.4f | total(norm)=%.4f",
            float(r_d.mean()), float(r_c.mean()), float(r_s.mean()), float(total.mean())
        )

        return total.astype(np.float32).tolist()

    return combined_reward


# =============================================================================
# GRPOConfig compatibility helper
# =============================================================================

def build_grpo_config(cfg: Dict[str, Any], logger: logging.Logger, tokenizer, eos_ids: List[int]) -> GRPOConfig:
    """
    Build GRPOConfig in a way that is robust across TRL versions:
    - Filters cfg["trainer"] to only args accepted by GRPOConfig.__init__
    - Injects generation_kwargs with eos/pad token ids if supported
    """
    raw_args = dict(cfg.get("trainer", {}))

    # Make sure generation kwargs include stopping token ids (Transformers supports list eos_token_id)
    gen_kwargs = dict(raw_args.get("generation_kwargs", {}))
    gen_kwargs["eos_token_id"] = eos_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    raw_args["generation_kwargs"] = gen_kwargs

    sig = inspect.signature(GRPOConfig.__init__)
    allowed = set(sig.parameters.keys())
    allowed.discard("self")

    filtered_args = {k: v for k, v in raw_args.items() if k in allowed}
    dropped = sorted(set(raw_args.keys()) - set(filtered_args.keys()))
    if dropped:
        logger.warning("Dropping unsupported GRPOConfig args for this TRL version: %s", dropped)

    return GRPOConfig(**filtered_args)


# =============================================================================
# Main entrypoint
# =============================================================================

def main() -> None:
    # ---- CLI ----
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True, help="Path to JSON config file")
    args = parser.parse_args()

    logger = setup_logging()

    # ---- Load config ----
    with open(args.config, "r") as f:
        cfg: Dict[str, Any] = json.load(f)

    # ---- Optional HF login (helps private repos / rate limits) ----
    hf_token = cfg.get("hf_token") or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if hf_token:
        hf_login(token=hf_token)
        logger.info("Logged into Hugging Face Hub via token")

    # ---- Reproducibility ----
    seed = int(cfg.get("data", {}).get("seed", 42))
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ---- Load model + tokenizer ----
    model, tokenizer = load_policy_and_tokenizer(cfg, logger)

    # ---- Load dataset snapshot ----
    dcfg = cfg["data"]
    ds = load_dataset_from_hub_snapshot(
        dataset_id=dcfg["dataset_id"],
        split=dcfg.get("dataset_split", "train"),
        logger=logger,
    )

    # Ensure required columns are present
    required_cols = ["prompt", "chosen", "rejected"]
    for col in required_cols:
        if col not in ds.column_names:
            raise ValueError(f"Dataset missing required column '{col}'. Found columns: {ds.column_names}")

    # Train/test split (tiny eval split by default)
    ds = ds.train_test_split(test_size=float(dcfg.get("test_size", 0.02)), seed=seed)
    train_ds, eval_ds = ds["train"], ds["test"]
    logger.info("Train/Eval sizes: %d / %d", len(train_ds), len(eval_ds))

    # Optional smoke subset
    n_tr = int(dcfg.get("smoke_subset_train", 0) or 0)
    n_ev = int(dcfg.get("smoke_subset_eval", 0) or 0)
    if n_tr > 0:
        train_ds = train_ds.select(range(min(n_tr, len(train_ds))))
        logger.info("Smoke subset train: %d rows", len(train_ds))
    if n_ev > 0:
        eval_ds = eval_ds.select(range(min(n_ev, len(eval_ds))))
        logger.info("Smoke subset eval: %d rows", len(eval_ds))

    # ---- Prompt formatting ----
    system_prompt = dcfg.get("system_prompt", "") or ""

    # Stop strings used for *reward scoring trim* (not generation stopping)
    stop_strings = [
        "\nUser:",
        "\nAssistant:",
        "\n### User:",
        "\n### Assistant:",
        "\n<|user|>",
        "\n<|assistant|>",
    ]

    def map_fn(ex: Dict[str, Any]) -> Dict[str, Any]:
        raw = ex["prompt"]
        ex["prompt_raw"] = raw
        ex["prompt"] = build_prompt(tokenizer, system_prompt, raw, prefer_chat_template=True)
        return ex

    train_ds = train_ds.map(map_fn)
    eval_ds = eval_ds.map(map_fn)

    # ---- LoRA config ----
    pcfg = cfg.get("peft", {})
    lora_cfg = LoraConfig(
        r=int(pcfg.get("r", 8)),
        lora_alpha=int(pcfg.get("lora_alpha", 16)),
        lora_dropout=float(pcfg.get("lora_dropout", 0.05)),
        target_modules=list(pcfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])),
        bias=str(pcfg.get("bias", "none")),
        task_type=str(pcfg.get("task_type", "CAUSAL_LM")),
    )

    # ---- Generation stopping token ids ----
    eos_ids = infer_stop_token_ids(tokenizer)
    logger.info("tokenizer.pad_token_id=%s eos_token_id=%s eos_ids=%s",
                tokenizer.pad_token_id, tokenizer.eos_token_id, eos_ids)

    # ---- Build rewards ----
    combined_reward = make_combined_reward(cfg, logger)
    trim_wrapper = make_trim_wrapper(stop_strings)
    reward_funcs = [trim_wrapper(combined_reward)]

    # ---- Build GRPOConfig (version-safe) ----
    grpo_args = build_grpo_config(cfg, logger, tokenizer, eos_ids)

    # Debug model placement
    logger.info("model.device=%s dtype=%s", next(model.parameters()).device, next(model.parameters()).dtype)

    from transformers.tokenization_utils_base import BatchEncoding
    import traceback

    _orig_to = BatchEncoding.to

    def _to_debug(self, *args, **kwargs):
        # The message you're seeing happens when the first positional arg is None,
        # e.g. batch.to(None) OR when device=None is passed explicitly.
        if (len(args) >= 1 and args[0] is None) or ("device" in kwargs and kwargs["device"] is None):
            print("\n[DEBUG] BatchEncoding.to(None) called. Stack trace:")
            traceback.print_stack(limit=25)
            print()
        return _orig_to(self, *args, **kwargs)

    BatchEncoding.to = _to_debug

    # ---- Build trainer ----
    peft_cfg = cfg.get("peft", {})
    use_peft = bool(peft_cfg.get("enabled", True))

    lora_cfg = None
    if use_peft:
        lora_cfg = LoraConfig(
            r=int(peft_cfg.get("r", 8)),
            lora_alpha=int(peft_cfg.get("lora_alpha", 16)),
            lora_dropout=float(peft_cfg.get("lora_dropout", 0.05)),
            target_modules=list(peft_cfg.get("target_modules", ["q_proj","k_proj","v_proj","o_proj"])),
            bias=str(peft_cfg.get("bias", "none")),
            task_type=str(peft_cfg.get("task_type", "CAUSAL_LM")),
        )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=lora_cfg,   # None = no LoRA
        reward_funcs=reward_funcs,
    )

    # ---- Train ----
    trainer.train()

    # ---- Save LoRA adapter + tokenizer ----
    out_dir = trainer.args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    logger.info("Saved adapter+tokenizer to %s", out_dir)

    # ---- Quick sanity generation (rank 0 only) ----
    if get_local_rank() == 0:
        try:
            logger.info("Running quick sanity generation...")
            test_prompt = "Write a short friendly reply in British English about making a cup of tea."
            chat = build_prompt(tokenizer, system_prompt, test_prompt, prefer_chat_template=True)
            inputs = tokenizer(chat, return_tensors="pt").to(trainer.model.device)

            # Use the same eos_ids list here too
            max_new = int(cfg.get("trainer", {}).get("max_completion_length", 64))
            temp = float(cfg.get("trainer", {}).get("temperature", 0.9))
            top_p = float(cfg.get("trainer", {}).get("top_p", 0.95))

            with torch.no_grad():
                gen = trainer.model.generate(
                    **inputs,
                    max_new_tokens=min(max_new, 128),
                    do_sample=True,
                    temperature=temp,
                    top_p=top_p,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=eos_ids,
                )

            decoded = tokenizer.decode(gen[0], skip_special_tokens=True)
            completion = decoded[len(chat):] if decoded.startswith(chat) else decoded
            completion = hard_trim_completion(completion, stop_strings)
            logger.info("SANITY COMPLETION:\n%s", completion.strip())
        except Exception as e:
            logger.warning("Sanity generation failed (non-fatal): %s", e)


if __name__ == "__main__":
    main()