"""
src/grpo.py

GRPO training script for dialectal English preference learning.

Main improvements in this version:
- Loads COMET once and reuses it across training.
- Loads SentenceTransformer once and reuses it across training.
- Logs individual raw and normalized reward components.
- Surfaces KL-like metrics if TRL provides them in trainer logs.
- Sets float32 matmul precision for Tensor Core GPUs.
- Keeps hard prompt length guard and chat prompt formatting.

Run:
  accelerate launch --num_processes=1 -m src.grpo -c configs/qwen.json
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from datasets import load_from_disk
from huggingface_hub import login as hf_login
from huggingface_hub import snapshot_download
from peft import LoraConfig
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from trl import GRPOConfig, GRPOTrainer

# COMET
from comet import download_model as comet_download_model
from comet import load_from_checkpoint as comet_load_from_checkpoint

# --- Reward components (project modules) ---
from rewards.dialect_reward import dialect_reward

# --- Prompt formatting fallback (project module) ---
from src.formatting import build_chat_prompt


# =============================================================================
# Logging and small utilities
# =============================================================================

def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("grpo")


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))


def ensure_pad_token(tokenizer) -> None:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def hard_trim_completion(text: str, stop_strings: Sequence[str]) -> str:
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


def resolve_model_max_length(tokenizer, model, fallback: int = 2048) -> int:
    tmax = getattr(tokenizer, "model_max_length", None)
    if isinstance(tmax, int) and 64 < tmax < 100_000:
        return int(tmax)

    mmax = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    if isinstance(mmax, int) and 64 < mmax < 100_000:
        return int(mmax)

    return int(fallback)


def truncate_prompt_to_max_tokens(tokenizer, text: str, max_tokens: int) -> str:
    if not text:
        return text

    enc = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        truncation=True,
        max_length=max_tokens,
    )
    ids = enc["input_ids"]
    return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


class RunningZScore:
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
    mcfg = cfg["model"]
    model_id = mcfg["model_id"]
    tok_id = mcfg.get("tokenizer_id", model_id)

    tokenizer = AutoTokenizer.from_pretrained(tok_id, use_fast=True)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    ensure_pad_token(tokenizer)

    quant_cfg = _build_quant_config(mcfg)

    local_rank = get_local_rank()
    device_map = {"": local_rank}
    logger.info("LOCAL_RANK=%s | device_map=%s", local_rank, device_map)

    dtype = torch.bfloat16 if cfg.get("trainer", {}).get("bf16", False) else None

    from_pretrained_kwargs: Dict[str, Any] = dict(
        low_cpu_mem_usage=True,
        quantization_config=quant_cfg,
        device_map=device_map,
        return_dict=True,
    )
    if dtype is not None:
        from_pretrained_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(model_id, **from_pretrained_kwargs)

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = not bool(cfg.get("trainer", {}).get("gradient_checkpointing", False))

    return model, tokenizer


# =============================================================================
# Dataset loading and prompt formatting
# =============================================================================

def load_dataset_from_hub_snapshot(dataset_id: str, split: str, logger: logging.Logger):
    local_path = snapshot_download(dataset_id, repo_type="dataset")
    logger.info("Downloaded dataset snapshot to: %s", local_path)

    ds_any = load_from_disk(local_path)

    if hasattr(ds_any, "column_names"):
        logger.info("Loaded dataset (Dataset) with columns: %s", ds_any.column_names)
        return ds_any

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
    if prefer_chat_template and hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return build_chat_prompt(tokenizer, system_prompt, user_prompt)


def infer_stop_token_ids(tokenizer) -> List[int]:
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
    for i in eos_ids:
        if i not in seen:
            deduped.append(i)
            seen.add(i)
    return deduped


# =============================================================================
# Cached reward scorers
# =============================================================================

class CachedCosineScorer:
    def __init__(self, model_name: str, logger: logging.Logger):
        self.model_name = model_name
        self.logger = logger
        self.model: Optional[SentenceTransformer] = None

    def _ensure_loaded(self) -> None:
        if self.model is None:
            self.logger.info("Loading SentenceTransformer once: %s", self.model_name)
            self.model = SentenceTransformer(self.model_name)

    def score(self, completions: Sequence[str], chosen: Sequence[str]) -> np.ndarray:
        self._ensure_loaded()
        assert self.model is not None

        emb_c = self.model.encode(list(completions), convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
        emb_r = self.model.encode(list(chosen), convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
        sims = torch.sum(emb_c * emb_r, dim=-1)
        return sims.detach().cpu().float().numpy()


class CachedCometScorer:
    def __init__(self, model_name: str, batch_size: int, force_cpu: bool, logger: logging.Logger):
        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.force_cpu = bool(force_cpu)
        self.logger = logger
        self.model = None
        self.device = "cpu" if self.force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    def _ensure_loaded(self) -> None:
        if self.model is None:
            self.logger.info("Loading COMET once: %s", self.model_name)
            ckpt_path = comet_download_model(self.model_name)
            self.model = comet_load_from_checkpoint(ckpt_path)

            # Move once if possible
            if hasattr(self.model, "to"):
                self.model = self.model.to(self.device)

            if hasattr(self.model, "eval"):
                self.model.eval()

    def score(self, prompts: Sequence[str], completions: Sequence[str], chosen: Sequence[str]) -> np.ndarray:
        self._ensure_loaded()
        assert self.model is not None

        data = [{"src": p, "mt": c, "ref": r} for p, c, r in zip(prompts, completions, chosen)]

        # COMET API returns (scores, system_score) in current common versions
        if self.device == "cuda":
            out = self.model.predict(
                data,
                batch_size=self.batch_size,
                gpus=1,
                progress_bar=False,
            )
        else:
            out = self.model.predict(
                data,
                batch_size=self.batch_size,
                gpus=0,
                progress_bar=False,
            )

        if isinstance(out, tuple):
            scores = out[0]
        else:
            scores = out.scores if hasattr(out, "scores") else out

        return np.array(scores, dtype=np.float32)


# =============================================================================
# Reward construction
# =============================================================================

@dataclass
class RewardTelemetry:
    raw_dialect_mean: float = 0.0
    raw_comet_mean: float = 0.0
    raw_cosine_mean: float = 0.0
    norm_dialect_mean: float = 0.0
    norm_comet_mean: float = 0.0
    norm_cosine_mean: float = 0.0
    total_mean: float = 0.0
    total_std: float = 0.0


def make_trim_wrapper(stop_strings: Sequence[str]):
    def trim_wrapper(reward_fn):
        def _wrapped(prompts, completions, **kwargs):
            trimmed = [hard_trim_completion(c, stop_strings) for c in completions]
            return reward_fn(prompts, trimmed, **kwargs)
        _wrapped.__name__ = getattr(reward_fn, "__name__", "reward_fn")
        return _wrapped
    return trim_wrapper


class CombinedReward:
    def __init__(self, cfg: Dict[str, Any], logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger

        rcfg = cfg.get("rewards", {})
        weights = rcfg.get("weights", {})
        self.w_dialect = float(weights.get("dialect", 0.50))
        self.w_comet = float(weights.get("comet", 0.25))
        self.w_cosine = float(weights.get("cosine", 0.25))

        ncfg = rcfg.get("normalization", {})
        self.method = str(ncfg.get("method", "batch_zscore"))
        self.clip_z = float(ncfg.get("clip_z", 5.0))
        self.eps = float(ncfg.get("eps", 1e-6))
        self.beta = float(ncfg.get("beta", 0.99))
        self.warmup_steps = int(ncfg.get("warmup_steps", 0))

        self.rz_dialect = RunningZScore(beta=self.beta, eps=self.eps)
        self.rz_comet = RunningZScore(beta=self.beta, eps=self.eps)
        self.rz_cosine = RunningZScore(beta=self.beta, eps=self.eps)

        self.comet = CachedCometScorer(
            model_name=rcfg.get("comet_model_name", "Unbabel/wmt22-comet-da"),
            batch_size=int(rcfg.get("comet_batch_size", 8)),
            force_cpu=bool(rcfg.get("comet_force_cpu", True)),
            logger=logger,
        )
        self.cosine = CachedCosineScorer(
            model_name=rcfg.get("sim_model_name", "sentence-transformers/all-MiniLM-L6-v2"),
            logger=logger,
        )

        self.latest = RewardTelemetry()
        self._logged_keys = False

    def _zscore_batch(self, x: np.ndarray) -> np.ndarray:
        mu = x.mean()
        sd = x.std()
        return (x - mu) / (sd + self.eps)

    def _clip(self, x: np.ndarray) -> np.ndarray:
        if self.clip_z and self.clip_z > 0:
            return np.clip(x, -self.clip_z, self.clip_z)
        return x

    def __call__(self, prompts, completions, **kw):
        chosen = kw.get("chosen")
        if chosen is None:
            raise ValueError("Expected 'chosen' in kwargs for COMET/cosine rewards.")

        prompt_raw = kw.get("prompt_raw", prompts)

        r_d = np.array(dialect_reward(prompts, completions), dtype=np.float32)
        r_c = self.comet.score(prompt_raw, completions, chosen)
        r_s = self.cosine.score(completions, chosen)

        if self.method == "none":
            z_d, z_c, z_s = r_d, r_c, r_s

        elif self.method == "batch_zscore":
            z_d = self._clip(self._zscore_batch(r_d))
            z_c = self._clip(self._zscore_batch(r_c))
            z_s = self._clip(self._zscore_batch(r_s))

        elif self.method == "running_zscore":
            self.rz_dialect.update(r_d)
            self.rz_comet.update(r_c)
            self.rz_cosine.update(r_s)

            if self.rz_dialect.steps <= self.warmup_steps:
                z_d = self._clip(self._zscore_batch(r_d))
                z_c = self._clip(self._zscore_batch(r_c))
                z_s = self._clip(self._zscore_batch(r_s))
            else:
                z_d = self._clip(self.rz_dialect.normalize(r_d))
                z_c = self._clip(self.rz_comet.normalize(r_c))
                z_s = self._clip(self.rz_cosine.normalize(r_s))
        else:
            raise ValueError(f"Unknown rewards.normalization.method: {self.method}")

        total = (self.w_dialect * z_d) + (self.w_comet * z_c) + (self.w_cosine * z_s)

        self.latest = RewardTelemetry(
            raw_dialect_mean=float(r_d.mean()),
            raw_comet_mean=float(r_c.mean()),
            raw_cosine_mean=float(r_s.mean()),
            norm_dialect_mean=float(z_d.mean()),
            norm_comet_mean=float(z_c.mean()),
            norm_cosine_mean=float(z_s.mean()),
            total_mean=float(total.mean()),
            total_std=float(total.std()),
        )

        if not self._logged_keys:
            self.logger.info("combined_reward kwargs keys: %s", sorted(list(kw.keys())))
            self._logged_keys = True

        self.logger.info(
            "reward raw mean | dialect=%.4f comet=%.4f cosine=%.4f | total(norm)=%.4f",
            self.latest.raw_dialect_mean,
            self.latest.raw_comet_mean,
            self.latest.raw_cosine_mean,
            self.latest.total_mean,
        )

        return total.astype(np.float32).tolist()


# =============================================================================
# Trainer subclass for better logging
# =============================================================================

class LoggingGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, reward_tracker: Optional[CombinedReward] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_tracker = reward_tracker

    def log(self, logs: Dict[str, float], *args, **kwargs):
        logs = dict(logs)

        # Attach latest reward component telemetry
        if self.reward_tracker is not None:
            rt = self.reward_tracker.latest
            logs.setdefault("train/reward_raw/dialect_mean", rt.raw_dialect_mean)
            logs.setdefault("train/reward_raw/comet_mean", rt.raw_comet_mean)
            logs.setdefault("train/reward_raw/cosine_mean", rt.raw_cosine_mean)
            logs.setdefault("train/reward_norm/dialect_mean", rt.norm_dialect_mean)
            logs.setdefault("train/reward_norm/comet_mean", rt.norm_comet_mean)
            logs.setdefault("train/reward_norm/cosine_mean", rt.norm_cosine_mean)
            logs.setdefault("train/reward_total/mean", rt.total_mean)
            logs.setdefault("train/reward_total/std", rt.total_std)

        # Surface KL-like keys if TRL provides them under another name
        for src_key in ["kl", "approx_kl", "objective/kl", "train/kl", "train/approx_kl"]:
            if src_key in logs:
                logs.setdefault("train/kl", logs[src_key])

        super().log(logs, *args, **kwargs)


# =============================================================================
# GRPOConfig compatibility helper
# =============================================================================

def build_grpo_config(cfg: Dict[str, Any], logger: logging.Logger, tokenizer, eos_ids: List[int]) -> GRPOConfig:
    raw_args = dict(cfg.get("trainer", {}))

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True, help="Path to JSON config file")
    args = parser.parse_args()

    logger = setup_logging()

    # Better Tensor Core utilization on A100
    torch.set_float32_matmul_precision("high")

    with open(args.config, "r") as f:
        cfg: Dict[str, Any] = json.load(f)

    logger.info("Loaded config from: %s", args.config)

    hf_token = cfg.get("hf_token") or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if hf_token:
        hf_login(token=hf_token)
        logger.info("Logged into Hugging Face Hub via token")

    seed = int(cfg.get("data", {}).get("seed", 42))
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model, tokenizer = load_policy_and_tokenizer(cfg, logger)

    dcfg = cfg["data"]
    ds = load_dataset_from_hub_snapshot(
        dataset_id=dcfg["dataset_id"],
        split=dcfg.get("dataset_split", "train"),
        logger=logger,
    )

    required_cols = ["prompt", "chosen", "rejected"]
    for col in required_cols:
        if col not in ds.column_names:
            raise ValueError(f"Dataset missing required column '{col}'. Found columns: {ds.column_names}")

    ds = ds.train_test_split(test_size=float(dcfg.get("test_size", 0.02)), seed=seed)
    train_ds, eval_ds = ds["train"], ds["test"]
    logger.info("Train/Eval sizes: %d / %d", len(train_ds), len(eval_ds))

    n_tr = int(dcfg.get("smoke_subset_train", 0) or 0)
    n_ev = int(dcfg.get("smoke_subset_eval", 0) or 0)
    if n_tr > 0:
        train_ds = train_ds.select(range(min(n_tr, len(train_ds))))
        logger.info("Smoke subset train: %d rows", len(train_ds))
    if n_ev > 0:
        eval_ds = eval_ds.select(range(min(n_ev, len(eval_ds))))
        logger.info("Smoke subset eval: %d rows", len(eval_ds))

    eos_ids = infer_stop_token_ids(tokenizer)
    logger.info(
        "tokenizer.pad_token_id=%s eos_token_id=%s eos_ids=%s",
        tokenizer.pad_token_id,
        tokenizer.eos_token_id,
        eos_ids,
    )

    system_prompt = dcfg.get("system_prompt", "") or ""

    stop_strings = [
        "\nUser:",
        "\nAssistant:",
        "\n### User:",
        "\n### Assistant:",
        "\n<|user|>",
        "\n<|assistant|>",
    ]

    model_max_len = resolve_model_max_length(tokenizer, model, fallback=2048)
    max_completion_len = int(cfg.get("trainer", {}).get("max_completion_length", 64))
    safety_margin = int(cfg.get("trainer", {}).get("length_safety_margin", 8))

    max_prompt_len = model_max_len - max_completion_len - safety_margin
    max_prompt_len = max(32, int(max_prompt_len))

    logger.info(
        "Length guard: model_max_len=%d, max_completion_len=%d, safety_margin=%d => max_prompt_len=%d",
        model_max_len, max_completion_len, safety_margin, max_prompt_len
    )

    tokenizer.model_max_length = max_prompt_len
    cfg.setdefault("trainer", {})["max_prompt_length"] = max_prompt_len

    def map_fn(ex: Dict[str, Any]) -> Dict[str, Any]:
        raw = ex["prompt"]
        ex["prompt_raw"] = raw

        built = build_prompt(tokenizer, system_prompt, raw, prefer_chat_template=True)
        built = truncate_prompt_to_max_tokens(tokenizer, built, max_prompt_len)

        ex["prompt"] = built
        return ex

    train_ds = train_ds.map(map_fn)
    eval_ds = eval_ds.map(map_fn)

    peft_cfg = cfg.get("peft", {})
    use_peft = bool(peft_cfg.get("enabled", True))

    lora_cfg = None
    if use_peft:
        lora_cfg = LoraConfig(
            r=int(peft_cfg.get("r", 8)),
            lora_alpha=int(peft_cfg.get("lora_alpha", 16)),
            lora_dropout=float(peft_cfg.get("lora_dropout", 0.05)),
            target_modules=list(peft_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])),
            bias=str(peft_cfg.get("bias", "none")),
            task_type=str(peft_cfg.get("task_type", "CAUSAL_LM")),
        )

    reward_tracker = CombinedReward(cfg, logger)
    trim_wrapper = make_trim_wrapper(stop_strings)
    reward_funcs = [trim_wrapper(reward_tracker)]

    grpo_args = build_grpo_config(cfg, logger, tokenizer, eos_ids)

    logger.info("model.device=%s dtype=%s", next(model.parameters()).device, next(model.parameters()).dtype)

    trainer = LoggingGRPOTrainer(
        model=model,
        args=grpo_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=lora_cfg,
        reward_funcs=reward_funcs,
        reward_tracker=reward_tracker,
    )

    trainer.train()

    out_dir = trainer.args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    logger.info("Saved adapter+tokenizer to %s", out_dir)

    if get_local_rank() == 0:
        try:
            logger.info("Running quick sanity generation...")
            test_prompt = "Write a short friendly reply in British English about making a cup of tea."
            chat = build_prompt(tokenizer, system_prompt, test_prompt, prefer_chat_template=True)
            chat = truncate_prompt_to_max_tokens(tokenizer, chat, max_prompt_len)

            inputs = tokenizer(chat, return_tensors="pt").to(trainer.model.device)

            max_new = int(cfg.get("trainer", {}).get("max_completion_length", 64))
            temp = float(cfg.get("trainer", {}).get("temperature", 0.9))
            top_p = float(cfg.get("trainer", {}).get("top_p", 0.95))

            with torch.no_grad():
                gen = trainer.model.generate(
                    **inputs,
                    max_new_tokens=max_new,
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
