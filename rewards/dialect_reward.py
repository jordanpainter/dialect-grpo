# rewards/dialect_reward.py
from __future__ import annotations

import os
from typing import List, Optional

import torch

from .dialect_reward_model import RewardModel

# Lazy singleton so we only load weights once per process
_RM: Optional[RewardModel] = None


def _get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))


def _default_device() -> str:
    if torch.cuda.is_available():
        lr = _get_local_rank()
        torch.cuda.set_device(lr)
        return f"cuda:{lr}"
    return "cpu"


def _get_rm() -> RewardModel:
    global _RM
    if _RM is None:
        model_path = os.environ.get("DIALECT_REWARD_MODEL", "srirag/feature-identifier")
        device = os.environ.get("DIALECT_REWARD_DEVICE", None)  # override if set
        if device is None:
            device = _default_device()
        _RM = RewardModel(model_path=model_path, device=device)
    return _RM


def dialect_reward(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """
    GRPO reward function: takes prompts + completions, returns a scalar reward per completion.
    Uses the dialect feature identifier model.
    """
    rm = _get_rm()

    # The reward model expects a list of texts; we only score completions here
    rewards_t = rm.reward(completions)
    if isinstance(rewards_t, torch.Tensor):
        rewards_t = rewards_t.detach().cpu()
        return [float(x) for x in rewards_t.tolist()]
    return [float(x) for x in rewards_t]