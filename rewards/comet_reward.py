# rewards/comet_reward.py
import os
from typing import List, Optional

import torch

# COMET package: pip install unbabel-comet
from comet import download_model, load_from_checkpoint

# ---- singleton cache (so each process loads once) ----
_COMET_MODEL = None
_COMET_NAME = None


def _get_local_rank() -> int:
    # works for torchrun/accelerate/DDP
    return int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        # In DDP each process is pinned to one GPU
        lr = _get_local_rank()
        return torch.device(f"cuda:{lr}")
    return torch.device("cpu")


def _load_comet(model_name: str, device: Optional[torch.device] = None):
    global _COMET_MODEL, _COMET_NAME

    if _COMET_MODEL is not None and _COMET_NAME == model_name:
        return _COMET_MODEL

    device = torch.device("cuda:1")

    # downloads to HF cache, returns checkpoint path
    ckpt_path = download_model(model_name)
    model = load_from_checkpoint(ckpt_path)

    # Make sure it's on the right device
    model.to(device)
    model.eval()

    _COMET_MODEL = model
    _COMET_NAME = model_name
    return model


@torch.inference_mode()
def cometkiwi_reward(
    prompts: List[str],
    completions: List[str],
    *,
    # TRL will pass extra columns via kwargs; we’ll try to use prompt_raw if present.
    prompt_raw: Optional[List[str]] = None,
    model_name: str = "Unbabel/wmt22-cometkiwi-da",
    batch_size: int = 16,
    gpus: Optional[int] = None,
    **kwargs,
) -> List[float]:
    """
    Reference-free QE reward using COMETKiwi.
    Uses src=prompt_raw (preferred) else src=prompts, and mt=completions.

    Returns: list[float] length == len(completions)
    """

    if prompt_raw is None:
        # fallback: prompts are your *chat-formatted* prompts, which is worse as "src"
        srcs = prompts
    else:
        srcs = prompt_raw

    model = _load_comet(model_name)

    data = [{"src": s, "mt": m} for s, m in zip(srcs, completions)]

    # In DDP each rank has its own GPU; so by default gpus=1 is fine.
    # If you're running on CPU, set gpus=0.
    if gpus is None:
        gpus = 1 if torch.cuda.is_available() else 0

    out = model.predict(data, batch_size=batch_size, gpus=gpus)
    scores = out["scores"]  # list[float], typically 0..1 for Kiwi models

    # Ensure plain Python floats
    return [float(s) for s in scores]
