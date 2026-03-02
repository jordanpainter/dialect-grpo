# rewards/comet_reward.py
import os
from typing import List, Optional

import torch
import torch.distributed as dist

# COMET package: pip install unbabel-comet
from comet import download_model, load_from_checkpoint

_COMET_MODEL = None
_COMET_NAME = None
_COMET_DEVICE = None


def _get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))


def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def _barrier():
    if _is_dist():
        dist.barrier()


def _pick_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        # each DDP process uses its own GPU
        lr = _get_local_rank()
        return torch.device(f"cuda:{lr}")
    return torch.device("cpu")


def _load_comet(model_name: str, device: torch.device):
    global _COMET_MODEL, _COMET_NAME, _COMET_DEVICE

    if (
        _COMET_MODEL is not None
        and _COMET_NAME == model_name
        and _COMET_DEVICE == str(device)
    ):
        return _COMET_MODEL

    # Avoid multiple ranks downloading simultaneously (can be flaky on shared FS)
    if _is_dist():
        if dist.get_rank() == 0:
            ckpt_path = download_model(model_name)
        _barrier()
        if dist.get_rank() != 0:
            ckpt_path = download_model(model_name)
    else:
        ckpt_path = download_model(model_name)

    model = load_from_checkpoint(ckpt_path)
    model.to(device)
    model.eval()

    _COMET_MODEL = model
    _COMET_NAME = model_name
    _COMET_DEVICE = str(device)
    return model


@torch.inference_mode()
def cometkiwi_reward(
    prompts: List[str],
    completions: List[str],
    *,
    prompt_raw: Optional[List[str]] = None,
    model_name: str = "Unbabel/wmt22-cometkiwi-da",
    batch_size: int = 8,
    force_cpu: bool = True,
    **kwargs,
) -> List[float]:
    """
    Reference-free QE reward using COMETKiwi.
    Uses src=prompt_raw (preferred) else src=prompts, and mt=completions.
    """

    srcs = prompt_raw if prompt_raw is not None else prompts
    device = _pick_device(force_cpu=force_cpu)

    model = _load_comet(model_name, device=device)

    data = [{"src": s, "mt": m} for s, m in zip(srcs, completions)]

    # IMPORTANT:
    # Using model.predict(..., gpus=1) under DDP can cause device selection conflicts.
    # For correctness-first smoke tests, keep this on CPU.
    out = model.predict(data, batch_size=batch_size, gpus=0)
    scores = out["scores"]
    return [float(s) for s in scores]
