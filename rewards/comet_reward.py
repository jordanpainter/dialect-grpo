# rewards/comet_reward.py
import os
from typing import List, Optional, Tuple, Dict

import torch
import torch.distributed as dist

# COMET package: pip install unbabel-comet
from comet import download_model, load_from_checkpoint


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
        lr = _get_local_rank()
        torch.cuda.set_device(lr)
        return torch.device(f"cuda:{lr}")
    return torch.device("cpu")


# Cache per (model_name, device_str) so you can swap models/devices safely
_COMET_CACHE: Dict[Tuple[str, str], object] = {}


def _load_comet(model_name: str, device: torch.device):
    key = (model_name, str(device))
    if key in _COMET_CACHE:
        return _COMET_CACHE[key]

    # Avoid multiple ranks downloading simultaneously (can be flaky on shared FS)
    if _is_dist():
        if dist.get_rank() == 0:
            _ = download_model(model_name)
        _barrier()
        ckpt_path = download_model(model_name)  # uses cache for non-rank0 after barrier
    else:
        ckpt_path = download_model(model_name)

    model = load_from_checkpoint(ckpt_path)
    model.to(device)
    model.eval()

    _COMET_CACHE[key] = model
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

    # If accelerate/DDP sets CUDA_VISIBLE_DEVICES per rank (typical),
    # then gpus=1 means "use the single visible GPU" for that rank.
    use_gpu = (device.type == "cuda")
    out = model.predict(data, batch_size=batch_size, gpus=1 if use_gpu else 0)

    scores = out["scores"]
    return [float(s) for s in scores]


@torch.inference_mode()
def comet_reward_with_ref(
    prompts: List[str],
    completions: List[str],
    *,
    chosen: List[str],
    prompt_raw: Optional[List[str]] = None,
    model_name: str = "Unbabel/wmt22-comet-da",
    batch_size: int = 8,
    force_cpu: bool = True,
    **kwargs,
) -> List[float]:
    """
    Reference-based COMET: uses src + mt + ref
    Here ref = chosen completion, mt = generated completion.
    """
    assert chosen is not None
    srcs = prompt_raw if prompt_raw is not None else prompts
    device = _pick_device(force_cpu=force_cpu)
    model = _load_comet(model_name, device=device)

    data = [{"src": s, "mt": m, "ref": r} for s, m, r in zip(srcs, completions, chosen)]

    use_gpu = (device.type == "cuda")
    out = model.predict(data, batch_size=batch_size, gpus=1 if use_gpu else 0)

    scores = out["scores"]
    return [float(s) for s in scores]