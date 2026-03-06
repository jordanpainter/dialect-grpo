import os
from typing import Dict, Tuple

import torch
from sentence_transformers import SentenceTransformer

# Cache per (model_name, device)
_ST_MODELS: Dict[Tuple[str, str], SentenceTransformer] = {}


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))


def _pick_device() -> str:
    if torch.cuda.is_available():
        lr = _local_rank()
        torch.cuda.set_device(lr)
        return f"cuda:{lr}"
    return "cpu"


def _get_model(model_name: str, device: str) -> SentenceTransformer:
    key = (model_name, device)
    if key not in _ST_MODELS:
        _ST_MODELS[key] = SentenceTransformer(model_name, device=device)
    return _ST_MODELS[key]


def embedding_margin_reward(
    prompts=None,
    completions=None,
    chosen=None,
    rejected=None,
    **kwargs,
):
    """
    Reward = cos(emb(completion), emb(chosen)) - cos(emb(completion), emb(rejected))
    """
    assert completions is not None
    assert chosen is not None and rejected is not None

    device = _pick_device()
    model_name = kwargs.get("sim_model_name", "sentence-transformers/all-MiniLM-L6-v2")
    st = _get_model(model_name, device=device)

    emb_y = st.encode(completions, convert_to_tensor=True, normalize_embeddings=True)
    emb_c = st.encode(chosen, convert_to_tensor=True, normalize_embeddings=True)
    emb_r = st.encode(rejected, convert_to_tensor=True, normalize_embeddings=True)

    sim_c = (emb_y * emb_c).sum(dim=1)  # cosine since normalized
    sim_r = (emb_y * emb_r).sum(dim=1)

    rewards = (sim_c - sim_r).detach().cpu().tolist()
    return [float(x) for x in rewards]


def embedding_similarity_reward(
    completions=None,
    chosen=None,
    **kwargs,
):
    """
    Reward = cos(emb(completion), emb(chosen))
    """
    assert completions is not None
    assert chosen is not None

    device = _pick_device()
    model_name = kwargs.get("sim_model_name", "sentence-transformers/all-MiniLM-L6-v2")
    st = _get_model(model_name, device=device)

    emb_y = st.encode(completions, convert_to_tensor=True, normalize_embeddings=True)
    emb_c = st.encode(chosen, convert_to_tensor=True, normalize_embeddings=True)

    sim = (emb_y * emb_c).sum(dim=1)  # cosine since normalized
    rewards = sim.detach().cpu().tolist()
    return [float(x) for x in rewards]