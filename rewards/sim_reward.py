import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

# Cached global model to avoid re-loading every call
_ST_MODEL = None

def _get_model(model_name: str, device: str):
    global _ST_MODEL
    if _ST_MODEL is None:
        _ST_MODEL = SentenceTransformer(model_name, device=device)
    return _ST_MODEL

def embedding_margin_reward(
    prompts=None,
    completions=None,
    chosen=None,
    rejected=None,
    **kwargs,
):
    """
    Reward = cos(emb(completion), emb(chosen)) - cos(emb(completion), emb(rejected))

    TRL will pass completions as a list of strings (standard format).
    chosen/rejected are lists aligned to completions length.
    """
    assert completions is not None
    assert chosen is not None and rejected is not None

    # If TRL repeats dataset rows for num_generations, chosen/rejected will already be repeated.
    # If not, you can expand them here (but usually TRL handles it).
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = kwargs.get("sim_model_name", "sentence-transformers/all-MiniLM-L6-v2")

    st = _get_model(model_name, device=device)

    # Encode in batch. SentenceTransformer returns numpy by default; ask for tensor.
    emb_y = st.encode(completions, convert_to_tensor=True, normalize_embeddings=True)
    emb_c = st.encode(chosen, convert_to_tensor=True, normalize_embeddings=True)
    emb_r = st.encode(rejected, convert_to_tensor=True, normalize_embeddings=True)

    sim_c = (emb_y * emb_c).sum(dim=1)  # cosine since normalized
    sim_r = (emb_y * emb_r).sum(dim=1)

    rewards = (sim_c - sim_r).detach().cpu().tolist()
    return [float(x) for x in rewards]
