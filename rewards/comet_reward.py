from typing import List
import torch
from comet import download_model, load_from_checkpoint

class COMETReward:
    """
    reward = COMET(src, mt, ref=chosen) - COMET(src, mt, ref=rejected)

    TRL passes prompts, completions, and any extra dataset columns via **kwargs.
    """
    def __init__(self, checkpoint: str, device: str = "cpu", batch_size: int = 8, use_prompt_raw: bool = True):
        self.device = device
        self.batch_size = batch_size
        self.use_prompt_raw = use_prompt_raw

        ckpt_path = download_model(checkpoint)
        self.model = load_from_checkpoint(ckpt_path)
        self.model.eval()
        self.model.to(device)

    @torch.inference_mode()
    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        chosen: List[str],
        rejected: List[str],
        **kwargs,
    ) -> List[float]:
        srcs = kwargs.get("prompt_raw", prompts) if self.use_prompt_raw else prompts

        pos_data = [{"src": s, "mt": y, "ref": c} for s, y, c in zip(srcs, completions, chosen)]
        neg_data = [{"src": s, "mt": y, "ref": r} for s, y, r in zip(srcs, completions, rejected)]

        gpus = 1 if self.device.startswith("cuda") else 0

        # COMET predict returns an object with `.scores` (list of floats). :contentReference[oaicite:2]{index=2}
        pos_out = self.model.predict(pos_data, batch_size=self.batch_size, gpus=gpus)
        neg_out = self.model.predict(neg_data, batch_size=self.batch_size, gpus=gpus)

        pos_scores = list(pos_out.scores)
        neg_scores = list(neg_out.scores)
        return [float(a) - float(b) for a, b in zip(pos_scores, neg_scores)]
