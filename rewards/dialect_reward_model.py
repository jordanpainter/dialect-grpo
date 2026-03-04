import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from .dialect_feature_model import MultiheadDialectFeatureModel

class RewardModel:
    """
    Loads a trained multi-head dialect feature classifier and
    exposes an RL-ready scalar reward function.

    Reward = log(1 + sum(sigmoid(logits)))
    Optionally apply a length penalty or normalisation.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        length_penalty_alpha: float = 0.0,
    ):
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # load model
        self.model = MultiheadDialectFeatureModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        # number of features (for potential normalisation)
        self.num_features = self.model.num_features

    @torch.no_grad()
    def predict_raw_scores(self, texts: list[str], max_length: int = 256):
        """
        Returns raw sum of probabilities per text, and lengths.
        """

        # tokenizer collates batch
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length,
        ).to(self.device)

        outputs = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

        logits = outputs.logits
        probs = torch.sigmoid(logits)

        # sum of probabilities
        raw = probs.sum(dim=1)

        # number of tokens in text (for length penalty)
        lengths = inputs["attention_mask"].sum(dim=1).float()

        return raw.cpu(), lengths.cpu()

    @torch.no_grad()
    def reward(
        self,
        texts: list[str],
        max_length: int = 256,
        normalise: bool = False,
    ) -> torch.Tensor:
        """
        Returns scalar reward for a batch of texts.

        Reward = log1p(raw_sum)
        with optional length penalty and rate normalisation.
        """

        raw, lengths = self.predict_raw_scores(texts, max_length=max_length)

        # apply normalisation if requested
        if normalise:
            raw = raw / self.num_features

        # log1p compression
        reward = torch.log1p(raw)

        return reward

    @torch.no_grad()
    def margin(
        self,
        chosen: list[str],
        rejected: list[str],
        **kwargs,
    ) -> torch.Tensor:
        """
        Computes margin reward for a batch of (chosen, rejected) pairs.

        Useful for ranking-based RL objectives.
        """

        # compute raw reward on both sets
        r_chosen = self.reward(chosen, **kwargs)
        r_rejected = self.reward(rejected, **kwargs)

        return r_chosen - r_rejected

    def to(self, device: str):
        """
        Moves model & tokenizer to a different device if needed.
        """
        self.device = device
        self.model.to(device)
        return self