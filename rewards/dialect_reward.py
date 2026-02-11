from typing import List

class DialectRewardStub:
    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        return [0.0 for _ in completions]
