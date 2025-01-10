from typing import List, Tuple

class ReRanker:
    def __init__(
            self,
            model_name: str = "ms-marco-MiniLM-L-12-v2",
            device: str = "cpu",
    ):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ValueError(
                "The sentence_transformers python package is not installed. Please install it with `pip install sentence_transformers`"
            )
        self.model_name = f"cross-encoder/{model_name}"
        self.__model = CrossEncoder(self.model_name, device=device)

    def __call__(self, pairs: List[Tuple[str, str]]):
        out = self.__model.predict(pairs)
        return out