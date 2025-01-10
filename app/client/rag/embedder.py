from typing import cast
from chromadb import Embeddings, Documents, EmbeddingFunction

class Embedder(EmbeddingFunction[Documents]):
    def __init__(
            self,
            model_name: str = "all-mpnet-base-v2",
            device: str = "cpu",
            normalize_embeddings: bool = False,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ValueError(
                "The sentence_transformers python package is not installed. Please install it with `pip install sentence_transformers`"
            )
        self.__model = SentenceTransformer(model_name, device=device)
        self.__norm_embd = normalize_embeddings

    def __call__(self, input: Documents) -> Embeddings:
        return cast(Embeddings, self.__model.encode(
            list(input),
            convert_to_numpy=True,
            normalize_embeddings=self.__norm_embd,
        ).tolist())