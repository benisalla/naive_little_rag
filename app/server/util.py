from typing import cast
import numpy as np
from chromadb import Embeddings, Documents, EmbeddingFunction
from llama_index.core import SimpleDirectoryReader

def textDB2EmbDB(collection, data_dir, token_splitter, is_add=False):
    documents = SimpleDirectoryReader(input_dir=data_dir).load_data()
    nodes = token_splitter.get_nodes_from_documents(documents)

    prev_id = max(np.array(collection.get()["ids"]).astype(int)) if is_add else 0
    ids = []
    docs = []
    metadata = []
    for idx, node in enumerate(nodes):

        if not node.text:
            continue

        ids.append(str(idx + prev_id))

        metadata.append({
            "id_": node.id_,
            **node.metadata,
            "start_char_idx": node.start_char_idx,
            "end_char_idx": node.end_char_idx,
            "metadata_seperator": node.metadata_seperator
        })

        docs.append(node.text)

    collection.add(ids=ids, documents=docs, metadatas=metadata)

    return collection


class RagEmbedder(EmbeddingFunction[Documents]):
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
