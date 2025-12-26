from sentence_transformers import SentenceTransformer
from typing import List, Union
from app.core.config import settings

class OfflineEmbedder:
    """
    Offline embedding model using sentence-transformers.
    """

    def __init__(self):
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.dim = self.model.get_sentence_embedding_dimension()
        
    def embed(self, text: Union[str, List[str]]) -> List[float]:
        """
        Embed a single query string or a list of strings.
        Returns a list of floats (if single string) or list of list of floats?
        Actually for Qdrant upload we usually want list of list of floats if input is list.
        But the original code returned `vector.tolist()`.
        Let's standardize: 
        - If input is str -> return List[float]
        - If input is List[str] -> return List[List[float]]
        """
        if isinstance(text, str):
            vector = self.model.encode(text, normalize_embeddings=True)
            return vector.tolist()
        else:
            vectors = self.model.encode(text, normalize_embeddings=True)
            return vectors.tolist()

    def embed_query(self, query: str) -> List[float]:
        """
        Alias for retrieval compatibility.
        """
        return self.embed(query)

