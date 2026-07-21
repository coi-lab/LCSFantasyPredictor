"""
RAG Embedder module for LCS Fantasy Pipeline.
Handles document, match summary, and patch note embeddings.
"""

import math
import random
from typing import Dict, List, Union

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class RAGEmbedder:
    """
    Generates text and statistical feature embeddings for match summaries,
    patch notes, and historical learnings.
    """

    def __init__(self, model_name: str = "placeholder-embedding-model", dimension: int = 384):
        self.model_name = model_name
        self.dimension = dimension

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding vector for a single text document or query.

        Parameters
        ----------
        text : str
            Input text string (match summary, prompt snippet, or patch note).

        Returns
        -------
        List[float]
            Embedding vector float list of dimension `self.dimension`.
        """
        if not text:
            return [0.0] * self.dimension
        
        seed = abs(hash(text)) % (2**32)
        if HAS_NUMPY:
            np.random.seed(seed)
            vector = np.random.normal(loc=0.0, scale=1.0, size=self.dimension)
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            return vector.tolist()
        else:
            rng = random.Random(seed)
            vector = [rng.gauss(0.0, 1.0) for _ in range(self.dimension)]
            sq_sum = sum(x * x for x in vector)
            norm = math.sqrt(sq_sum)
            if norm > 0:
                vector = [x / norm for x in vector]
            return vector

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embedding vectors for a list of texts.
        """
        return [self.embed_text(t) for t in texts]

    def embed_match_summary(self, match_data: Dict[str, Union[str, float, int]]) -> List[float]:
        """
        Convert structured match summary dict into embedding format.
        """
        text_representation = (
            f"Game ID: {match_data.get('gameid')}, League: {match_data.get('league')}, "
            f"Patch: {match_data.get('patch')}, Player: {match_data.get('playername')}, "
            f"Position: {match_data.get('position')}, K/D/A: {match_data.get('kills')}/{match_data.get('deaths')}/{match_data.get('assists')}, "
            f"Fantasy Pts: {match_data.get('fantasy_pts')}"
        )
        return self.embed_text(text_representation)


if __name__ == "__main__":
    embedder = RAGEmbedder()
    vec = embedder.embed_text("Patch 14.1 jungle pathing changes")
    print(f"Generated embedding vector of length {len(vec)}")
