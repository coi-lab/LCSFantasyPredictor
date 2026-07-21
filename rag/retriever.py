"""
RAG Retriever module for LCS Fantasy Pipeline.
Retrieves relevant match context, patch notes, and historical learnings.
"""

import os
import sys
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from learning.feedback_loop import LearningEngine
from rag.embedder import RAGEmbedder


class RAGRetriever:
    """
    Retrieves match context, patch notes, and self-correcting learnings
    to augment prompt context for fantasy analysis and projection models.
    """

    def __init__(
        self,
        embedder: Optional[RAGEmbedder] = None,
        learning_engine: Optional[LearningEngine] = None
    ):
        self.embedder = embedder or RAGEmbedder()
        self.learning_engine = learning_engine or LearningEngine()
        self.documents_index: List[Dict[str, Any]] = []

    def index_documents(self, docs: List[Dict[str, Any]]) -> int:
        """
        Add documents (match logs, patch notes, analyst articles) to in-memory vector index.
        """
        for doc in docs:
            content = doc.get("content", "")
            doc["embedding"] = self.embedder.embed_text(content)
            self.documents_index.append(doc)
        return len(self.documents_index)

    def retrieve_historical_learnings(self) -> Dict[str, Any]:
        """
        Fetch dynamic state, systemic biases, and prompt context snippets from LearningEngine.
        """
        return self.learning_engine.get_active_learnings()

    def query_context(self, query_text: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Query vector documents and join active systemic learnings to form augmented prompt context.

        Parameters
        ----------
        query_text : str
            Target search query.
        top_k : int
            Number of top matching documents to return.

        Returns
        -------
        Dict[str, Any]
            Dict containing retrieved documents, relevant learnings, and prompt snippets.
        """
        query_vec = self.embedder.embed_text(query_text)
        
        # Calculate cosine similarity with indexed documents
        results = []
        for doc in self.documents_index:
            doc_vec = doc.get("embedding", [])
            if doc_vec and len(doc_vec) == len(query_vec):
                sim = float(
                    sum(q * d for q, d in zip(query_vec, doc_vec))
                )  # vectors normalized by embedder
                results.append((sim, doc))

        results.sort(key=lambda x: x[0], reverse=True)
        retrieved_docs = [item[1] for item in results[:top_k]]

        active_learnings = self.retrieve_historical_learnings()

        return {
            "query": query_text,
            "retrieved_documents": retrieved_docs,
            "systemic_biases": active_learnings.get("systemic_biases", []),
            "prompt_snippets": active_learnings.get("prompt_context_snippets", []),
            "heuristic_adjustments": active_learnings.get("heuristic_adjustments", {})
        }


if __name__ == "__main__":
    retriever = RAGRetriever()
    # Sample index
    retriever.index_documents([
        {"id": "doc1", "content": "Inspired early jungle pathing in patch 14.1 favoring void grubs."},
        {"id": "doc2", "content": "Bwipo top lane counterpick metrics in LCS Spring Split."}
    ])
    context = retriever.query_context("Inspired early game jungle performance", top_k=1)
    print("Retrieved Context:")
    print("Documents:", [d["content"] for d in context["retrieved_documents"]])
    print("Prompt Snippets:", context["prompt_snippets"])
