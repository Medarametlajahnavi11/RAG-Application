import faiss
import numpy as np

from scripts.embedder import DEFAULT_EMBEDDING_MODEL, Embedder
from scripts.indexer import Indexer


class Retriever:
    def __init__(
        self,
        indexDir="index",
        modelName=DEFAULT_EMBEDDING_MODEL,
        device=None,
        minimumScore=0.25,
        embedder=None,
    ):
        if minimumScore is not None and not -1.0 <= minimumScore <= 1.0:
            raise ValueError("minimumScore must be between -1 and 1")
        self.modelName = modelName
        self.minimumScore = minimumScore
        self.index, self.metadata = Indexer(indexDir).loadIndex(modelName=modelName)
        if self.index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise ValueError("Index does not use normalized cosine similarity.")
        self.embedder = embedder or Embedder(modelName=modelName, device=device)
        if self.index.d <= 0:
            raise ValueError("Index has an invalid embedding dimension.")

    def embedQuery(self, query):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string")
        vectors = self.embedder.embedTexts([query.strip()])
        if vectors.shape != (1, self.index.d):
            raise ValueError(
                "Query embedding dimensions do not match the index; rebuild it."
            )
        if not np.isfinite(vectors).all():
            raise ValueError("Query embedding contains non-finite values")
        norm = np.linalg.norm(vectors[0])
        if norm == 0:
            raise ValueError("Query embedding has zero magnitude")
        return (vectors / norm).astype(np.float32, copy=False)

    def search(self, query, topK=5):
        if not isinstance(topK, int) or topK <= 0:
            raise ValueError("topK must be a positive integer")
        queryVector = self.embedQuery(query)
        scores, indices = self.index.search(queryVector, min(topK, self.index.ntotal))
        results = []
        for score, indexPosition in zip(scores[0], indices[0]):
            if indexPosition < 0:
                continue
            if indexPosition >= len(self.metadata):
                raise ValueError("FAISS returned an invalid metadata index.")
            score = float(score)
            if self.minimumScore is not None and score < self.minimumScore:
                continue
            record = self.metadata[indexPosition]
            results.append(
                {
                    "source": record["source"],
                    "page": record["page"],
                    "text": record["text"],
                    "score": score,
                }
            )
        return results
