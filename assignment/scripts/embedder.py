import numpy as np
import torch
from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    def __init__(self, modelName=DEFAULT_EMBEDDING_MODEL, device=None):
        self.modelName = modelName
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(modelName, device=self.device)

    def embedTexts(self, texts, batchSize=32):
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Every text to embed must be a non-empty string")
        if batchSize <= 0:
            raise ValueError("batchSize must be greater than zero")
        embeddings = self.model.encode(
            texts,
            batch_size=batchSize,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise ValueError("Embedding model returned an invalid embedding matrix")
        if not np.isfinite(vectors).all():
            raise ValueError("Embedding model returned non-finite values")
        return vectors

    def embedChunks(self, chunks, batchSize=32):
        if not chunks:
            return []
        texts = [chunk.get("text") for chunk in chunks]
        embeddings = self.embedTexts(texts, batchSize=batchSize)
        return [
            {**chunk, "embedding": embedding.tolist()}
            for chunk, embedding in zip(chunks, embeddings)
        ]

