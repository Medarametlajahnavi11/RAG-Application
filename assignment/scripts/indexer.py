import argparse
import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path

import faiss
import numpy as np

from scripts.chunker import Chunker
from scripts.embedder import DEFAULT_EMBEDDING_MODEL, Embedder
from scripts.pdfloader import PDFLoader


logger = logging.getLogger(__name__)


class Indexer:
    def __init__(self, indexDir="index"):
        self.indexDir = Path(indexDir)
        self.indexDir.mkdir(parents=True, exist_ok=True)

    def buildIndex(self, embeddedChunks, modelName=DEFAULT_EMBEDDING_MODEL):
        if not embeddedChunks:
            raise ValueError("No chunks provided for indexing")

        records = []
        vectors = []
        for position, chunk in enumerate(embeddedChunks):
            embedding = np.asarray(chunk.get("embedding"), dtype=np.float32)
            if embedding.ndim != 1 or not embedding.size:
                raise ValueError(f"Chunk {position} has an invalid embedding")
            if not np.isfinite(embedding).all():
                raise ValueError(f"Chunk {position} has non-finite embedding values")
            if not chunk.get("source") or not chunk.get("text"):
                raise ValueError(
                    f"Chunk {position} must include source filename and text"
                )
            if chunk.get("page") is None:
                raise ValueError(f"Chunk {position} is missing its page number")
            if vectors and embedding.size != vectors[0].size:
                raise ValueError("All chunk embeddings must have the same dimension")
            vectors.append(embedding)
            records.append(
                {
                    "source": str(chunk["source"]),
                    "page": int(chunk["page"]),
                    "chunk": int(chunk.get("chunk", position + 1)),
                    "text": str(chunk["text"]),
                }
            )

        matrix = np.vstack(vectors).astype(np.float32, copy=False)
        norms = np.linalg.norm(matrix, axis=1)
        if np.any(norms == 0):
            raise ValueError("Chunk embeddings must have non-zero magnitude")
        matrix /= norms[:, None]
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        metadata = {
            "embedding_model": modelName,
            "embedding_dimension": int(matrix.shape[1]),
            "chunks": records,
        }
        self._writeAtomically(index, metadata)
        logger.info(
            "Built FAISS index: %s chunks, %s dimensions, model %s",
            len(records),
            matrix.shape[1],
            modelName,
        )
        return index

    def _writeAtomically(self, index, metadata):
        self.indexDir.mkdir(parents=True, exist_ok=True)
        indexTemp = None
        metadataTemp = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=self.indexDir, suffix=".index.tmp", delete=False
            ) as temp:
                indexTemp = temp.name
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.indexDir,
                suffix=".json.tmp",
                delete=False,
            ) as temp:
                metadataTemp = temp.name
            faiss.write_index(index, indexTemp)
            with open(indexTemp, "rb") as indexFile:
                metadata["index_sha256"] = hashlib.sha256(
                    indexFile.read()
                ).hexdigest()
            with open(metadataTemp, "w", encoding="utf-8") as temp:
                json.dump(metadata, temp, ensure_ascii=False, indent=2)
                temp.flush()
                os.fsync(temp.fileno())
            os.replace(metadataTemp, self.indexDir / "metadata.json")
            metadataTemp = None
            os.replace(indexTemp, self.indexDir / "faiss.index")
            indexTemp = None
        finally:
            for tempPath in (indexTemp, metadataTemp):
                if tempPath and os.path.exists(tempPath):
                    os.unlink(tempPath)

    def loadIndex(self, modelName=DEFAULT_EMBEDDING_MODEL):
        indexPath = self.indexDir / "faiss.index"
        metadataPath = self.indexDir / "metadata.json"
        if not indexPath.is_file() or not metadataPath.is_file():
            raise FileNotFoundError(
                "FAISS index or metadata is missing. Rebuild the knowledge base."
            )
        try:
            index = faiss.read_index(str(indexPath))
            metadata = json.loads(metadataPath.read_text(encoding="utf-8"))
        except (RuntimeError, OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Knowledge-base index is unreadable: {exc}") from exc
        if not isinstance(metadata, dict) or not isinstance(
            metadata.get("chunks"), list
        ):
            raise ValueError("Index metadata is stale or incompatible; rebuild it.")
        with indexPath.open("rb") as indexFile:
            indexHash = hashlib.sha256(indexFile.read()).hexdigest()
        if metadata.get("index_sha256") != indexHash:
            raise ValueError("Index and metadata do not match; rebuild the index.")
        records = metadata["chunks"]
        if index.ntotal <= 0 or index.ntotal != len(records):
            raise ValueError("Index and metadata are misaligned; rebuild the index.")
        if metadata.get("embedding_model") != modelName:
            raise ValueError(
                "Index uses a different embedding model; rebuild the knowledge base."
            )
        if metadata.get("embedding_dimension") != index.d:
            raise ValueError("Index metadata has the wrong embedding dimension.")
        for position, record in enumerate(records):
            if (
                not isinstance(record, dict)
                or not record.get("source")
                or not record.get("text")
                or record.get("page") is None
            ):
                raise ValueError(
                    f"Index metadata record {position} is invalid; rebuild the index."
                )
        return index, records


def build_knowledge_base(
    pdfDir="data/pdfs",
    indexDir="index",
    chunkSize=700,
    chunkOverlap=100,
    modelName=DEFAULT_EMBEDDING_MODEL,
):
    pdfRoot = Path(pdfDir)
    pdfPaths = sorted(path for path in pdfRoot.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf")
    if not pdfPaths:
        raise ValueError(f"No PDF files found in {pdfRoot}")

    loader = PDFLoader()
    chunker = Chunker(chunkSize=chunkSize, chunkOverlap=chunkOverlap)
    allChunks = []
    pageCount = 0
    for pdfPath in pdfPaths:
        pages = loader.extractContent(pdfPath, extractImages=False)
        pageCount += len(pages)
        source = pdfPath.relative_to(pdfRoot).as_posix()
        for page in pages:
            page["source"] = source
        allChunks.extend(chunker.createChunks(pages))

    if not allChunks:
        raise ValueError(
            "No extractable PDF text was found. Image-only/scanned PDFs need OCR."
        )
    embedder = Embedder(modelName=modelName)
    embeddedChunks = embedder.embedChunks(allChunks)
    index = Indexer(indexDir).buildIndex(embeddedChunks, modelName=modelName)
    summary = {
        "documents": len(pdfPaths),
        "pages": pageCount,
        "chunks": len(embeddedChunks),
        "vectors": index.ntotal,
    }
    logger.info(
        "Knowledge base rebuilt from %s PDFs, %s pages, %s chunks",
        summary["documents"],
        summary["pages"],
        summary["chunks"],
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a PDF knowledge base")
    parser.add_argument("--pdfDir", default="data/pdfs")
    parser.add_argument("--indexDir", default="index")
    parser.add_argument("--chunkSize", type=int, default=700)
    parser.add_argument("--overlap", type=int, default=100)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    result = build_knowledge_base(
        pdfDir=args.pdfDir,
        indexDir=args.indexDir,
        chunkSize=args.chunkSize,
        chunkOverlap=args.overlap,
    )
    print(json.dumps(result, indent=2))
