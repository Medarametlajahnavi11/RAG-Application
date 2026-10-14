import json
import re


class Chunker:
    def __init__(self, chunkSize=700, chunkOverlap=100):
        if chunkSize <= 0:
            raise ValueError("chunkSize must be greater than zero")
        if chunkOverlap < 0 or chunkOverlap >= chunkSize:
            raise ValueError("chunkOverlap must be between 0 and chunkSize - 1")
        self.chunkSize = chunkSize
        self.chunkOverlap = chunkOverlap

    def _chunkText(self, text):
        if not isinstance(text, str) or not text.strip():
            return []

        normalized = re.sub(r"\s+", " ", text).strip()
        chunks = []
        start = 0
        while start < len(normalized):
            end = min(start + self.chunkSize, len(normalized))
            if end < len(normalized):
                boundary = normalized.rfind(" ", start, end)
                if boundary > start:
                    end = boundary
            chunk = normalized[start:end].strip()
            if chunk and (not chunks or chunk != chunks[-1]):
                chunks.append(chunk)
            if end >= len(normalized):
                break
            nextStart = max(start + 1, end - self.chunkOverlap)
            while nextStart < len(normalized) and normalized[nextStart].isspace():
                nextStart += 1
            start = nextStart
        return chunks

    def createChunks(self, pagesText):
        """Chunk page strings or page records without crossing page boundaries."""
        allChunks = []
        for pageIndex, pageRecord in enumerate(pagesText, start=1):
            if isinstance(pageRecord, dict):
                pageText = pageRecord.get("text", "")
                source = pageRecord.get("source", "unknown.pdf")
                pageNumber = pageRecord.get("page", pageIndex)
            else:
                pageText = pageRecord
                source = "unknown.pdf"
                pageNumber = pageIndex
            for chunkIndex, text in enumerate(self._chunkText(pageText), start=1):
                allChunks.append(
                    {
                        "source": str(source),
                        "page": int(pageNumber),
                        "chunk": chunkIndex,
                        "text": text,
                    }
                )
        return allChunks


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Chunk extracted PDF text")
    parser.add_argument("--inputFile", required=True)
    parser.add_argument("--chunkSize", type=int, default=700)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--outputFile", default="data/processed/chunks.json")
    args = parser.parse_args()

    loaded = json.loads(Path(args.inputFile).read_text(encoding="utf-8"))
    pages = loaded if isinstance(loaded, list) else [
        {"source": "unknown.pdf", "page": 1, "text": loaded}
    ]
    chunks = Chunker(args.chunkSize, args.overlap).createChunks(pages)
    output = Path(args.outputFile)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Chunked {len(chunks)} segments to {output}")
