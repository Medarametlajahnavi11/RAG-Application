import logging
from pathlib import Path

import pdfplumber
from pdfplumber.utils.exceptions import PdfminerException


logger = logging.getLogger(__name__)


class PDFLoader:
    def extractContent(
        self, pdfPath, outputDir="data/processed", extractImages=True
    ):
        """Extract page text and save any renderable embedded images.

        Each result keeps its source filename and one-based PDF page number.
        Image files are preserved for the existing image-processing workflow;
        they are not represented as extracted text or sent to the RAG index.
        """
        path = Path(pdfPath)
        if not path.is_file():
            raise FileNotFoundError(f"PDF file not found: {path}")

        imagesDir = Path(outputDir) / "images"
        imageCounter = 0
        pages = []

        try:
            with pdfplumber.open(path) as pdf:
                for pageNum, page in enumerate(pdf.pages, start=1):
                    text = (page.extract_text() or "").strip()
                    if extractImages:
                        for imgIndex, image in enumerate(page.images, start=1):
                            imageName = (
                                f"{path.stem}_page{pageNum}_img{imgIndex}.jpg"
                            )
                            imagePath = imagesDir / imageName
                            try:
                                imagesDir.mkdir(parents=True, exist_ok=True)
                                bbox = (
                                    image["x0"],
                                    image["top"],
                                    image["x1"],
                                    image["bottom"],
                                )
                                rendered = page.within_bbox(bbox).to_image(
                                    resolution=150
                                )
                                rendered.original.convert("RGB").save(
                                    imagePath, format="JPEG"
                                )
                                imageCounter += 1
                            except (
                                KeyError,
                                OSError,
                                ValueError,
                                AttributeError,
                                TypeError,
                                RuntimeError,
                                PdfminerException,
                            ) as exc:
                                logger.warning(
                                    "Could not extract image %s on page %s of %s: %s",
                                    imgIndex,
                                    pageNum,
                                    path.name,
                                    exc,
                                )
                    pages.append(
                        {"source": path.name, "page": pageNum, "text": text}
                    )
        except (PdfminerException, OSError, ValueError) as exc:
            raise ValueError(f"Could not read PDF '{path.name}': {exc}") from exc
        logger.info(
            "Extracted %s pages and saved %s images from %s",
            len(pages),
            imageCounter,
            path.name,
        )
        return pages


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract text from a PDF")
    parser.add_argument("--pdf", required=True, help="Path to PDF")
    parser.add_argument("--out", default="data/processed", help="Image output directory")
    args = parser.parse_args()

    extractedPages = PDFLoader().extractContent(args.pdf, args.out)
    outputPath = Path(args.out) / f"{Path(args.pdf).stem}_text.json"
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    import json

    outputPath.write_text(
        json.dumps(extractedPages, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Extracted {len(extractedPages)} pages to {outputPath}")
