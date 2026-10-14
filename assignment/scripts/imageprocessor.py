import os
from PIL import Image
import pytesseract
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

class ImageProcessor:
    def __init__(self, captionModelName="Salesforce/blip-image-captioning-base", device=None):
        """
        Initialize the image captioning and OCR modules.
        """
        print("🔧 Loading image captioning model (this may take a minute)...")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = BlipProcessor.from_pretrained(captionModelName)
        self.model = BlipForConditionalGeneration.from_pretrained(captionModelName).to(self.device)
        print(f" Model loaded on {self.device}")

    def summarizeImage(self, imagePath):
        """
        Generate a combined summary for an image using:
          1. OCR (if any text is present)
          2. Image captioning
        """
        try:
            image = Image.open(imagePath).convert("RGB")

            # 1️⃣ OCR text extraction
            ocrText = pytesseract.image_to_string(image).strip()
            ocrText = " ".join(ocrText.split())  # clean up whitespace

            # 2️⃣ Image captioning
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputIds = self.model.generate(**inputs, max_new_tokens=40)
            caption = self.processor.decode(outputIds[0], skip_special_tokens=True)

            # 3️⃣ Combine
            if ocrText:
                summary = f"Image shows: {caption}. Visible text reads: '{ocrText}'."
            else:
                summary = f"Image shows: {caption}."

            return summary

        except Exception as e:
            print(f" Failed to process image {imagePath}: {e}")
            return f"[Error summarizing image: {os.path.basename(imagePath)}]"

    def processImageFolder(self, imagesDir):
        """
        Process all images in a directory and return a dictionary mapping:
            {image_path: summary}
        """
        summaries = {}
        imageFiles = [f for f in os.listdir(imagesDir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not imageFiles:
            print(f"No images found in {imagesDir}")
            return summaries

        for imgFile in imageFiles:
            imgPath = os.path.join(imagesDir, imgFile)
            summary = self.summarizeImage(imgPath)
            summaries[imgPath] = summary
            print(f" {imgFile} → {summary}")

        print(f"Processed {len(summaries)} images.")
        return summaries


if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Summarize images with OCR + captioning")
    parser.add_argument("--imagesDir", type=str, required=True, help="Path to folder containing images")
    parser.add_argument("--outFile", type=str, default="data/processed/image_summaries.json", help="Path to output JSON")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.outFile), exist_ok=True)
    processor = ImageProcessor()
    results = processor.processImageFolder(args.imagesDir)

    with open(args.outFile, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"📝 Image summaries saved to {args.outFile}")
