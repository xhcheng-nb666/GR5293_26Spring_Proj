from __future__ import annotations

import json
import re
from pathlib import Path

import fitz  # PyMuPDF
import pytesseract
from PIL import Image


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
DEBUG_IMG_DIR = PROCESSED_DIR / "ocr_debug_images"

# Adjust this threshold as needed.
# If native extracted text is shorter than this, we fall back to OCR.
MIN_TEXT_LENGTH = 30


def clean_text(text: str) -> str:
    """
    Basic cleanup for extracted text.
    Keeps line breaks but removes excess blank lines and extra spaces.
    """
    text = text.replace("\u00a0", " ")
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def native_extract(page: fitz.Page) -> str:
    """
    Try native PDF text extraction.
    """
    text = page.get_text("text")
    return clean_text(text)


def render_page_to_image(page: fitz.Page, zoom: float = 2.0) -> Image.Image:
    """
    Render a PDF page to a PIL Image for OCR.
    zoom=2.0 usually gives decent OCR quality.
    """
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)

    mode = "RGB" if pix.n < 4 else "RGBA"
    img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
    return img


def ocr_extract(page: fitz.Page, debug_image_path: Path | None = None) -> str:
    """
    OCR fallback: render page to image, then run Tesseract OCR.
    Optionally save the rendered image for debugging.
    """
    image = render_page_to_image(page, zoom=2.5)

    if debug_image_path is not None:
        debug_image_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(debug_image_path)

    text = pytesseract.image_to_string(image)
    return clean_text(text)


def extract_page(page: fitz.Page, source_name: str, page_number: int) -> dict:
    """
    Extract a single page using native extraction first, OCR fallback second.
    """
    native_text = native_extract(page)

    if len(native_text.strip()) >= MIN_TEXT_LENGTH:
        return {
            "source": source_name,
            "page": page_number,
            "text": native_text,
            "method": "native",
            "native_text_length": len(native_text),
        }

    debug_img_path = DEBUG_IMG_DIR / f"{Path(source_name).stem}_page_{page_number}.png"
    ocr_text = ocr_extract(page, debug_img_path)

    return {
        "source": source_name,
        "page": page_number,
        "text": ocr_text,
        "method": "ocr",
        "native_text_length": len(native_text),
    }


def save_pdf_txt(pdf_name: str, pages: list[dict], output_dir: Path) -> None:
    """
    Save one .txt file per PDF, preserving page boundaries and extraction method.
    """
    output_path = output_dir / f"{Path(pdf_name).stem}.txt"
    with output_path.open("w", encoding="utf-8") as f:
        for item in pages:
            f.write(f"=== Page {item['page']} | method={item['method']} ===\n")
            f.write(item["text"])
            f.write("\n\n")


def extract_single_pdf(pdf_path: Path) -> list[dict]:
    """
    Extract all pages from a single PDF.
    """
    pages = []
    with fitz.open(pdf_path) as doc:
        for idx, page in enumerate(doc):
            page_number = idx + 1
            print(f"Processing {pdf_path.name} | page {page_number}...")
            page_result = extract_page(page, pdf_path.name, page_number)
            pages.append(page_result)
    return pages


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_IMG_DIR.mkdir(parents=True, exist_ok=True)

    pdf_paths = sorted(RAW_DIR.glob("*.pdf"))

    if not pdf_paths:
        print(f"No PDF files found in: {RAW_DIR.resolve()}")
        return

    all_pages = []

    for pdf_path in pdf_paths:
        print(f"\n{'=' * 60}")
        print(f"Processing {pdf_path.name}...")
        print(f"{'=' * 60}")

        pages = extract_single_pdf(pdf_path)
        all_pages.extend(pages)

        save_pdf_txt(pdf_path.name, pages, PROCESSED_DIR)

        json_path = PROCESSED_DIR / f"{pdf_path.stem}_pages.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(pages, f, ensure_ascii=False, indent=2)

        n_native = sum(1 for p in pages if p["method"] == "native")
        n_ocr = sum(1 for p in pages if p["method"] == "ocr")

        print(f"\nFinished {pdf_path.name}")
        print(f"Total pages: {len(pages)}")
        print(f"Native extraction pages: {n_native}")
        print(f"OCR fallback pages: {n_ocr}")
        print(f"Saved JSON to: {json_path}")
        print(f"Saved TXT to: {PROCESSED_DIR / f'{pdf_path.stem}.txt'}")

    combined_json_path = PROCESSED_DIR / "all_pages.json"
    with combined_json_path.open("w", encoding="utf-8") as f:
        json.dump(all_pages, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print("All PDFs processed.")
    print(f"Total PDFs: {len(pdf_paths)}")
    print(f"Total extracted pages: {len(all_pages)}")
    print(f"Combined JSON saved to: {combined_json_path}")
    print(f"OCR debug images saved under: {DEBUG_IMG_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()