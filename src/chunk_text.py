from __future__ import annotations

import json
import re
from pathlib import Path


INPUT_PATH = Path("data/processed/all_pages.json")
OUTPUT_PATH = Path("data/processed/all_chunks.json")

# Rough character-based chunking is fine for a first pass.
MAX_CHARS = 1200
OVERLAP_CHARS = 150
MIN_USEFUL_CHARS = 40


def normalize_text(text: str) -> str:
    """
    Light cleanup only. Do not over-engineer this.
    """
    text = text.replace("\u00a0", " ")
    text = text.replace("Al", "AI")  # optional small OCR fix
    text = text.replace("OpendAI", "OpenAI")
    text = text.replace("Llamalndex", "LlamaIndex")

    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]

    # Drop lines that are mostly symbols
    cleaned_lines = []
    for line in lines:
        alnum_count = sum(ch.isalnum() for ch in line)
        if alnum_count >= 3:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def is_useful_text(text: str) -> bool:
    """
    Skip pages/chunks that are too short or mostly junk.
    """
    if len(text) < MIN_USEFUL_CHARS:
        return False

    words = text.split()
    if len(words) < 5:
        return False

    return True


def split_text(text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
    """
    Simple character-based splitting with overlap.
    Tries to split on newline or sentence boundary when possible.
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + max_chars, n)

        if end < n:
            window = text[start:end]

            # Prefer splitting near the end on double newline, newline, or period.
            split_candidates = [
                window.rfind("\n\n"),
                window.rfind("\n"),
                window.rfind(". "),
            ]
            best_split = max(split_candidates)

            if best_split != -1 and best_split > max_chars // 2:
                end = start + best_split + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break

        start = max(end - overlap, start + 1)

    return chunks


def build_chunks(pages: list[dict]) -> list[dict]:
    chunks = []

    for page_item in pages:
        source = page_item["source"]
        page = page_item["page"]
        method = page_item.get("method", "unknown")
        raw_text = page_item.get("text", "")

        text = normalize_text(raw_text)

        if not is_useful_text(text):
            continue

        split_chunks = split_text(text)

        for idx, chunk_text in enumerate(split_chunks):
            if not is_useful_text(chunk_text):
                continue

            chunk_id = f"{Path(source).stem}_p{page}_c{idx}"

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "source": source,
                    "page_start": page,
                    "page_end": page,
                    "method": method,
                    "text": chunk_text,
                }
            )

    return chunks


def main() -> None:
    if not INPUT_PATH.exists():
        print(f"Input file not found: {INPUT_PATH.resolve()}")
        return

    with INPUT_PATH.open("r", encoding="utf-8") as f:
        pages = json.load(f)

    chunks = build_chunks(pages)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"Read {len(pages)} page records")
    print(f"Wrote {len(chunks)} chunks to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()