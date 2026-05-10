from src.extract_pdf import clean_text
from src.chunk_text import is_useful_text, split_text


def test_clean_text_removes_extra_spaces_and_blank_lines():
    raw = "Hello   world\n\n\nThis   is   a test.\n"
    cleaned = clean_text(raw)
    assert "Hello" in cleaned
    assert "This" in cleaned
    assert "\n\n" not in cleaned


def test_is_useful_text_rejects_too_short_text():
    assert is_useful_text("Hi") is False
    assert is_useful_text("one two three") is False


def test_is_useful_text_accepts_reasonable_text():
    text = "This is a meaningful chunk of text with enough words to be useful."
    assert is_useful_text(text) is True


def test_split_text_overlap_behavior():
    text = "A" * 1000 + "B" * 1000 + "C" * 1000
    chunks = split_text(text, max_chars=1200, overlap=100)

    assert len(chunks) >= 2

    # crude overlap sanity check:
    # end of first chunk should overlap with start of second chunk
    first_tail = chunks[0][-100:]
    second_head = chunks[1][:200]
    assert any(ch in second_head for ch in first_tail)