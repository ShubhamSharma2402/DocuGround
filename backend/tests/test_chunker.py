"""Unit tests for the chunker: overlap, boundaries, page metadata."""

from __future__ import annotations

import pytest

from app.pipeline.chunk import TextChunk, chunk_pages, _trim_to_overlap
from app.pipeline.parse import PageContent


def make_pages(texts: list[str]) -> list[PageContent]:
    return [PageContent(page_num=i + 1, text=t, used_ocr=False) for i, t in enumerate(texts)]


class TestTrimToOverlap:
    def test_short_text_returned_unchanged(self):
        assert _trim_to_overlap("hello", 20) == "hello"

    def test_overlap_respects_word_boundary(self):
        text = "The quick brown fox jumps over the lazy dog"
        result = _trim_to_overlap(text, 15)
        for word in result.split():
            assert word in text.split()

    def test_exact_overlap_length(self):
        text = "abc defg hijkl mnop"
        result = _trim_to_overlap(text, 10)
        assert len(result) <= 10 or " " not in text[-10:]


class TestChunkPages:
    DOC_ID = "test-doc"
    USER_ID = "test-user"

    def _chunk(self, texts: list[str], size: int = 200, overlap: int = 40) -> list[TextChunk]:
        pages = make_pages(texts)
        return chunk_pages(pages, self.DOC_ID, self.USER_ID, chunk_size=size, chunk_overlap=overlap)

    def test_empty_page_produces_no_chunks(self):
        assert self._chunk([""]) == []

    def test_single_short_page_one_chunk(self):
        chunks = self._chunk(["This is a short sentence that fits in one chunk."])
        assert len(chunks) == 1
        assert chunks[0].page == 1
        assert chunks[0].doc_id == self.DOC_ID
        assert chunks[0].user_id == self.USER_ID

    def test_chunk_index_is_sequential(self):
        long_text = " ".join(["word"] * 200)
        chunks = self._chunk([long_text])
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_no_chunk_exceeds_max_size(self):
        long_text = "The quick brown fox jumps over the lazy dog. " * 30
        chunks = self._chunk([long_text], size=100, overlap=20)
        for chunk in chunks:
            assert len(chunk.text) <= 120

    def test_page_number_preserved(self):
        chunks = self._chunk(["Page one content here.", "Page two content here."])
        pages_seen = {c.page for c in chunks}
        assert 1 in pages_seen
        assert 2 in pages_seen

    def test_overlap_produces_multiple_chunks(self):
        text = "Alpha beta gamma delta epsilon. " * 10
        chunks = self._chunk([text], size=100, overlap=30)
        assert len(chunks) >= 2

    def test_overlap_smaller_than_size_required(self):
        with pytest.raises(ValueError, match="must be less than"):
            chunk_pages(make_pages(["text"]), self.DOC_ID, self.USER_ID,
                        chunk_size=50, chunk_overlap=50)

    def test_very_long_sentence_split_at_word_boundary(self):
        long_sentence = " ".join([f"word{i}" for i in range(50)])
        chunks = self._chunk([long_sentence], size=100, overlap=20)
        all_words = set(long_sentence.split())
        for chunk in chunks:
            for word in chunk.text.split():
                assert word in all_words

    def test_metadata_correct_on_all_chunks(self):
        text = " ".join(["sentence number one."] * 20)
        chunks = self._chunk([text])
        for chunk in chunks:
            assert chunk.doc_id == self.DOC_ID
            assert chunk.user_id == self.USER_ID
            assert chunk.page == 1
