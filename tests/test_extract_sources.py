"""
Tests for app.py's extract_sources() — in particular a real reported bug:
a source path recorded with backslashes (e.g. "data\\report.pdf", left
over from whichever machine originally built the Pinecone index) showed
up as a second, differently-labeled citation for the same file already
shown under its clean name ("report.pdf"), because os.path.basename()
only respects the separator of whatever OS the code happens to be
running on -- a Linux server doesn't treat "\\" as a path separator at
all.
"""

from app import extract_sources


class FakeDoc:
    def __init__(self, source=None, page=None, page_label=None, retrieval_score=None):
        self.metadata = {"source": source, "page": page}
        if page_label is not None:
            self.metadata["page_label"] = page_label
        if retrieval_score is not None:
            self.metadata["retrieval_score"] = retrieval_score


def test_windows_and_posix_style_paths_for_the_same_file_collapse_to_one_source():
    docs = [
        FakeDoc(source="data\\reference.pdf", page=127),
        FakeDoc(source="reference.pdf", page=127),
        FakeDoc(source="data/reference.pdf", page=127),
    ]
    result = extract_sources(docs)
    assert len(result) == 1
    assert result[0]["source"] == "reference.pdf"
    assert result[0]["page"] == "128"


def test_mixed_separators_in_one_path_still_resolve_to_the_final_segment():
    docs = [FakeDoc(source="C:\\Users\\pritam\\data/subfolder\\reference.pdf", page=0)]
    result = extract_sources(docs)
    assert result[0]["source"] == "reference.pdf"


def test_plain_filename_with_no_path_separators_is_unaffected():
    docs = [FakeDoc(source="report.pdf", page=2)]
    result = extract_sources(docs)
    assert result[0]["source"] == "report.pdf"


def test_missing_source_falls_back_to_a_generic_label():
    docs = [FakeDoc(source=None, page=0)]
    result = extract_sources(docs)
    assert result[0]["source"] == "the knowledge base"


def test_empty_context_returns_empty_list():
    assert extract_sources([]) == []
    assert extract_sources(None) == []


def test_page_label_preferred_over_raw_zero_indexed_page():
    docs = [FakeDoc(source="book.pdf", page=5, page_label="xii")]
    result = extract_sources(docs)
    assert result[0]["page"] == "xii"


def test_raw_page_used_and_incremented_when_no_page_label():
    docs = [FakeDoc(source="book.pdf", page=5)]
    result = extract_sources(docs)
    # A string, like every other page value extract_sources() returns
    # (see resolve_page_display() in src/helper.py) -- not an int -- so
    # that a citation's page and whatever src/pipeline.py's
    # document_prompt told the LLM can never quietly disagree in type as
    # well as value.
    assert result[0]["page"] == "6"  # PyPDFLoader pages are 0-indexed


def test_retrieval_score_is_preserved_per_source():
    docs = [FakeDoc(source="my_notes.pdf", page=0, retrieval_score=0.83)]
    result = extract_sources(docs)
    assert result[0]["score"] == 0.83


def test_missing_retrieval_score_is_none_not_a_crash():
    """A cached answer's sources never had retrieval re-run, so they have
    no score at all -- extract_sources() must handle that gracefully
    rather than assuming the key exists (it isn't called on the cached
    path, but the underlying doc shape it's built to tolerate is the
    same one a cache-hit source list is built from -- see app.py)."""
    docs = [FakeDoc(source="my_notes.pdf", page=0)]
    result = extract_sources(docs)
    assert result[0]["score"] is None


def test_same_file_different_pages_are_kept_as_separate_sources():
    docs = [FakeDoc(source="book.pdf", page=1), FakeDoc(source="book.pdf", page=2)]
    result = extract_sources(docs)
    assert len(result) == 2


def test_same_file_and_page_collapse_to_one_source_even_with_different_scores():
    """The dedup key is (label, page) alone now that there's no 'uploaded'
    flag to also key on (every document in the knowledge base is indexed
    and cited identically regardless of whether it arrived via data/seed/
    or an upload -- see src/documents.py's module docstring) -- two
    chunks of the same page, e.g. from reranking plus a chunk-boundary
    split, must still show up as one citation, not two."""
    docs = [
        FakeDoc(source="notes.pdf", page=2, retrieval_score=0.91),
        FakeDoc(source="notes.pdf", page=2, retrieval_score=0.64),
    ]
    result = extract_sources(docs)
    assert len(result) == 1
