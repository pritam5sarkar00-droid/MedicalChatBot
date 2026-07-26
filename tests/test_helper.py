from langchain_core.documents import Document

from src.helper import filter_to_minimal_docs


def test_filter_to_minimal_docs_preserves_source_page_and_page_label():
    """This is a regression test: filter_to_minimal_docs used to keep only
    'source' in metadata, silently dropping 'page'. Since app.py's
    extract_sources() builds every citation chip from these fields, that
    bug meant citations would show a filename but never a page number, for
    every single answer, with no error or test failure anywhere -- it only
    would have shown up by actually reading a live PDF end-to-end. This
    test exists so that gap can never come back unnoticed.

    page_label is kept alongside the raw page index for the same reason:
    dropping it here would silently make extract_sources() fall back to
    raw sequential page numbers for every citation, even for PDFs whose
    embedded page labels diverge from that (front matter, unnumbered
    cover pages, etc.) -- see filter_to_minimal_docs' own docstring."""
    docs = [
        Document(
            page_content="Asthma is a chronic airway condition...",
            metadata={
                "source": "data/seed/diabetes.pdf",
                "page": 22,
                "page_label": "23",
                "producer": "some noisy PDF library metadata",
                "creationdate": "2004-12-18",
            },
        )
    ]

    result = filter_to_minimal_docs(docs)

    assert len(result) == 1
    assert result[0].page_content == docs[0].page_content
    assert result[0].metadata == {"source": "data/seed/diabetes.pdf", "page": 22, "page_label": "23"}


def test_filter_to_minimal_docs_handles_missing_page():
    # Some loaders/document types genuinely have no page number -- should
    # degrade to None, not raise.
    docs = [Document(page_content="text", metadata={"source": "book.pdf"})]
    result = filter_to_minimal_docs(docs)
    assert result[0].metadata == {"source": "book.pdf", "page": None, "page_label": None}


def test_filter_to_minimal_docs_handles_empty_list():
    assert filter_to_minimal_docs([]) == []
