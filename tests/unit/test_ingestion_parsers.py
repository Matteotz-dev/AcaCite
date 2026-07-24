from enum import Enum
from pathlib import Path
from types import SimpleNamespace

from app.ingestion.code import parse as parse_code
from app.ingestion.markdown import parse as parse_markdown
from app.ingestion.pdf import parse as parse_pdf
from app.ingestion.plaintext import parse as parse_text


def test_plaintext_is_stable_and_retains_lines(tmp_path: Path):
    path = tmp_path / "note.txt"
    path.write_text("First paragraph.\nline two\n\nSecond.\n", encoding="utf-8")
    first = parse_text(path)
    second = parse_text(path)
    assert first == second
    assert [(c.line_start, c.line_end) for c in first.chunks] == [(1, 2), (4, 4)]


def test_markdown_retains_heading_paths_and_lines(tmp_path: Path):
    path = tmp_path / "paper.md"
    path.write_text("# Title\nintro\n\n## Method\nclosure\n", encoding="utf-8")
    document = parse_markdown(path)
    assert document.title == "Title"
    assert document.chunks[1].heading_path == ("Title", "Method")
    assert (document.chunks[1].line_start, document.chunks[1].line_end) == (4, 5)


def test_python_symbols_have_exact_line_spans(tmp_path: Path):
    path = tmp_path / "solver.py"
    path.write_text("def one():\n    return 1\n\ndef two():\n    return 2\n", encoding="utf-8")
    chunks = parse_code(path).chunks
    assert [(c.symbol, c.line_start, c.line_end) for c in chunks] == [
        ("one", 1, 3), ("two", 4, 5)
    ]


def test_openfoam_subdictionary_has_exact_span(tmp_path: Path):
    path = tmp_path / "momentumTransport"
    path.write_text("simulationType LES;\nLES\n{\n    model NGM;\n}\n", encoding="utf-8")
    chunk = parse_code(path).chunks[0]
    assert (chunk.symbol, chunk.line_start, chunk.line_end) == ("LES", 2, 5)


class Label(Enum):
    SECTION_HEADER = "section_header"
    TEXT = "text"


def test_docling_adapter_preserves_page_numbers(tmp_path: Path):
    path = tmp_path / "fixture.pdf"
    path.write_bytes(b"%PDF-fixture")
    items = [
        SimpleNamespace(text="Methods", label=Label.SECTION_HEADER, prov=[SimpleNamespace(page_no=2)]),
        SimpleNamespace(text="A closure.", label=Label.TEXT, prov=[SimpleNamespace(page_no=2)]),
    ]
    document = SimpleNamespace(name="Fixture", iterate_items=lambda: iter((item, 0) for item in items))
    converter = SimpleNamespace(convert=lambda _path: SimpleNamespace(document=document))
    parsed = parse_pdf(path, converter=converter)
    assert parsed == parse_pdf(path, converter=converter)
    assert parsed.chunks[1].heading_path == ("Methods",)
    assert (parsed.chunks[1].page_start, parsed.chunks[1].page_end) == (2, 2)
