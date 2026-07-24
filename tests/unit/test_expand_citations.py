from __future__ import annotations

import sqlite3

from scripts.expand_citations import Work, extract_seed_references, looks_like_reference, safe_pdf_name


def test_looks_like_reference_accepts_bibliographic_text():
    assert looks_like_reference(
        "Clark, R. A., Ferziger, J. H. & Reynolds, W. C. 1979 Evaluation of subgrid-scale models."
    )


def test_safe_pdf_name_uses_doi_and_sanitized_title():
    name = safe_pdf_name(Work(
        key="doi:10.1234/example",
        depth=1,
        doi="10.1234/example",
        title="A useful paper: closures / filters?",
    ))
    assert name.startswith("10.1234_example-")
    assert name.endswith(".pdf")
    assert "/" not in name


def test_extract_seed_references_from_reference_heading(tmp_path):
    database = tmp_path / "provenance.sqlite3"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE documents (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            current_version_id TEXT,
            deleted_at TEXT
        );
        CREATE TABLE document_versions (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL
        );
        CREATE TABLE chunks (
            document_version_id TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            heading_path_json TEXT,
            text TEXT NOT NULL
        );
        """
    )
    connection.execute(
        "INSERT INTO documents VALUES (?, ?, ?, ?)",
        ("doc-1", "Source Paper", "version-1", None),
    )
    connection.execute("INSERT INTO document_versions VALUES (?, ?)", ("version-1", "doc-1"))
    connection.execute(
        "INSERT INTO chunks VALUES (?, ?, ?, ?)",
        (
            "version-1",
            10,
            '["References"]',
            "Clark, R. A., Ferziger, J. H. & Reynolds, W. C. 1979 Evaluation of subgrid-scale models.",
        ),
    )
    connection.commit()

    works = extract_seed_references(database, ["Source Paper"])

    assert len(works) == 1
    assert works[0].depth == 1
    assert works[0].cited_by == ["Source Paper"]
