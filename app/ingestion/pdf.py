"""Docling PDF adapter that preserves item-level page provenance."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import NormalizedChunk, NormalizedDocument


PARSER_VERSION = "docling-2"


def _label(item: Any) -> str:
    value = getattr(item, "label", "text")
    return str(getattr(value, "value", value)).lower()


def _pages(item: Any) -> tuple[int | None, int | None]:
    pages = sorted({int(prov.page_no) for prov in (getattr(item, "prov", None) or []) if getattr(prov, "page_no", None)})
    return (pages[0], pages[-1]) if pages else (None, None)


def parse(path: Path, *, converter: Any | None = None) -> NormalizedDocument:
    """Convert a local PDF; converter injection keeps provenance logic unit-testable."""
    if converter is None:
        try:
            from docling.datamodel.accelerator_options import AcceleratorDevice
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import AcceleratorOptions, PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption
        except ImportError as exc:  # pragma: no cover - deployment diagnostic
            raise RuntimeError("PDF ingestion requires the 'docling' package") from exc
        pipeline = PdfPipelineOptions(
            accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU)
        )
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline)}
        )
    document = converter.convert(path).document
    headings: list[str] = []
    chunks: list[NormalizedChunk] = []
    for item, _level in document.iterate_items():
        text = str(getattr(item, "text", "") or "").strip()
        if not text and hasattr(item, "export_to_markdown"):
            text = str(item.export_to_markdown(document)).strip()
        if not text:
            continue
        label = _label(item)
        if "title" in label or "section_header" in label:
            headings = [text]
        page_start, page_end = _pages(item)
        chunk_type = "table" if "table" in label else "figure_caption" if "caption" in label else "section"
        chunks.append(NormalizedChunk(
            text=text, chunk_type=chunk_type, heading_path=tuple(headings),
            page_start=page_start, page_end=page_end, language="text",
        ))
    title = getattr(document, "name", None) or path.stem
    return NormalizedDocument(
        source_path=path.resolve(), title=str(title), mime_type="application/pdf",
        language="text", parser_name="docling", parser_version=PARSER_VERSION,
        chunks=tuple(chunks),
    )
