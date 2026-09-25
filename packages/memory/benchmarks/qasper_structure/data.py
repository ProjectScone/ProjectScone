"""Deterministic, annotation-free export of the complete official QASPER test set."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator

from scone_memory.retrieval.section_routing import SectionSnapshot

EXPECTED_COUNTS = (416, 1451)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")


class Paragraph(_Frozen):
    """An unchanged source paragraph and its half-open UTF-8 byte span."""

    text: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @model_validator(mode="after")
    def check_span(self) -> Self:
        if self.end - self.start != len(self.text.encode("utf-8")):
            raise ValueError("paragraph span does not match its UTF-8 length")
        return self


class Paper(_Frozen):
    id: str = Field(min_length=1)
    content: str
    paragraphs: tuple[Paragraph, ...]

    @model_validator(mode="after")
    def check_paragraphs(self) -> Self:
        raw = self.content.encode("utf-8")
        previous_end = 0
        for paragraph in self.paragraphs:
            if paragraph.start < previous_end or raw[paragraph.start:paragraph.end] != paragraph.text.encode("utf-8"):
                raise ValueError("paragraph spans must be ordered, nonoverlapping, and exact")
            previous_end = paragraph.end
        return self


class Question(_Frozen):
    id: str = Field(min_length=1)
    paper_id: str = Field(min_length=1)
    question: str = Field(min_length=1)


class _RawSection(_Frozen):
    section_name: str | None
    paragraphs: list[str]


class _RawFigure(_Frozen):
    file: str
    caption: str


class _RawQuestion(BaseModel):
    # Deliberately discard ALL annotation and annotator fields at the boundary.
    model_config = ConfigDict(frozen=True, strict=True, extra="ignore")
    question_id: str
    question: str

    @field_validator("question_id", "question")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question identity and text must be nonblank")
        return value


class _RawPaper(_Frozen):
    title: str
    abstract: str
    full_text: list[_RawSection]
    qas: list[_RawQuestion]
    figures_and_tables: list[_RawFigure]


def _heading(value: str | None, fallback: str) -> str:
    normalized = " ".join((value or "").split()).rstrip("#").strip()
    return fallback if normalized.casefold() in ("", "unknown", "[unknown]", "null") else normalized


def render_paper(paper_id: str, source: _RawPaper) -> Paper:
    """Render source-order sections; absent ancestors contain headings only."""
    chunks: list[str] = []
    paragraphs: list[Paragraph] = []
    expected_paths: list[tuple[str, ...]] = []
    byte_offset = 0
    title = _heading(source.title, "Untitled paper")

    def append(text: str) -> None:
        nonlocal byte_offset
        chunks.append(text)
        byte_offset += len(text.encode("utf-8"))

    def heading(path: tuple[str, ...]) -> None:
        append(f"{'#' * len(path)} {path[-1]}\n\n")
        expected_paths.append(path)

    def paragraph(text: str) -> None:
        start = byte_offset
        append(text)
        paragraphs.append(Paragraph(text=text, start=start, end=byte_offset))
        append("\n\n")

    heading((title,))
    heading((title, "Abstract"))
    paragraph(source.abstract)
    active: tuple[str, ...] = ()
    for index, section in enumerate(source.full_text, 1):
        components = (section.section_name or "").split(" ::: ")
        if len(components) > 5:
            raise ValueError("section path exceeds Markdown's six heading levels")
        path = tuple(_heading(part, f"Section {index}.{depth}")
                     for depth, part in enumerate(components, 1))
        common = 0
        while common < min(len(active), len(path)) and active[common] == path[common]:
            common += 1
        # A parent occurring after its child needs a new heading to close the child.
        if common == len(path) and active != path:
            common -= 1
        for depth in range(common, len(path)):
            heading((title, *path[:depth + 1]))
        for text in section.paragraphs:
            paragraph(text)
        active = path
    if source.figures_and_tables:
        heading((title, "Figure and table captions"))
        for figure in source.figures_and_tables:
            paragraph(figure.caption)
    paper = Paper(id=paper_id, content="".join(chunks), paragraphs=tuple(paragraphs))
    snapshot = SectionSnapshot.from_markdown("qasper-test", paper.id, paper.content)
    actual_paths = [snapshot.path(node.id) for node in snapshot.nodes if node.title]
    if actual_paths != expected_paths:
        raise ValueError(f"rendered outline differs from source hierarchy for {paper_id}")
    return paper


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def export(raw_path: Path, output: Path) -> None:
    """Validate the full official test split and write a reproducible inference bundle.

    Annotation values never enter Paper or Question. The source file's SHA-256
    binds the separate gold data for later scoring without exposing it here.
    """
    raw_bytes = raw_path.read_bytes()
    decoded: object = json.loads(raw_bytes, object_pairs_hook=_unique_object)
    source = TypeAdapter(dict[str, _RawPaper]).validate_python(decoded, strict=True)
    if any(not key.strip() or len(key) > 512 for key in source):
        raise ValueError("paper identities must be nonblank and bounded")
    papers: list[Paper] = []
    questions: list[Question] = []
    seen_questions: set[str] = set()
    for paper_id in sorted(source):
        paper = source[paper_id]
        papers.append(render_paper(paper_id, paper))
        for question in sorted(paper.qas, key=lambda item: item.question_id):
            if question.question_id in seen_questions:
                raise ValueError(f"duplicate question identity: {question.question_id}")
            seen_questions.add(question.question_id)
            questions.append(Question(id=question.question_id, paper_id=paper_id, question=question.question))
    counts = (len(papers), len(questions))
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"expected full official QASPER test counts {EXPECTED_COUNTS}, got {counts}")
    files = {
        "corpus.jsonl": b"".join(_json_bytes(paper.model_dump(mode="json")) for paper in papers),
        "questions.jsonl": b"".join(_json_bytes(question.model_dump(mode="json")) for question in questions),
    }
    metadata = {
        "dataset": "QASPER", "version": "0.3", "split": "test", "schema_version": 1,
        "source": {"filename": raw_path.name, "sha256": hashlib.sha256(raw_bytes).hexdigest(),
                   "bytes": len(raw_bytes)},
        "counts": {"papers": len(papers), "questions": len(questions),
                   "paragraphs": sum(len(paper.paragraphs) for paper in papers),
                   "full_text_paragraphs": sum(len(section.paragraphs) for paper in source.values()
                                               for section in paper.full_text),
                   "captions": sum(len(paper.figures_and_tables) for paper in source.values()),
                   "content_bytes": sum(len(paper.content.encode("utf-8")) for paper in papers)},
        "files": {name: {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
                  for name, content in files.items()},
        "rendering": "Source-order Markdown hierarchy; UTF-8 paragraph spans are half-open byte offsets.",
        "limitation": "Full text and supplied figure/table captions only; image pixels and table cells "
                      "in images are unavailable. All questions, including unanswerable and figure-based, are retained.",
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (output / name).write_bytes(content)
    (output / "dataset.json").write_bytes(_json_bytes(metadata))
