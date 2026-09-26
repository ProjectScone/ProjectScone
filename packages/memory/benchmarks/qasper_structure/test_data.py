from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from qasper_structure import data
from qasper_structure.data import Paper, Paragraph, Question, _RawPaper, export, render_paper
from scone_memory.retrieval.section_routing import SectionSnapshot


def _paper() -> dict[str, object]:
    return {
        "title": "Résumé Ω", "abstract": "An abstract with café and 🧪.",
        "full_text": [
            {"section_name": "Methods ::: Model ::: Encoder ::: Details", "paragraphs": ["  é\n你好 🧪\t ", ""]},
            {"section_name": "Methods ::: Model ::: Decoder", "paragraphs": ["decoder only"]},
            {"section_name": "Results", "paragraphs": ["results only"]},
        ],
        "figures_and_tables": [{"file": "figure.png", "caption": "Figure 1: A caption."}],
        "qas": [
            {"question_id": "z", "question": "Unanswerable question?", "answers": [{"answer": {
                "unanswerable": True, "free_form_answer": "SECRET_GOLD", "evidence": ["SECRET_EVIDENCE"]}}]},
            {"question_id": "a", "question": "Figure question?", "answers": [{"answer": {
                "unanswerable": False, "evidence": ["FLOAT SELECTED: Figure 1"]}}]},
        ],
    }


def test_hierarchy_missing_parents_and_sibling_boundaries() -> None:
    paper = render_paper("p", _RawPaper.model_validate(_paper()))
    snapshot = SectionSnapshot.from_markdown("test", paper.id, paper.content)
    nodes = {snapshot.path(node.id): node for node in snapshot.nodes if node.title}
    title = ("Résumé Ω",)
    methods = nodes[(*title, "Methods")]
    model = nodes[(*title, "Methods", "Model")]
    encoder = nodes[(*title, "Methods", "Model", "Encoder")]
    details = nodes[(*title, "Methods", "Model", "Encoder", "Details")]
    decoder = nodes[(*title, "Methods", "Model", "Decoder")]
    results = nodes[(*title, "Results")]
    assert not methods.text.strip() and not model.text.strip() and not encoder.text.strip()
    assert details.parent_id == encoder.id
    assert decoder.parent_id == model.id
    assert encoder.end == details.end == decoder.start
    assert methods.end == model.end == decoder.end == results.start
    assert "decoder only" not in snapshot.original(encoder.id)
    assert "results only" not in snapshot.original(methods.id)


def test_parent_after_child_closes_child() -> None:
    source = _paper()
    source["full_text"] = [
        {"section_name": "Methods ::: Detail", "paragraphs": ["child"]},
        {"section_name": "Methods", "paragraphs": ["parent"]},
        {"section_name": "Methods ::: Other", "paragraphs": ["other"]},
    ]
    paper = render_paper("p", _RawPaper.model_validate(source))
    snapshot = SectionSnapshot.from_markdown("test", "p", paper.content)
    child = next(node for node in snapshot.nodes if node.title == "Detail")
    assert "parent" not in snapshot.original(child.id)
    other = next(node for node in snapshot.nodes if node.title == "Other")
    assert snapshot.path(other.id) == ("Résumé Ω", "Methods", "Other")


def test_unicode_paragraphs_roundtrip_including_empty_strings() -> None:
    source = _RawPaper.model_validate(_paper())
    paper = render_paper("p", source)
    expected = [source.abstract, *(text for section in source.full_text for text in section.paragraphs),
                *(figure.caption for figure in source.figures_and_tables)]
    assert [paragraph.text for paragraph in paper.paragraphs] == expected
    for paragraph in paper.paragraphs:
        assert paper.content.encode()[paragraph.start:paragraph.end].decode() == paragraph.text
    assert Paper.model_validate_json(paper.model_dump_json()) == paper


def test_neutral_fallback_headings() -> None:
    source = _paper()
    source["full_text"] = [{"section_name": value, "paragraphs": ["text"]}
                           for value in (None, "", "unknown", "Parent ::: ")]
    paper = render_paper("p", _RawPaper.model_validate(source))
    assert "## Section 1.1\n" in paper.content
    assert "## Section 2.1\n" in paper.content
    assert "## Section 3.1\n" in paper.content
    assert "### Section 4.2\n" in paper.content


def test_deterministic_export_hashes_and_no_gold_leak(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data, "EXPECTED_COUNTS", (2, 4))
    second = _paper()
    second["qas"] = [{"question_id": "q4", "question": "Fourth?"}, {"question_id": "q3", "question": "Third?"}]
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps({"p2": second, "p1": _paper()}), encoding="utf-8")
    out = tmp_path / "one"
    export(raw, out)
    export(raw, tmp_path / "two")
    for name in ("corpus.jsonl", "questions.jsonl", "dataset.json"):
        assert (out / name).read_bytes() == (tmp_path / "two" / name).read_bytes()
    corpus = [Paper.model_validate_json(line) for line in (out / "corpus.jsonl").read_text().splitlines()]
    questions = [Question.model_validate_json(line) for line in (out / "questions.jsonl").read_text().splitlines()]
    assert [paper.id for paper in corpus] == ["p1", "p2"]
    assert [(q.paper_id, q.id) for q in questions] == [("p1", "a"), ("p1", "z"), ("p2", "q3"), ("p2", "q4")]
    assert questions[1].question == "Unanswerable question?"
    for name in ("corpus.jsonl", "questions.jsonl"):
        content = (out / name).read_text()
        assert "SECRET_GOLD" not in content and "SECRET_EVIDENCE" not in content
        assert '"answers"' not in content and '"evidence"' not in content and '"unanswerable"' not in content
    manifest = json.loads((out / "dataset.json").read_text())
    assert manifest["source"]["sha256"] == hashlib.sha256(raw.read_bytes()).hexdigest()
    assert manifest["counts"]["questions"] == 4
    for name in ("corpus.jsonl", "questions.jsonl"):
        assert manifest["files"][name]["sha256"] == hashlib.sha256((out / name).read_bytes()).hexdigest()


@pytest.mark.parametrize("cross_paper", [False, True])
def test_duplicate_question_ids_rejected(tmp_path: Path, cross_paper: bool) -> None:
    paper = _paper()
    paper["qas"] = [{"question_id": "duplicate", "question": "First?"},
                    {"question_id": "duplicate", "question": "Second?"}]
    source = {"p1": paper} if not cross_paper else {"p1": _paper(), "p2": _paper()}
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="duplicate question identity"):
        export(raw, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_duplicate_paper_ids_rejected(tmp_path: Path) -> None:
    raw = tmp_path / "raw.json"
    raw.write_text('{"duplicate": {}, "duplicate": {}}')
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        export(raw, tmp_path / "out")


def test_incomplete_official_split_rejected(tmp_path: Path) -> None:
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps({"p": _paper()}))
    with pytest.raises(ValueError, match="expected full official QASPER test counts"):
        export(raw, tmp_path / "out")


def test_types_frozen_strict_and_offsets_validated() -> None:
    question = Question(id="q", paper_id="p", question="What?")
    with pytest.raises(ValidationError):
        question.id = "changed"
    with pytest.raises(ValidationError):
        Paragraph.model_validate({"text": "é", "start": "0", "end": 2})
    with pytest.raises(ValidationError):
        Paragraph(text="é", start=0, end=1)
    with pytest.raises(ValidationError):
        Paper(id="p", content="different", paragraphs=(Paragraph(text="é", start=0, end=2),))
    with pytest.raises(ValidationError):
        Question.model_validate({"id": "q", "paper_id": "p", "question": "What?", "answers": []})


def test_development_export_keeps_split_and_drops_annotations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data, 'DEV_COUNTS', (1, 2), raising=False)
    raw = tmp_path / 'dev.json'
    raw.write_text(json.dumps({'p': _paper()}))
    export(raw, tmp_path / 'bundle', split='dev')
    metadata = json.loads((tmp_path / 'bundle/dataset.json').read_text())
    assert metadata['split'] == 'dev'
    assert metadata['counts']['questions'] == 2
    assert 'SECRET_GOLD' not in (tmp_path / 'bundle/questions.jsonl').read_text()
    with pytest.raises(ValueError, match='test counts'):
        export(raw, tmp_path / 'wrong')


@pytest.mark.parametrize('literal', ['# a tweet, not a paper heading\\', '```python\nprint("é")\n```',
                                    '~~~\n# quoted heading\n~~~~\n``````'])
def test_literal_markdown_in_paragraph_does_not_change_paper_hierarchy(literal: str) -> None:
    source = _paper()
    source['full_text'] = [{'section_name': 'Cleaning', 'paragraphs': [literal]},
                           {'section_name': 'Results', 'paragraphs': ['correct result']}]
    paper = render_paper('p', _RawPaper.model_validate(source))
    snapshot = SectionSnapshot.from_markdown('dev', 'p', paper.content)
    paths = [snapshot.path(node.id) for node in snapshot.nodes if node.title]
    assert paths == [('Résumé Ω',), ('Résumé Ω', 'Abstract'), ('Résumé Ω', 'Cleaning'),
                     ('Résumé Ω', 'Results'), ('Résumé Ω', 'Figure and table captions')]
    paragraph = next(p for p in paper.paragraphs if p.text == literal)
    assert paper.content.encode()[paragraph.start:paragraph.end].decode() == literal
