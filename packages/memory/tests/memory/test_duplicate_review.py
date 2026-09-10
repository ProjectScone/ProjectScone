from dataclasses import replace

import pytest

from scone_memory.deduplication import CandidateBatch, DeduplicationConfig, DocumentDuplicateDetector, DocumentRevision, SemanticCandidate
from scone_memory.deduplication.review import project_document, review_notification


class Provider:
    def __init__(self, text):
        self.text = text
    async def candidates(self, document, *, query_embeddings, embedder_id, limit):
        return CandidateBatch(documents=(DocumentRevision(document.scope, "old", "1", self.text),))


async def report_for(text, repeated):
    doc = DocumentRevision("scope", "new", "1", text)
    report = await DocumentDuplicateDetector(DeduplicationConfig(semantic_enabled=False)).inspect(doc, Provider(repeated))
    return doc, report


async def test_default_review_keeps_original_search_text_and_exclusion_is_explicit():
    text = "A repeated paragraph copied from the original source document."
    doc, report = await report_for(text, text)
    kept = project_document(doc, report)
    assert kept.action == "keep"
    assert kept.retained[0].text == text and kept.excluded_bytes == 0
    excluded = project_document(doc, report, action="exclude_document")
    assert not excluded.retained and excluded.excluded_bytes == len(text.encode())
    assert doc.text == text


async def test_suppression_preserves_unicode_unique_passages_and_offsets():
    repeated = "The calibrator starts before each scheduled research campaign."
    text = "新 findings. " + repeated + " Café independent conclusion."
    doc, report = await report_for(text, repeated)
    projection = project_document(doc, report, action="suppress_copied_passages")
    assert [span.text for span in projection.retained] == ["新 findings. ", " Café independent conclusion."]
    assert projection.excluded_bytes == len(repeated.encode())
    for span in projection.retained:
        assert text.encode()[span.start:span.end].decode() == span.text
    assert doc.text == text


async def test_overlapping_copied_ranges_are_removed_once():
    text = "A repeated paragraph copied from the original source document. UNIQUE."
    doc, report = await report_for(text, text[:-8])
    report = replace(report, matches=report.matches * 3)
    projection = project_document(doc, report, action="suppress_copied_passages")
    assert projection.retained[0].text == " UNIQUE."
    assert projection.excluded_bytes == len(text[:-8].encode())


async def test_stale_content_revision_scope_and_offsets_are_rejected():
    doc, report = await report_for("An original paragraph with enough detail to match.", "An original paragraph with enough detail to match.")
    for changed in (replace(doc, text=doc.text + "!"), replace(doc, revision="2"), replace(doc, scope="other"), replace(doc, byte_offset=1)):
        with pytest.raises(ValueError, match="report"):
            project_document(changed, report, action="suppress_copied_passages")


async def test_invalid_or_split_utf8_ranges_are_rejected():
    doc, report = await report_for("Café repeated phrase with sufficient detail in the original.", "Café repeated phrase with sufficient detail in the original.")
    for span in (replace(report.matches[0], start=4), replace(report.matches[0], end=999)):
        with pytest.raises(ValueError, match="range"):
            project_document(doc, replace(report, matches=(span,)), action="suppress_copied_passages")


async def test_paraphrase_flags_never_remove_literal_text():
    from scone_memory.deduplication import SemanticMatch
    text = "The residents left their homes after flood warnings."
    doc, report = await report_for(text, "Unrelated different candidate with zero copied passages.")
    semantic = SemanticMatch(0, len(text.encode()), "old", "1", 0, 30, None, .99, "semantic-model")
    report = replace(report, semantic_matches=(semantic,), requires_review=True)
    projection = project_document(doc, report, action="suppress_copied_passages")
    assert projection.retained[0].text == text and projection.excluded_bytes == 0
    notice = review_notification(report)
    assert notice is not None and "suppress_copied_passages" not in notice.available_actions
    assert notice.semantic_candidates == 1 and notice.copied_fraction == 0


async def test_notification_links_documentation_and_preserves_source_references():
    text = "Repeated information with traceable documentation source reference."
    _, report = await report_for(text, text)
    notice = review_notification(report, documentation_url="https://example.test/docs/duplicates")
    assert notice is not None and notice.requires_review
    assert notice.documentation_url == "https://example.test/docs/duplicates"
    assert notice.available_actions == ("keep", "suppress_copied_passages", "exclude_document")
    assert notice.sources[0].source_id == "old"
    assert "review" in notice.message.lower()


async def test_clean_document_has_no_review_notification():
    _, report = await report_for("Entirely unique subject matter.", "Different things in another source.")
    assert review_notification(report) is None


async def test_projection_rejects_unknown_action():
    doc, report = await report_for("A unique document.", "A different text.")
    with pytest.raises(ValueError, match="action"):
        project_document(doc, report, action="delete_original")
