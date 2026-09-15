"""The client carries a scan's chosen OCR language and the languages a host offers."""
import pytest

from scone.document_models import DocumentFormats, PdfOcr
from scone.errors import SconeError


def catalog(**pdf_ocr):
    return {'formats': {}, 'max_input_bytes': 1024,
            'pdf_ocr': {'available': True, 'modes': ['all_pages'], 'reading_orders': ['provider'], **pdf_ocr}}


def test_a_chosen_language_round_trips_and_an_unchosen_one_is_not_sent():
    chosen = PdfOcr('all_pages', 'provider', 'jpn+eng')
    assert chosen.to_json() == {'mode': 'all_pages', 'reading_order': 'provider', 'language': 'jpn+eng'}
    assert PdfOcr.from_json(chosen.to_json()) == chosen
    plain = PdfOcr('all_pages', 'provider')
    assert plain.to_json() == {'mode': 'all_pages', 'reading_order': 'provider'}
    assert PdfOcr.from_json(plain.to_json()) == plain


@pytest.mark.parametrize('language', ['', '../eng', 'a' * 33, 7])
def test_a_malformed_language_is_refused_on_both_sides(language):
    with pytest.raises(SconeError):
        PdfOcr('all_pages', 'provider', language)  # type: ignore[arg-type]
    with pytest.raises(SconeError):
        PdfOcr.from_json({'mode': 'all_pages', 'reading_order': 'provider', 'language': language})


def test_the_catalog_lists_offered_languages_and_a_host_without_them_lists_none():
    assert DocumentFormats.from_json(catalog(languages=['deu', 'jpn+eng'])).pdf_ocr_languages == ('deu', 'jpn+eng')
    assert DocumentFormats.from_json(catalog()).pdf_ocr_languages == ()
    with pytest.raises(SconeError):
        DocumentFormats.from_json(catalog(languages=['../eng']))
