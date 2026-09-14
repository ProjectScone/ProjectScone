"""The host's offered OCR languages come from one setting and build one engine each."""
import stat

import pytest

from scone_memory.runtime.config import Settings
from scone_memory.runtime.document_ocr import build_document_ocr


def test_the_setting_lists_languages_and_each_builds_its_own_engine(tmp_path, monkeypatch):
    executable = tmp_path / 'tesseract'
    executable.write_text('#!/bin/sh\n')
    executable.chmod(executable.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setattr('scone_memory.ingestion.document_ocr.find_spec', lambda _: object())
    settings = Settings.from_env({'SCONE_DOCUMENT_OCR_EXECUTABLE': str(executable),
                                  'SCONE_DOCUMENT_OCR_LANGUAGES': 'deu, jpn+eng'})
    assert settings.document_ocr_languages == ('deu', 'jpn+eng')
    ocr = build_document_ocr(settings)
    assert ocr is not None and ocr.languages == ('deu', 'jpn+eng')
    assert ocr.engine_for is not None and ocr.engine_for('deu').language == 'deu'
    assert Settings.from_env({}).document_ocr_languages == ()


def test_a_malformed_offered_language_stops_the_host_at_start(tmp_path, monkeypatch):
    executable = tmp_path / 'tesseract'
    executable.write_text('#!/bin/sh\n')
    executable.chmod(executable.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setattr('scone_memory.ingestion.document_ocr.find_spec', lambda _: object())
    with pytest.raises(ValueError, match='language'):
        build_document_ocr(Settings.from_env({'SCONE_DOCUMENT_OCR_EXECUTABLE': str(executable),
                                              'SCONE_DOCUMENT_OCR_LANGUAGES': 'deu,../x'}))
