"""Object rendering preserves data tokens while removing layout whitespace."""
import json

import pytest

from scone_memory.providers.structured_answer import object_answer


@pytest.mark.parametrize('envelope', [None, 'action_first', 'answer_first'])
def test_compaction_preserves_numeric_tokens_escapes_keys_and_string_spaces(envelope):
    raw = r'{ "decimal": 1.234567890123456789, "huge": 1E+400, "neg": -0, "quoted": "\" x ", "slash": "\\", "unicode": "\u754c", "nested": [ { "a": true }, null ] }'
    expected = r'{"decimal":1.234567890123456789,"huge":1E+400,"neg":-0,"quoted":"\" x ","slash":"\\","unicode":"\u754c","nested":[{"a":true},null]}'
    if envelope == 'action_first':
        raw = ' \n { "action" : "answer", "answer" : '+raw+' } \t '
    elif envelope == 'answer_first':
        raw = '{"answer":'+raw+', "action":"answer"}'
    assert object_answer(raw, action_envelope=envelope is not None) == expected


@pytest.mark.parametrize('raw', ['[]', 'null', '"text"', '{"a":NaN}', '{"a":Infinity}',
    '{"a":1,"a":2}', '{"a":{"x":1,"x":2}}', '{"\\u0061":1,"a":2}',
    '```json\n{}\n```', '{} trailing', '{} {}'])
def test_invalid_objects_are_rejected_without_repair(raw):
    with pytest.raises(ValueError):
        object_answer(raw)


@pytest.mark.parametrize('raw', ['{"action":"answer","answer":"{}"}',
    '{"action":"answer","answer":[]}', '{"action":"answer","answer":{},"extra":1}',
    '{"action":"answer","answer":{},"answer":{}}', '{"action":"answer"}'])
def test_invalid_action_envelopes_are_rejected(raw):
    with pytest.raises(ValueError):
        object_answer(raw, action_envelope=True)


def test_non_answer_action_is_left_for_the_tool_parser():
    assert object_answer('{"action":"search_memory","query":"Juniper","limit":1}',
                         action_envelope=True) is None


def test_unicode_size_is_checked_after_object_rendering():
    with pytest.raises(ValueError, match='byte limit'):
        object_answer(json.dumps({'text':'界'*22000}, ensure_ascii=False))
