"""Keys pressed on a phone, as a conversation takes them."""

from __future__ import annotations

import pytest

from scone_memory.realtime.keypad import KEYS, Keypress, keypad_key


def test_a_keypress_is_one_of_sixteen_keys_and_says_how_it_arrived():
    assert sorted(KEYS) == sorted("0123456789*#ABCD")
    assert Keypress("5").source == "event"
    assert Keypress("#", "inband", offset_ms=120.0, tone_ms=51.2).tone_ms == 51.2
    for bad in ("", "55", "a", "E", " 5"):
        with pytest.raises(ValueError, match="keypress is one of"):
            Keypress(bad)
    with pytest.raises(ValueError, match="arrives by"):
        Keypress("5", "voice")
    assert [keypad_key(v) for v in ("d", " 7 ", 0, True, 1.0, "##")] == ["D", "7", "0", None, None, None]
