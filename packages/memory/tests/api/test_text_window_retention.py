"""A finished window still serves what it holds; only new text is refused."""

import pytest

from scone_memory.api.text_stream import TextWindow


def test_text_stays_readable_after_the_window_finishes():
    window = TextWindow()
    window.append("one ")
    window.append("two")
    window.finish()
    assert window.closed
    assert window.next_after(0) == (None, (1, "one "))
    assert window.next_after(1) == (None, (2, "two"))
    assert window.next_after(2) == (None, None)
    with pytest.raises(RuntimeError):
        window.append("three")
