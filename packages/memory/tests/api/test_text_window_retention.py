"""A finished window keeps its text for the readers already listening, and
offers none of it to anyone who arrives after the end."""

import pytest

from scone_memory.api.text_stream import TextWindow


def _window_with_two_chunks() -> TextWindow:
    window = TextWindow()
    window.append("one ")
    window.append("two")
    return window


def test_text_stays_readable_for_a_reader_attached_before_the_end():
    window = _window_with_two_chunks()
    window.attach()
    window.finish()
    assert window.closed
    assert window.next_after(0) == (None, (1, "one "))
    assert window.next_after(1) == (None, (2, "two"))
    assert window.next_after(2) == (None, None)
    with pytest.raises(RuntimeError):
        window.append("three")


def test_text_is_forgotten_when_the_last_attached_reader_leaves():
    window = _window_with_two_chunks()
    window.attach()
    window.attach()
    window.finish()
    window.detach()
    assert window.next_after(0) == (None, (1, "one "))
    window.detach()
    assert window.next_after(0) == (None, None)


def test_a_window_finished_with_no_reader_offers_nothing():
    window = _window_with_two_chunks()
    window.finish()
    assert window.closed
    assert window.next_after(0) == (None, None)


def test_a_failed_window_offers_nothing_even_to_an_attached_reader():
    window = _window_with_two_chunks()
    window.attach()
    with pytest.raises(ValueError):
        window.append("x" * 65537)
    assert window.failed and window.closed
    assert window.next_after(0) == (None, None)
