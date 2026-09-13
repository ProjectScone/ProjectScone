"""Process-local windows onto the answer each running step is writing.

The loop delivers a step's public text to a sink; this holds one text
window per running step, keyed by space, run and step, so a reader can
follow the answer as it is written and a later reader gets the receipt
instead. Provisional by design: nothing here is stored, nothing survives
the process, and the verified result is what ``result`` returns. Text
streamed before a turn that then called tools is withdrawn, which the
window records as a chunk with no text, so a reader clears it rather than
keeping it. A step that fails leaves a failed, closed window that offers
no text at all.
"""

from __future__ import annotations

from ..api.text_stream import TextWindow

Key = tuple[str, str, str]


class StepText:
    """The sink one step writes through; ``PublicText`` by shape."""

    def __init__(self, window: TextWindow) -> None:
        self.window = window

    async def append(self, text: str) -> None:
        self.window.append(text)

    def withdraw(self) -> None:
        self.window.withdraw()


class AgentRunText:
    def __init__(self) -> None:
        self._windows: dict[Key, TextWindow] = {}

    def open(self, space: str, run_id: str, step_id: str) -> StepText:
        """A fresh window for a step about to run. A step run again -- a
        resumed pause, a retry -- gets a new window; the old one, if any,
        is closed first so no reader keeps following it."""
        key = (space, run_id, step_id)
        old = self._windows.get(key)
        if old is not None and not old.closed:
            old.finish()
        window = TextWindow()
        self._windows[key] = window
        return StepText(window)

    def window(self, space: str, run_id: str, step_id: str) -> TextWindow | None:
        return self._windows.get((space, run_id, step_id))

    def close(self, space: str, run_id: str, step_id: str, *, failed: bool) -> None:
        window = self._windows.get((space, run_id, step_id))
        if window is None:
            return
        if failed:
            window.failed = True
        window.finish()

    def close_all(self) -> None:
        for window in self._windows.values():
            if not window.closed:
                window.failed = True
                window.finish()
