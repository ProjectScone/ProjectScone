"""Keep deployment termination inside Python's resource-cleanup boundaries."""
from collections.abc import Iterator
from contextlib import contextmanager
import signal
import threading
from types import FrameType


def _terminate(signum: int, _frame: FrameType | None) -> None:
    # Uvicorn replays SIGTERM after ASGI shutdown, before the caller's finally.
    raise SystemExit(128 + signum)


@contextmanager
def termination_unwinds() -> Iterator[None]:
    """Unwind owned storage cleanup on SIGTERM, then restore the host handler."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    if signal.getsignal(signal.SIGTERM) != signal.SIG_DFL:
        yield
        return
    previous = signal.signal(signal.SIGTERM, _terminate)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)
