"""Opt out before optional ONNX dependencies initialize native telemetry."""
from __future__ import annotations

from collections.abc import Callable
import importlib
import os
from typing import cast


def prepare_onnx_runtime() -> None:
    """Disable startup telemetry and events for an already imported runtime.

    Hosts importing ONNX before Scone must set ORT_DISABLE_TELEMETRY=1 before
    that import themselves: the API cannot undo earlier initialization events.
    """
    os.environ["ORT_DISABLE_TELEMETRY"] = "1"
    runtime = importlib.import_module("onnxruntime")
    disable = cast(Callable[[], None], getattr(runtime, "disable_telemetry_events"))
    disable()
