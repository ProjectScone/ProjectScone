"""The documented context example must execute, not just import."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("pipecat")


def test_context_example_supplies_sources_without_recapturing_them():
    script = Path(__file__).resolve().parents[1] / "examples" / "pipecat_context.py"
    assert script.is_file(), "The runnable context example is missing"
    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["mode"] == "scripted responder; no model inference or live media"
    assert report["request_receipt"]["status"] == "prepared"
    assert report["request_receipt"]["references"]
    assert report["request_receipt"]["recall_event_id"] is not None
    assert report["source_supplied_to_responder"] is True
    assert report["source_block_in_shared_history"] is False
    assert report["stored_transcripts"] == 1
    assert [item["text"] for item in report["captured_recall"]] == [
        "Scripted reply: the request received a Juniper source."
    ]
