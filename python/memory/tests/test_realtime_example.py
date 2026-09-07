"""Run the documented native source-to-response example, without a provider."""

import json
from pathlib import Path
import subprocess
import sys


def test_native_conversation_example_preserves_sources_and_public_capture():
    script = Path(__file__).resolve().parents[1] / "examples" / "realtime_conversation.py"
    assert script.is_file(), "Native conversation example is missing"
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["mode"] == "scripted provider; no inference or live media"
    assert report["memory_context"]["status"] == "prepared"
    assert report["memory_context"]["references"]
    assert report["source_supplied"] is True
    assert report["transcript"] == ["How is Juniper calibrated?", "Use Polaris."]
