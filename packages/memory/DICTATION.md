# Local dictation

Dictation is optional and independent of any application or model-provider account.
`GET /v1/dictation/status` and `POST /v1/dictation/transcribe?language=en` use the
host's authentication and return the same `mode: "local"` contract for either
backend. Configure an explicit worker interpreter, model directory, and ffmpeg
executable. All three paths must be absolute. With none configured, dictation is
unavailable; there is no hosted transcription fallback.

## Backends

`SCONE_DICTATION_BACKEND=mlx` is the default, preserving Apple silicon deployments.
Install `scripts/dictation-requirements.txt` into a dedicated virtual environment
and provide the previously downloaded MLX model (`config.json` plus `weights.npz`
or `model.safetensors`). Existing configurations need no new setting.

For a Linux CPU host, use `SCONE_DICTATION_BACKEND=faster-whisper` and install
`scripts/dictation-cpu-requirements.txt` into a separate Python 3.12 virtual
environment. On Ubuntu 24.04 the OS packages are `python3-venv`, `ffmpeg`, and
`libgomp1`. No GPU, MLX, remote inference service, or API key is required.
Both x86-64 and AArch64 Linux have
[CTranslate2 binary wheel support](https://opennmt.net/CTranslate2/installation.html).
Install dependencies on the target architecture; do not copy a Mac or x86 virtual
environment onto an ARM host. Confirm native model inference on the selected host.

```sh
python3 -m venv /opt/scone-dictation
/opt/scone-dictation/bin/python -m pip install -r scripts/dictation-cpu-requirements.txt
```

Download a CTranslate2 Whisper model during provisioning, before starting the
application. A bounded starting point for a 4-vCPU / 8-GB server is the
[multilingual base model](https://huggingface.co/Systran/faster-whisper-base/tree/ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66).
This one-time download uses the public model artifact host; inference uses only
the resulting files on the locally managed server. Run the following with an
operator-owned destination directory:

```sh
/opt/scone-dictation/bin/python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Systran/faster-whisper-base",
    revision="ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66",
    local_dir="/opt/scone-models/faster-whisper-base",
    allow_patterns=["config.json", "model.bin", "tokenizer.json", "vocabulary.txt"],
    token=False,
)
PY
```

Keep the model directory readable, but not writable, by the application user.
Deploy the four model files together. In particular, `tokenizer.json` is required:
the [upstream loader](https://github.com/SYSTRAN/faster-whisper/blob/v1.2.1/faster_whisper/transcribe.py)
has a download fallback when that file is absent, so Scone rejects such a model
before importing the inference backend. Model IDs and relative paths are not
accepted at runtime.

```dotenv
SCONE_DICTATION_BACKEND=faster-whisper
SCONE_DICTATION_PYTHON=/opt/scone-dictation/bin/python
SCONE_DICTATION_MODEL=/opt/scone-models/faster-whisper-base
SCONE_DICTATION_FFMPEG=/usr/bin/ffmpeg
```

Use one API process to preserve the process-local admission limits. The CPU
worker explicitly uses `device="cpu"`, `compute_type="int8"`, two CPU threads,
one inference worker, beam size one, and `local_files_only=True`. Runtime Hub
access, implicit credentials, and telemetry are disabled in the worker's
restricted environment. It inherits no provider credentials. Changing the
application's embedding or model provider does not change dictation.

## Bounds and cleanup

Uploads are limited to 5 MiB and 15 seconds, with four admitted uploads per API
process. Only one transcription may run at a time. ffmpeg normalizes the supplied
audio over pipes to mono 16-kHz PCM, with no external input protocols, a 20-second
conversion deadline, and a bounded output size. Recordings exceeding 60 seconds
are rejected before model inference. The model has a 90-second deadline; the
complete transcription operation has a 120-second route deadline. Overload is
reported as `dictation_busy`.

Audio stays in memory and anonymous pipes; Scone creates no temporary audio
files. The model subprocess is killed and reaped on cancellation or timeout.
Failures release transcription capacity and return sanitized errors. The
endpoint returns text without storing audio or transcript records. Transcripts
are limited to 16,000 characters. The model loads in each isolated worker; actual
latency depends on the server and must be checked with representative recordings.

## Verification

From `packages/memory`, run:

```sh
python -m pytest -q tests/speech/test_dictation.py tests/speech/test_whisper_worker.py tests/runtime/test_dictation_config.py tests/api/test_dictation.py
```

The worker tests use NumPy and substitute only the heavyweight inference library;
they exercise local model validation, CPU selection, bounded inference arguments,
WAV conversion, output assembly, failure propagation, and preservation of the
MLX contract. Separate tests run real subprocesses to verify cancellation and
timeout cleanup. Run an authenticated real-audio request on the target Linux host
after installing the model, and repeat it with outbound networking disabled to
verify that the packaged model is complete. Status checks validate configured
files; a successful status alone does not prove native dependencies or model
contents are valid.
