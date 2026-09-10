"""Exercise the real CLI/HTTP/storage lifecycle using disposable fixture data.

No pytest, ASGI test transport, hosted services, or model downloads. This checks
operational behavior, not semantic retrieval or generated-answer accuracy.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
from importlib.metadata import version
from io import BytesIO
import ipaddress
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time
from typing import Iterator
from urllib.parse import urlsplit
import uuid

import httpx
import scone_memory


def loopback(value: str) -> str:
    url = urlsplit(value)
    try:
        valid_host = url.hostname == 'localhost' or ipaddress.ip_address(url.hostname or '').is_loopback
        valid_port = url.port is None or url.port > 0
    except ValueError as error:
        raise argparse.ArgumentTypeError('expected a loopback HTTP(S) endpoint') from error
    if (not valid_host or not valid_port or url.scheme not in ('http', 'https')
            or url.username or url.password or url.query or url.fragment or url.path not in ('', '/')):
        raise argparse.ArgumentTypeError('expected a loopback HTTP(S) endpoint without credentials or path')
    return value.rstrip('/')


def fixtures() -> tuple[bytes, bytes]:
    from PIL import Image
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    font = DictionaryObject({NameObject('/Type'): NameObject('/Font'),
        NameObject('/Subtype'): NameObject('/Type1'), NameObject('/BaseFont'): NameObject('/Helvetica')})
    for text in ('Juniper calibration uses Polaris.', 'The deployment window opens on Friday.'):
        page = writer.add_blank_page(width=612, height=792)
        page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'):
            DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
        stream = DecodedStreamObject()
        stream.set_data(f'BT /F1 12 Tf 72 720 Td ({text}) Tj ET'.encode())
        page[NameObject('/Contents')] = writer._add_object(stream)
    pdf, picture = BytesIO(), BytesIO()
    writer.write(pdf)
    Image.new('RGB', (32, 32), 'yellow').save(picture, format='PNG')
    return pdf.getvalue(), picture.getvalue()


@contextmanager
def serve(directory: Path, env: dict[str, str], attempt: int) -> Iterator[httpx.Client]:
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    with (directory / f'server-{attempt}.log').open('x') as log:
        process = subprocess.Popen([sys.executable, '-m', 'scone_memory.runtime.cli', 'serve'],
            env={**env, 'SCONE_PORT': str(port)}, stdout=log, stderr=log)
        try:
            with httpx.Client(base_url=f'http://127.0.0.1:{port}', timeout=60, trust_env=False) as client:
                deadline = time.monotonic() + 60
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f'server exited; inspect server-{attempt}.log')
                    try:
                        if client.get('/healthz', timeout=1).status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    if time.monotonic() >= deadline:
                        raise RuntimeError('server startup exceeded 60 seconds')
                    time.sleep(.05)
                yield client
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                raise RuntimeError('server required forced termination')
            if process.returncode not in (0, 143):
                raise RuntimeError(f'server shutdown returned {process.returncode}; inspect server-{attempt}.log')


def require(condition: bool, label: str, checks: list[str]) -> None:
    if not condition:
        raise RuntimeError(label)
    checks.append(label)


def exercise(client: httpx.Client, key: str, other: str, pdf: bytes, picture: bytes,
             records: dict[str, object], checks: list[str], timings: list[dict[str, object]]) -> None:
    headers = {'authorization': f'Bearer {key}'}

    def request(method: str, path: str, *, body: object = None,
                raw: bytes | None = None, mime: str | None = None,
                auth: str | None = key, expected: int = 200) -> httpx.Response:
        started = time.perf_counter()
        response = client.request(method, path, json=body, content=raw,
            headers={**({'authorization': f'Bearer {auth}'} if auth else {}),
                     **({'content-type': mime} if mime else {})})
        timings.append({'method': method, 'path': path, 'status': response.status_code,
                        'elapsed_ms': (time.perf_counter() - started) * 1000})
        require(response.status_code == expected, f'{method} {path}: expected {expected}', checks)
        return response

    request('GET', '/v1/status', auth=None, expected=401)
    capabilities = request('GET', '/v1/capabilities').json()['features']
    require(capabilities['documents.pdf'] and capabilities['images.context'], 'ingestion advertised', checks)
    if not records:
        uploaded = request('POST', '/v1/attachments', raw=pdf, mime='application/pdf').json()
        body = {'attachment_id': uploaded['attachment_id']}
        saved = request('POST', '/v1/documents/pdf', body=body).json()
        records['pdf'] = saved['added']['episode_id']
        records['pdf_attachment'] = uploaded['attachment_id']
        repeat = request('POST', '/v1/documents/pdf', body=body).json()['added']
        require(repeat['deduplicated'] and repeat['episode_id'] == records['pdf'], 'PDF retry deduplicated', checks)
        image = request('POST', '/v1/attachments', raw=picture, mime='image/png').json()
        saved = request('POST', '/v1/images', body={'attachment_id': image['attachment_id'], 'context': {
            'source': 'fixture:yellow-tile', 'attributes': [
                {'kind': 'caption', 'value': 'A yellow calibration tile.', 'origin': 'supplied'}],
            'entities': [{'entity_id': 'fixture:tile', 'name': 'Calibration tile',
                          'relationship': 'depicts', 'attribute_indexes': [0]}]}}).json()
        records['image'] = saved['added']['episode_id']
        records['image_attachment'] = image['attachment_id']
        request('POST', '/v1/documents/pdf', body=body, auth=other, expected=404)

    recall = request('GET', '/v1/recall?q=Polaris').json()
    candidates = [item for item in recall['items'] if item['episode_id'] == records['pdf']]
    require(bool(candidates), 'PDF source returned by recall', checks)
    evidence = request('GET', f"/v1/episodes/{records['pdf']}/pdf?chunk_id={candidates[0]['chunk_id']}").json()
    require(evidence['pages'][0]['number'] == 1, 'retrieved PDF chunk resolves to page one', checks)
    require(request('GET', evidence['download_path']).content == pdf, 'original PDF bytes unchanged', checks)
    request('GET', evidence['download_path'], auth=other, expected=404)
    found = request('GET', '/v1/images/search?query=yellow&entity_id=fixture%3Atile').json()
    require(len(found['matches']) == 1, 'entity-filtered image found', checks)
    match = found['matches'][0]
    require(match['episode_id'] == records['image'], 'image source identity preserved', checks)
    require(request('GET', match['download_path']).content == picture, 'original image bytes unchanged', checks)
    require(not request('GET', '/v1/images/search?query=yellow', auth=other).json()['matches'],
            'foreign space returns no image matches', checks)
    # Keep every returned source and score for inspection; checks do not rewrite rankings.
    records['last_recall'] = recall
    records['last_image_search'] = found
    require(client.get('/healthz', headers=headers).status_code == 200, 'server remains healthy', checks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qdrant-url', required=True, type=loopback)
    parser.add_argument('--output', type=Path, required=True, help='New directory; preserves logs and raw results')
    args = parser.parse_args()
    directory = args.output.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=False)
    directory.chmod(0o700)
    pdf, picture = fixtures()
    (directory / 'fixture.pdf').write_bytes(pdf)
    (directory / 'fixture.png').write_bytes(picture)
    collection = 'scone_lifecycle_' + uuid.uuid4().hex
    key, other = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    env = {name: value for name, value in os.environ.items()
           if not name.startswith(('SCONE_', 'AWS_', 'OTEL_'))}
    env.update(SCONE_HOST='127.0.0.1', SCONE_DOCUMENTS='sqlite', SCONE_VECTORS='qdrant',
        SCONE_QDRANT_URL=args.qdrant_url, SCONE_QDRANT_COLLECTION=collection,
        SCONE_SQLITE_PATH=str(directory / 'memory.db'), SCONE_BLOBS='file',
        SCONE_BLOB_DIR=str(directory / 'blobs'), SCONE_EMBEDDER='hash',
        SCONE_API_KEYS=f'{key}:alpha:full,{other}:beta:full',
        HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', ORT_DISABLE_TELEMETRY='1',
        NO_PROXY='*', no_proxy='*')
    checks: list[str] = []
    timings: list[dict[str, object]] = []
    records: dict[str, object] = {}
    source_root = Path(scone_memory.__file__).parent
    source_hash = hashlib.sha256()
    for source in sorted(source_root.rglob('*.py')):
        source_hash.update(str(source.relative_to(source_root)).encode() + b'\0' + source.read_bytes() + b'\0')
    report: dict[str, object] = {'schema_version': 1, 'kind': 'operational_fixture',
        'python': sys.version, 'embedder': 'hash', 'generation_tested': False,
        'source_sha256': source_hash.hexdigest(),
        'versions': {name: version(name) for name in ('scone-memory', 'pypdf', 'pillow', 'httpx', 'qdrant-client', 'uvicorn')},
        'collection': collection, 'fixture_pdf_sha256': hashlib.sha256(pdf).hexdigest(),
        'fixture_image_sha256': hashlib.sha256(picture).hexdigest(),
        'checks': checks, 'requests': timings, 'records': records, 'passed': False}
    creation_attempted = False
    with httpx.Client(base_url=args.qdrant_url, timeout=30, trust_env=False, follow_redirects=False) as qdrant:
        try:
            require(qdrant.get(f'/collections/{collection}').status_code == 404,
                    'owned collection initially absent', checks)
            creation_attempted = True
            with serve(directory, env, 1) as client:
                exercise(client, key, other, pdf, picture, records, checks, timings)
            with serve(directory, env, 2) as client:
                exercise(client, key, other, pdf, picture, records, checks, timings)
                client.headers['authorization'] = f'Bearer {key}'
                for name in ('pdf', 'image'):
                    response = client.delete(f'/v1/episodes/{records[name]}')
                    require(response.status_code == 200, f'{name} deleted', checks)
                    require(client.get(f'/v1/episodes/{records[name]}').status_code == 410,
                            f'{name} source revoked', checks)
                require(not client.get('/v1/recall?q=Polaris').json()['items'], 'deleted evidence absent from recall', checks)
                require(not client.get('/v1/images/search?query=yellow').json()['matches'], 'deleted image absent from search', checks)
            with serve(directory, env, 3) as client:
                client.headers['authorization'] = f'Bearer {key}'
                require(not client.get('/v1/recall?q=Polaris').json()['items'], 'deletion survives second restart', checks)
                require(client.get(f"/v1/episodes/{records['pdf']}/pdf").status_code == 410,
                        'PDF provenance remains revoked after restart', checks)
            report['passed'] = True
        except BaseException as error:
            report['error'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            try:
                if creation_attempted:
                    response = qdrant.delete(f'/collections/{collection}')
                    require(response.status_code in (200, 404), 'owned collection deletion accepted', checks)
                    require(qdrant.get(f'/collections/{collection}').status_code == 404,
                            'owned collection cleanup verified', checks)
                report['cleanup_verified'] = True
            except BaseException as error:
                report['passed'] = False
                report['cleanup_verified'] = False
                report['cleanup_error'] = f'{type(error).__name__}: {error}'
                raise
            finally:
                (directory / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
                print(f'{len(checks)} checks; results: {directory / "results.json"}', flush=True)


if __name__ == '__main__':
    main()
