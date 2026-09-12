"""A real loopback multipart service, native decoding, persistence and evidence reads."""
from contextlib import contextmanager
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import shutil
from threading import Thread
import wave

import httpx
import pytest

from scone_memory import HashEmbedder, MemoryEngine
from scone_memory.api import create_app
from scone_memory.backends import SqliteDocumentStore, SqliteVectorIndex
from scone_memory.backends.blobs import FileBlobStore
from scone_memory.ingestion.document_media import DocumentMedia
from scone_memory.ingestion.files import ingest_document
from scone_memory.ingestion.formats.media import MediaDocumentParser
from scone_memory.providers.transcription.document import LocalDocumentTranscriber


@contextmanager
def transcription_service():
    requests=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def do_POST(self):
            size=int(self.headers['Content-Length'])
            if not 0<size<1_000_000:
                self.send_error(413); return
            raw=self.rfile.read(size)
            message=BytesParser(policy=default).parsebytes(
                ('Content-Type: '+self.headers['Content-Type']+'\r\n\r\n').encode()+raw)
            fields={part.get_param('name',header='content-disposition'):part.get_payload(decode=True)
                    for part in message.iter_parts()}
            requests.append((self.path,fields,self.headers.get('Authorization')))
            body=json.dumps({'text':'Café meeting Friday.', 'segments':[
                {'start':0.1,'end':0.8,'text':'Café meeting Friday.'}]}).encode()
            self.send_response(200)
            self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(body)))
            self.end_headers(); self.wfile.write(body)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=Thread(target=server.serve_forever,daemon=True); thread.start()
    try: yield f'http://127.0.0.1:{server.server_port}/v1',requests
    finally: server.shutdown(); server.server_close(); thread.join(timeout=5)


async def test_real_local_transcription_is_retained_and_not_repeated_after_restart(tmp_path):
    ffmpeg=shutil.which('ffmpeg')
    if ffmpeg is None: pytest.skip('local ffmpeg required')
    output=io.BytesIO()
    with wave.open(output,'wb') as wav:
        wav.setnchannels(2); wav.setsampwidth(2); wav.setframerate(48000)
        wav.writeframes(b'\0'*192000)
    with transcription_service() as (endpoint,requests):
        provider=LocalDocumentTranscriber(base_url=endpoint,model='operator-selected',api_key='fixture-key')
        media=DocumentMedia(MediaDocumentParser(provider,ffmpeg_executable=ffmpeg),revision='selected-v1')
        identity=None
        for first in (True,False):
            engine=await MemoryEngine(SqliteDocumentStore(tmp_path/'db'),SqliteVectorIndex(tmp_path/'db'),
                HashEmbedder(),blobs=FileBlobStore(tmp_path/'blobs')).open()
            try:
                if first:
                    saved=await ingest_document(engine,'alpha',output.getvalue(),filename='meeting.wav',parser=media.parser())
                    identity=saved.added.episode_id
                app=create_app(engine,{'reader':'alpha'},roles={'reader':'read'},document_media=media)
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app),base_url='http://fixture',
                    headers={'authorization':'Bearer reader'}) as client:
                    evidence=await client.get(f'/v1/episodes/{identity}/document')
                    assert evidence.status_code==200,evidence.text
                    assert evidence.json()['segments'][0]['metadata']['start_seconds']=='0.1'
                    audio=await client.get(f'/v1/episodes/{identity}/document/audio')
                    assert audio.status_code==200,audio.text
                    assert audio.content==requests[0][1]['file']
                recalled=await engine.recall('alpha','meeting Friday')
                assert any(item.episode_id==identity for item in recalled.items)
                assert len(requests)==1
            finally: await engine.close()
        path,fields,authorization=requests[0]
        assert path=='/v1/audio/transcriptions' and authorization=='Bearer fixture-key'
        assert fields['model']==b'operator-selected'
        assert fields['response_format']==b'verbose_json'
        assert fields['timestamp_granularities[]']==b'segment'
        with wave.open(io.BytesIO(fields['file']),'rb') as wav:
            assert (wav.getnchannels(),wav.getframerate(),wav.getnframes())==(1,16000,16000)
