import asyncio
from io import BytesIO
import shutil
import sys

import pytest
from PIL import Image, ImageDraw, ImageFont

from scone_memory.core.errors import InvalidInput
from scone_memory.ocr.process import run_bounded
from scone_memory.ocr.tesseract import TesseractOcr, parse_tsv


HEADER = 'level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n'


def test_tsv_preserves_unicode_scores_and_provider_reading_order():
    raw = HEADER + '5\t1\t1\t1\t1\t1\t5\t10\t20\t10\t85\tCafé\n'
    result = parse_tsv(raw.encode(), width=100, height=50, max_regions=10, engine='test')
    assert result.regions[0].text == 'Café'
    assert result.regions[0].box == (.05, .2, .25, .4)
    assert result.regions[0].score == .85


@pytest.mark.parametrize('row', [
    '5\t1\t1\t1\t1\t1\t95\t10\t20\t10\t85\toverflow\n',
    '5\t1\t1\t1\t1\t1\t5\t10\t20\t10\tnan\tinvalid\n',
    '5\t1\t1\t1\t1\t1\t5\t10\t20\t10\t101\tinvalid\n',
])
def test_tsv_rejects_invalid_geometry_or_scores(row):
    with pytest.raises(InvalidInput):
        parse_tsv((HEADER + row).encode(), width=100, height=50, max_regions=10, engine='test')


async def test_bounded_process_refuses_excess_output():
    with pytest.raises(InvalidInput, match='output'):
        await run_bounded([sys.executable, '-c', 'print("x" * 100000)'], b'', timeout=10., max_output=10)


@pytest.mark.parametrize('cancel', [False, True])
async def test_process_deadline_and_cancellation_reap_child(monkeypatch, cancel):
    created = []
    ready = asyncio.Event()
    original = asyncio.create_subprocess_exec
    async def capture(*args, **kwargs):
        process = await original(*args, **kwargs)
        created.append(process)
        ready.set()
        return process
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', capture)
    task = asyncio.create_task(run_bounded([sys.executable, '-c', 'import time; time.sleep(30)'],
        b'', timeout=10. if cancel else .05, max_output=10))
    await ready.wait()
    if cancel:
        task.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else InvalidInput):
        await task
    assert created[0].returncode is not None


def test_command_configuration_cannot_inject_options_or_paths():
    for language in ('eng --user-words /tmp/x', '../eng', 'https://models/eng', ''):
        with pytest.raises(InvalidInput):
            TesseractOcr(language=language)


async def test_invalid_image_is_refused_before_spawning(monkeypatch):
    async def unexpected(*args, **kwargs):
        pytest.fail('invalid input spawned a process')
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', unexpected)
    with pytest.raises(InvalidInput, match='PNG'):
        await TesseractOcr().recognize(b'not an image')
    output = BytesIO()
    Image.new('RGB', (100, 100), 'white').save(output, format='PNG')
    with pytest.raises(InvalidInput, match='pixel'):
        await TesseractOcr().recognize(output.getvalue(), max_pixels=10)


@pytest.mark.skipif(shutil.which('tesseract') is None, reason='Tesseract executable is optional')
async def test_real_tesseract_returns_text_and_normalized_regions():
    image = Image.new('RGB', (1200, 200), 'white')
    ImageDraw.Draw(image).text((40, 40), 'ProjectScone calibration uses Polaris 12345',
        font=ImageFont.load_default(size=36), fill='black')
    output = BytesIO()
    image.save(output, format='PNG')
    result = await TesseractOcr(page_segmentation=6).recognize(output.getvalue())
    assert 'Polaris' in ' '.join(region.text for region in result.regions)
    assert (result.width, result.height) == image.size
    assert result.regions and all(region.box[0] < region.box[2] for region in result.regions)


@pytest.mark.skipif(sys.platform == 'win32', reason='POSIX process group contract')
async def test_wrapper_descendants_cannot_keep_timeout_cleanup_waiting():
    wrapper = 'import subprocess,sys; subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"])'
    started = asyncio.get_running_loop().time()
    with pytest.raises(InvalidInput, match='wall time'):
        await asyncio.wait_for(run_bounded([sys.executable, '-c', wrapper], b'',
            timeout=.2, max_output=100), 3.)
    assert asyncio.get_running_loop().time() - started < 3.
