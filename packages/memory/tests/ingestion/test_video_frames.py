"""Native frame selection must cite decoded timestamps, never nominal FPS."""
from fractions import Fraction
import json

import pytest

from scone_memory.core.errors import InvalidInput
from scone_memory.ingestion.video_frames import VideoFramePolicy, plan_frames


def inventory(timestamps, *, start=0, time_base='1/10', duration=30):
    return json.dumps({'streams': [{'index': 0, 'time_base': time_base, 'start_pts': start,
                                   'duration_ts': duration, 'width': 64, 'height': 32}],
                       'frames': [{'stream_index': 0, 'best_effort_timestamp': pts,
                                   'duration': 1, 'width': 64, 'height': 32} for pts in timestamps]}).encode()


def test_vfr_selects_actual_pts_and_coalesces_requests_for_same_frame():
    plan = plan_frames(inventory([50, 54, 77, 79], start=50), VideoFramePolicy(interval_seconds=1))
    assert plan.time_base == Fraction(1, 10)
    assert [(frame.ordinal, frame.presentation_timestamp, frame.requested_seconds)
            for frame in plan.frames] == [(0, 50, (0,)), (2, 77, (1, 2))]
    assert plan.start_timestamp == 50
    assert plan.duration == Fraction(3)


def test_sparse_video_does_not_fabricate_a_frame_after_last_pts():
    plan = plan_frames(inventory([0, 2], duration=100), VideoFramePolicy(interval_seconds=1))
    assert len(plan.frames) == 1
    assert plan.unavailable_requests == 9


@pytest.mark.parametrize('timestamps', [[2, 1], [1, 1], [], [True], ['1'], [None]])
def test_invalid_decoded_timestamps_are_refused(timestamps):
    with pytest.raises(InvalidInput):
        plan_frames(inventory(timestamps), VideoFramePolicy())


def test_budget_exhaustion_refuses_instead_of_truncating():
    with pytest.raises(InvalidInput, match='frame count'):
        plan_frames(inventory([0, 10, 20]), VideoFramePolicy(interval_seconds=1, max_frames=2))
    with pytest.raises(InvalidInput, match='duration'):
        plan_frames(inventory([0], duration=6010), VideoFramePolicy())


@pytest.mark.parametrize('time_base', ['0/1', '-1/10', '1/0', 'nan', '1e9', '1/2/3'])
def test_invalid_time_base_is_refused(time_base):
    with pytest.raises(InvalidInput):
        plan_frames(inventory([0], time_base=time_base), VideoFramePolicy())


@pytest.fixture
def native_tools():
    import shutil
    ffmpeg, ffprobe = shutil.which('ffmpeg'), shutil.which('ffprobe')
    if not ffmpeg or not ffprobe:
        pytest.skip('native ffmpeg and ffprobe required')
    pytest.importorskip('PIL')
    return ffmpeg, ffprobe


@pytest.mark.parametrize('variant', ['constant', 'vfr', 'offset', 'rotation'])
async def test_native_frames_match_pts_pixels_and_source(tmp_path, native_tools, variant):
    import hashlib
    import io
    import subprocess
    from PIL import Image
    from scone_memory.ingestion.video_frames import VideoFrameDecoder

    ffmpeg, ffprobe = native_tools
    path = tmp_path / 'source.mp4'
    filters = []
    if variant == 'vfr':
        filters = ['-vf', "select='eq(n,0)+eq(n,1)+eq(n,5)'", '-fps_mode', 'vfr']
    elif variant == 'offset':
        filters = ['-output_ts_offset', '7']
    subprocess.run([ffmpeg, '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=64x32:rate=2:duration=3',
                    *filters, '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(path)], check=True)
    if variant == 'rotation':
        rotated = tmp_path / 'rotated.mp4'
        help_text = subprocess.run([ffmpeg, '-h', 'full'], capture_output=True, check=True).stdout
        if b'-display_rotation' in help_text:
            command = [ffmpeg, '-v', 'error', '-display_rotation:v:0', '90', '-i', str(path), '-c', 'copy', str(rotated)]
        else:
            command = [ffmpeg, '-v', 'error', '-i', str(path), '-c', 'copy', '-metadata:s:v:0', 'rotate=90', str(rotated)]
        subprocess.run(command, check=True)
        path = rotated
    source = path.read_bytes()
    decoded = await VideoFrameDecoder(ffmpeg_path=ffmpeg, ffprobe_path=ffprobe).sample(
        source, path.name, policy=VideoFramePolicy(interval_seconds=1))
    assert decoded.source_sha256 == hashlib.sha256(source).hexdigest()
    assert len(decoded.decoder_revision) == 64
    assert decoded.policy_revision == 'decoded-video-frames-v1'
    expected_times = [Fraction(0), Fraction(5, 2)] if variant == 'vfr' else [Fraction(0), Fraction(1), Fraction(2)]
    actual_times = [(frame.selection.presentation_timestamp - decoded.plan.start_timestamp) * decoded.plan.time_base
                    for frame in decoded.frames]
    assert actual_times == expected_times
    assert decoded.plan.start_timestamp * decoded.plan.time_base == (7 if variant == 'offset' else 0)
    expected_size = (32, 64) if variant == 'rotation' else (64, 32)
    all_pixels = subprocess.run([ffmpeg, '-v', 'error', '-i', str(path), '-an', '-fps_mode', 'passthrough',
                                 '-pix_fmt', 'rgb24', '-f', 'rawvideo', 'pipe:1'],
                                capture_output=True, check=True).stdout
    bytes_per_frame = expected_size[0] * expected_size[1] * 3
    for frame in decoded.frames:
        expected_size = (32, 64) if variant == 'rotation' else (64, 32)
        with Image.open(io.BytesIO(frame.png)) as image:
            assert image.size == expected_size == (frame.width, frame.height)
            offset = frame.selection.ordinal * bytes_per_frame
            assert image.convert('RGB').tobytes() == all_pixels[offset:offset + bytes_per_frame]
        assert frame.sha256 == hashlib.sha256(frame.png).hexdigest()
    again = await VideoFrameDecoder(ffmpeg_path=ffmpeg, ffprobe_path=ffprobe).sample(
        source, path.name, policy=VideoFramePolicy(interval_seconds=1))
    assert again == decoded


async def test_native_audio_only_is_refused(tmp_path, native_tools):
    import subprocess
    from scone_memory.ingestion.video_frames import VideoFrameDecoder
    ffmpeg, ffprobe = native_tools
    path = tmp_path / 'audio.mp4'
    subprocess.run([ffmpeg, '-v', 'error', '-f', 'lavfi', '-i', 'sine=duration=1', '-c:a', 'aac', str(path)], check=True)
    with pytest.raises(InvalidInput, match='video stream'):
        await VideoFrameDecoder(ffmpeg_path=ffmpeg, ffprobe_path=ffprobe).sample(path.read_bytes(), path.name)


def test_older_ffprobe_frame_duration_preserves_final_frame_coverage():
    data = json.loads(inventory([0, 10, 20]))
    del data['streams'][0]['duration_ts']
    for frame in data['frames']:
        frame['pkt_duration'] = frame.pop('duration')
    plan = plan_frames(json.dumps(data).encode(), VideoFramePolicy(interval_seconds=1))
    assert plan.duration == Fraction(21, 10)
    assert len(plan.frames) == 3


@pytest.mark.parametrize('mutation', ['wrong_stream', 'pixels', 'missing_pts', 'wrong_start', 'bad_json'])
def test_inventory_evidence_must_be_consistent(mutation):
    data = json.loads(inventory([0, 10, 20]))
    if mutation == 'wrong_stream':
        data['frames'][1]['stream_index'] = 1
    elif mutation == 'pixels':
        data['frames'][1]['width'] = 20_000_001
    elif mutation == 'missing_pts':
        del data['frames'][1]['best_effort_timestamp']
    elif mutation == 'wrong_start':
        data['streams'][0]['start_pts'] = 1
    raw = b'{' if mutation == 'bad_json' else json.dumps(data).encode()
    with pytest.raises(InvalidInput):
        plan_frames(raw, VideoFramePolicy())


def test_png_framing_rejects_corruption_and_declared_budgets():
    import io
    from PIL import Image
    from scone_memory.ingestion.video_frames import _pngs
    stream = io.BytesIO()
    Image.new('RGB', (3, 2), 'red').save(stream, format='PNG')
    png = stream.getvalue()
    assert _pngs(png + png, VideoFramePolicy()) == (png, png)
    corrupted = bytearray(png)
    corrupted[-1] ^= 1
    for raw in (bytes(corrupted), png[:-1], b'no image', png + b'trailing'):
        with pytest.raises(InvalidInput):
            _pngs(raw, VideoFramePolicy())
    for policy in (VideoFramePolicy(max_frames=1), VideoFramePolicy(max_pixels=1),
                   VideoFramePolicy(max_frame_bytes=len(png) - 1),
                   VideoFramePolicy(max_total_bytes=len(png))):
        with pytest.raises(InvalidInput):
            _pngs(png + png, policy)


async def test_native_pixel_preflight_stops_before_frame_decode(tmp_path, native_tools, monkeypatch):
    import subprocess
    from scone_memory.ingestion import video_frames
    ffmpeg, ffprobe = native_tools
    path = tmp_path / 'source.mkv'
    subprocess.run([ffmpeg, '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=64x32:rate=2:duration=2',
                    '-an', '-c:v', 'ffv1', str(path)], check=True)
    labels = []
    original = video_frames.run_bounded

    async def capture(*args, **kwargs):
        labels.append(kwargs.get('label'))
        return await original(*args, **kwargs)

    monkeypatch.setattr(video_frames, 'run_bounded', capture)
    decoder = video_frames.VideoFrameDecoder(ffmpeg_path=ffmpeg, ffprobe_path=ffprobe)
    with pytest.raises(InvalidInput, match='pixel'):
        await decoder.sample(path.read_bytes(), path.name, policy=VideoFramePolicy(max_pixels=1))
    assert labels == ['video decoder identity', 'video decoder identity', 'video stream inspection']
    result = await decoder.sample(path.read_bytes(), path.name, policy=VideoFramePolicy(interval_seconds=1))
    assert len(result.frames) == 2
    assert result.plan.duration == 2


@pytest.mark.parametrize('name', ['video.ts', 'video.txt'])
async def test_video_sampler_does_not_take_typescript_or_other_formats(native_tools, name):
    from scone_memory.ingestion.video_frames import VideoFrameDecoder
    ffmpeg, ffprobe = native_tools
    with pytest.raises(InvalidInput, match='extension'):
        await VideoFrameDecoder(ffmpeg_path=ffmpeg, ffprobe_path=ffprobe).sample(b'not video', name)


@pytest.mark.parametrize('changes', [{'max_frames': 257}, {'interval_seconds': 0}, {'max_pixels': 20_000_001}])
def test_copied_policy_cannot_bypass_public_planner_limits(changes):
    copied = VideoFramePolicy().model_copy(update=changes)
    with pytest.raises(InvalidInput, match='policy'):
        plan_frames(inventory(range(257), time_base='1/1', duration=300), copied)


@pytest.mark.parametrize('configuration', ['policy', 'limits'])
async def test_copied_settings_are_refused_before_decoder_processes(native_tools, monkeypatch, configuration):
    from scone_memory.ingestion import video_frames
    from scone_memory.ingestion.formats.types import DocumentLimits
    ffmpeg, ffprobe = native_tools

    async def forbidden(*args, **kwargs):
        pytest.fail('invalid settings must be refused before subprocess creation')

    monkeypatch.setattr(video_frames, 'run_bounded', forbidden)
    policy = VideoFramePolicy().model_copy(update={'max_frames': 257}) if configuration == 'policy' else None
    limits = DocumentLimits().model_copy(update={'timeout_seconds': float('nan')}) if configuration == 'limits' else None
    with pytest.raises(InvalidInput, match='policy|limits'):
        await video_frames.VideoFrameDecoder(ffmpeg_path=ffmpeg, ffprobe_path=ffprobe).sample(
            b'bounded input', 'source.mp4', policy=policy, limits=limits)
