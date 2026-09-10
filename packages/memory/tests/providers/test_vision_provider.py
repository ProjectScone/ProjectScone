"""Real image validation and a mocked local model; no inference/downloads."""

import base64
import hashlib
import io
import json

import httpx
import pytest
from PIL import Image

from scone_memory.providers.llm import ChatError
from scone_memory.providers.vision import OpenAICompatibleVision


def pixels(format="PNG", size=(3, 2)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color="red").save(buffer, format=format)
    return buffer.getvalue()


def answer(text="A red rectangle.", finish_reason="stop"):
    return httpx.Response(200, json={"choices": [{
        "finish_reason": finish_reason, "message": {"content": text},
    }]})


@pytest.mark.parametrize(("format", "mime"), [("PNG", "image/png"), ("JPEG", "image/jpeg"), ("WEBP", "image/webp")])
async def test_image_and_task_are_sent_together_and_result_keeps_exact_provenance(format, mime):
    data = pixels(format)
    calls = []

    def handle(request):
        calls.append(request)
        return answer()

    model = OpenAICompatibleVision("http://127.0.0.1:8080/v1", "local-vision", api_key="test-secret",
                                   transport=httpx.MockTransport(handle))
    result = await model.describe(data, mime, prompt="Describe the colors.", source="upload:slide.png")
    assert result.text == "A red rectangle."
    assert result.attachment_id == hashlib.sha256(data).hexdigest()
    assert result.source == "upload:slide.png"
    assert (result.width, result.height, result.media_type, result.model) == (3, 2, mime, "local-vision")
    assert len(calls) == 1
    request = calls[0]
    assert str(request.url) == "http://127.0.0.1:8080/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer test-secret"
    body = json.loads(request.content)
    assert body["stream"] is False
    assert body["messages"][1]["content"] == [
        {"type": "text", "text": "Describe the colors."},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"}},
    ]


@pytest.mark.parametrize(("data", "mime", "options"), [
    (b"", "image/png", {}),
    (b"\x89PNG\r\n\x1a\nfake pixels", "image/png", {}),
    (pixels(), "image/jpeg", {}),
    (pixels(), "image/svg+xml", {}),
    (pixels(), "application/pdf", {}),
    (pixels(), "image/png", {"max_image_bytes": 12}),
    (pixels(size=(10, 10)), "image/png", {"max_image_pixels": 50}),
    (pixels()[:-10], "image/png", {}),
])
async def test_bad_images_are_rejected_before_transport(data, mime, options):
    def forbidden(request):
        pytest.fail("invalid image reached model")

    model = OpenAICompatibleVision("http://localhost:8080/v1", "vision", transport=httpx.MockTransport(forbidden), **options)
    with pytest.raises(ValueError):
        await model.describe(data, mime, prompt="Describe.")


async def test_animated_images_are_rejected():
    buffer = io.BytesIO()
    Image.new("RGB", (3, 2), "red").save(buffer, format="PNG", save_all=True,
                                      append_images=[Image.new("RGB", (3, 2), "blue")])
    model = OpenAICompatibleVision("http://localhost/v1", "vision")
    with pytest.raises(ValueError, match="single.frame"):
        await model.describe(buffer.getvalue(), "image/png", prompt="Describe.")


@pytest.mark.parametrize("options", [{"prompt": ""}, {"prompt": "x" * 16_001},
                                    {"prompt": "Describe", "attachment_id": "a" * 64},
                                    {"prompt": "Describe", "source": "x" * 4097}])
async def test_invalid_task_or_mismatched_attachment_rejected(options):
    model = OpenAICompatibleVision("http://localhost/v1", "vision")
    with pytest.raises(ValueError):
        await model.describe(pixels(), "image/png", **options)


@pytest.mark.parametrize("response", [
    httpx.Response(500, text="secret image echoed"),
    httpx.Response(302, headers={"location": "https://example.com/v1"}),
    httpx.Response(200, content=b"not json"),
    httpx.Response(200, json={"choices": []}),
    answer(""), answer(None), answer("truncated", "length"), answer("blocked", "content_filter"),
])
async def test_invalid_or_incomplete_results_fail_without_exposing_body(response):
    calls = []

    def handle(request):
        calls.append(request)
        return response

    model = OpenAICompatibleVision("http://localhost/v1", "vision", transport=httpx.MockTransport(handle))
    with pytest.raises(ChatError) as raised:
        await model.describe(pixels(), "image/png", prompt="Describe.")
    assert "secret image echoed" not in str(raised.value)
    assert len(calls) == 1


async def test_timeout_is_typed_without_exposing_request():
    def handle(request):
        raise httpx.ReadTimeout("secret request", request=request)

    model = OpenAICompatibleVision("http://localhost/v1", "vision", transport=httpx.MockTransport(handle))
    with pytest.raises(ChatError, match="ReadTimeout") as raised:
        await model.describe(pixels(), "image/png", prompt="Describe.")
    assert "secret request" not in str(raised.value)


async def test_response_byte_limit_stops_reading():
    class TooLarge(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"x" * 129
            pytest.fail("read continued past response limit")

    model = OpenAICompatibleVision("http://localhost/v1", "vision", max_response_bytes=128,
                                   transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=TooLarge())))
    with pytest.raises(ChatError, match="limit"):
        await model.describe(pixels(), "image/png", prompt="Describe.")


@pytest.mark.parametrize("url", ["https://api.openai.com/v1", "http://169.254.169.254", "http://user:key@localhost/v1"])
def test_nonlocal_or_credential_bearing_endpoints_rejected(url):
    with pytest.raises(ValueError):
        OpenAICompatibleVision(url, "vision")


@pytest.mark.parametrize("options", [{"timeout": 0}, {"timeout": float("inf")},
                                    {"max_image_bytes": True}, {"max_image_pixels": -1},
                                    {"max_response_bytes": 0}, {"max_tokens": 0}])
def test_invalid_configuration_rejected(options):
    with pytest.raises(ValueError):
        OpenAICompatibleVision("http://localhost/v1", "vision", **options)
