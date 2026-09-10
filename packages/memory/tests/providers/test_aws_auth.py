"""Offline signing contracts against botocore; no AWS requests or credentials discovery."""
from datetime import datetime, timezone
from hashlib import sha256

import httpx
import pytest

FIXED_TIME = datetime(2026, 9, 10, 12, 34, 56, tzinfo=timezone.utc)
ACCESS_KEY = "AKIDEXAMPLE"
SECRET_KEY = "example/Secret+Key="
TOKEN = "example/Session+Token="


@pytest.mark.parametrize("service", ["es", "aoss"])
@pytest.mark.parametrize("method,url,body", [
    ("GET", "https://search.example.test/index/_search?b=two&a=2&a=1&empty&space=a%20b", b""),
    ("POST", "https://search.example.test:8443/a%20b/_search", b'{"query":{"match_all":{}}}'),
    ("PUT", "https://search.example.test/index/_bulk?refresh=true", b'{"index":{"_id":"1"}}\n{"x":1}\n'),
])
async def test_signature_matches_botocore(service: str, method: str, url: str, body: bytes, monkeypatch):
    botocore_auth = pytest.importorskip("botocore.auth")
    from botocore.awsrequest import AWSRequest
    from botocore.credentials import Credentials
    from scone_memory.providers.aws_auth import AwsSigV4Auth

    monkeypatch.setattr(botocore_auth, "get_current_datetime", lambda: FIXED_TIME)
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200)

    auth = AwsSigV4Auth(ACCESS_KEY, SECRET_KEY, region="us-east-1", service=service,
                       session_token=TOKEN, clock=lambda: FIXED_TIME)
    async with httpx.AsyncClient(auth=auth, transport=httpx.MockTransport(handle)) as client:
        await client.request(method, url, content=body, headers={"Content-Type": "application/json"})
    signed = seen[0]
    unsigned_headers = {key: value for key, value in signed.headers.items() if key != "authorization"}
    oracle = AWSRequest(method=method, url=str(signed.url), data=body, headers=unsigned_headers)
    botocore_auth.SigV4Auth(Credentials(ACCESS_KEY, SECRET_KEY, TOKEN), service, "us-east-1").add_auth(oracle)
    assert signed.headers["Authorization"] == oracle.headers["Authorization"]
    assert signed.headers["X-Amz-Content-SHA256"] == sha256(body).hexdigest()
    assert signed.headers["X-Amz-Security-Token"] == TOKEN
    assert signed.headers["X-Amz-Date"] == "20260910T123456Z"


async def test_signer_hides_secrets_and_overwrites_stale_signing_headers(caplog):
    from scone_memory.providers.aws_auth import AwsSigV4Auth

    auth = AwsSigV4Auth(ACCESS_KEY, SECRET_KEY, region="us-east-1", clock=lambda: FIXED_TIME)
    with caplog.at_level("DEBUG"):
        request = httpx.Request("GET", "https://example.test/index", headers={
            "authorization": "stale", "x-amz-security-token": "stale-token", "x-amz-date": "stale-date",
            "x-amz-content-sha256": "UNSIGNED-PAYLOAD", "date": "stale-date",
        })
        flow = auth.auth_flow(request)
        signed = next(flow)
    assert "x-amz-security-token" not in signed.headers
    assert "date" not in signed.headers
    assert signed.headers["x-amz-date"] == "20260910T123456Z"
    assert "stale" not in signed.headers["authorization"]
    for secret in (ACCESS_KEY, SECRET_KEY, TOKEN):
        assert secret not in repr(auth)
        assert secret not in caplog.text


@pytest.mark.parametrize("kwargs", [
    {"access_key_id": ""}, {"secret_access_key": ""}, {"secret_access_key": None},
    {"access_key_id": None}, {"region": "us-east-1/evil"},
    {"service": "s3"}, {"session_token": "bad\ntoken"},
])
def test_invalid_signer_configuration_does_not_echo_credentials(kwargs: dict[str, object]):
    from scone_memory.providers.aws_auth import AwsSigV4Auth

    values = {"access_key_id": ACCESS_KEY, "secret_access_key": SECRET_KEY, "region": "us-east-1"} | kwargs
    with pytest.raises(ValueError) as error:
        AwsSigV4Auth(**values)
    assert SECRET_KEY not in str(error.value)


def test_signer_requires_https_before_using_credentials():
    from scone_memory.providers.aws_auth import AwsSigV4Auth

    auth = AwsSigV4Auth(ACCESS_KEY, SECRET_KEY, region="us-east-1")
    with pytest.raises(ValueError, match="HTTPS"):
        next(auth.auth_flow(httpx.Request("GET", "http://localhost:9200")))
