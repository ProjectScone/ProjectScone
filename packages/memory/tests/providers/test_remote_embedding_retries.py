"""A remote embedder re-sends only what a server said "not now" to, only
when asked to, and says how many attempts it made when it gives up."""

import logging

import httpx
import pytest

from scone_memory.embedders import remote
from scone_memory.embedders.remote import MAX_RETRY_WAIT, RemoteEmbedder
from scone_memory.runtime.config import InvalidInput, Settings, build_embedder

OK = {'data': [{'index': 0, 'embedding': [3, 4]}]}


def scripted(*replies):
    """A transport answering each request with the next reply: a status
    code, an exception class to raise, or (status, headers)."""
    requests = []

    def respond(request):
        requests.append(request)
        reply = replies[len(requests) - 1]
        if isinstance(reply, type):
            raise reply('scripted', request=request)
        status, headers = reply if isinstance(reply, tuple) else (reply, {})
        return httpx.Response(status, json=OK if status == 200 else {'error': 'no'}, headers=headers)

    return httpx.MockTransport(respond), requests


@pytest.fixture
def waits(monkeypatch):
    slept = []

    async def sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(remote.asyncio, 'sleep', sleep)
    return slept


async def test_by_default_nothing_is_resent(waits):
    for reply, error in ((503, RuntimeError), (httpx.ReadTimeout, httpx.ReadTimeout)):
        transport, requests = scripted(reply)
        with pytest.raises(error):
            await RemoteEmbedder('https://e.test/v1', 'm', dim=2, transport=transport).embed(['x'])
        assert len(requests) == 1 and waits == []


async def test_a_timeout_a_lost_connection_and_a_503_are_resent_until_one_succeeds(waits, caplog):
    transport, requests = scripted(httpx.ReadTimeout, httpx.ConnectError, 503, 200)
    embedder = RemoteEmbedder('https://e.test/v1', 'm', dim=2, transport=transport, retries=3, retry_wait=0.5)
    with caplog.at_level(logging.WARNING, logger=remote.__name__):
        assert await embedder.embed(['x']) == [[.6, .8]]
    assert len(requests) == 4 and waits == [0.5, 1.0, 2.0]
    retries = [r for r in caplog.records if getattr(r, 'event', None) == 'embedding_call.retry']
    assert [(r.attempt, r.reason) for r in retries] == [(1, 'ReadTimeout'), (2, 'ConnectError'), (3, '503')]


async def test_giving_up_says_how_many_attempts_were_made(waits):
    transport, requests = scripted(429, 429)
    with pytest.raises(RuntimeError, match='returned 429 after 2 attempts'):
        await RemoteEmbedder('https://e.test/v1', 'm', dim=2, transport=transport, retries=1, retry_wait=0).embed(['x'])
    assert len(requests) == 2
    transport, requests = scripted(httpx.ConnectError, httpx.ConnectError, httpx.ConnectError)
    with pytest.raises(RuntimeError, match='failed after 3 attempts: ConnectError'):
        await RemoteEmbedder('https://e.test/v1', 'm', dim=2, transport=transport, retries=2, retry_wait=0).embed(['x'])
    assert len(requests) == 3


async def test_a_refusal_of_the_request_itself_is_not_resent(waits):
    for status in (400, 401, 404, 500):
        transport, requests = scripted(status)
        with pytest.raises(RuntimeError, match=f'returned {status}$'):
            await RemoteEmbedder('https://e.test/v1', 'm', dim=2, transport=transport, retries=3, retry_wait=0).embed(['x'])
        assert len(requests) == 1
    transport, requests = scripted(httpx.UnsupportedProtocol)
    with pytest.raises(httpx.UnsupportedProtocol):
        await RemoteEmbedder('https://e.test/v1', 'm', dim=2, transport=transport, retries=3, retry_wait=0).embed(['x'])
    assert len(requests) == 1 and waits == []


async def test_retry_after_is_honoured_up_to_the_cap(waits, caplog):
    transport, _ = scripted((429, {'Retry-After': '2'}), (503, {'Retry-After': '999'}),
                            (503, {'Retry-After': 'Wed, 21 Oct 2015 07:28:00 GMT'}), (503, {'Retry-After': 'nan'}), 200)
    embedder = RemoteEmbedder('https://e.test/v1', 'm', dim=2, transport=transport, retries=4, retry_wait=0.25)
    with caplog.at_level(logging.WARNING, logger=remote.__name__):
        await embedder.embed(['x'])
    # An HTTP date or a non-number keeps the doubling wait: 0.25 * 2**2, 0.25 * 2**3.
    assert waits == [2.0, MAX_RETRY_WAIT, 1.0, 2.0]
    capped = [r.wait_capped for r in caplog.records if getattr(r, 'event', None) == 'embedding_call.retry']
    assert capped == [False, True, False, False]


@pytest.mark.parametrize('bad', [{'retries': -1}, {'retries': 6}, {'retries': True}, {'retries': 1.0},
                                 {'retry_wait': -1}, {'retry_wait': MAX_RETRY_WAIT + 1}, {'retry_wait': True}])
def test_impossible_retry_settings_are_refused(bad):
    with pytest.raises(ValueError):
        RemoteEmbedder('https://e.test/v1', 'm', **bad)


def test_the_setting_reaches_the_embedder_and_is_checked():
    env = {'SCONE_EMBEDDER': 'remote', 'SCONE_EMBED_URL': 'https://e.test/v1', 'SCONE_EMBED_MODEL': 'm'}
    assert build_embedder(Settings.from_env(env)).retries == 0
    assert build_embedder(Settings.from_env({**env, 'SCONE_EMBED_RETRIES': '2'})).retries == 2
    for bad in ('x', '6', '-1'):
        with pytest.raises(InvalidInput):
            build_embedder(Settings.from_env({**env, 'SCONE_EMBED_RETRIES': bad}))
