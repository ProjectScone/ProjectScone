"""Bounded ingest concurrency: when every slot is embedding, a new write
is told to come back rather than queued without limit, and reads go on."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import httpx

from scone_memory import InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.api import __main__ as serve
from scone_memory.api import create_app
from scone_memory.runtime.config import Settings

AUTH = {"Authorization": "Bearer k"}


class SlowEmbedder:
    """Embeds when told to; until then every embed call waits."""

    id = "slow-8"
    dim = 8

    def __init__(self):
        self.release = asyncio.Event()
        self.calls = 0

    async def embed(self, texts):
        self.calls += 1
        await self.release.wait()
        return [[0.1] * 8 for _ in texts]


@asynccontextmanager
async def client_for(app):
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://scone.test", headers=AUTH) as client:
            yield client


async def test_a_full_ingest_lane_answers_429_with_retry_after_and_reads_go_on():
    embedder = SlowEmbedder()
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), embedder).open()
    app = create_app(engine, {"k": "default"}, ingest_concurrency=1)
    async with client_for(app) as client:
        first = asyncio.create_task(client.post("/v1/episodes", json={"content": "the one being embedded"}))
        async with asyncio.timeout(5):
            while embedder.calls == 0:
                await asyncio.sleep(0.01)
        busy = await asyncio.wait_for(client.post("/v1/episodes", json={"content": "the one that must wait"}), 2)
        assert busy.status_code == 429 and busy.headers["retry-after"] == "1"
        assert busy.json()["code"] == "ingest_busy" and "1 record(s)" in busy.json()["error"]
        batch = await client.post("/v1/episodes/batch", json={"records": [{"content": "batched, also waits"}]})
        assert batch.status_code == 429
        assert (await client.get("/v1/status")).status_code == 200, "reads are not gated by the ingest lane"
        embedder.release.set()
        assert (await first).status_code == 200
        again = await client.post("/v1/episodes", json={"content": "the one that must wait"})
        assert again.status_code == 200 and again.json()["outcome"] == "accepted"
        assert (await client.get("/v1/status")).json()["episodes"] == 2


def test_the_lane_width_is_a_setting_that_serve_passes_through(tmp_path):
    settings = Settings.from_env({"SCONE_API_KEY": "k", "SCONE_INGEST_CONCURRENCY": "2"})
    assert settings.ingest_concurrency == 2
    assert Settings.from_env({"SCONE_API_KEY": "k"}).ingest_concurrency == 4
    engine = asyncio.run(MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), SlowEmbedder()).open())
    assert serve.build_app(settings, engine).state.ingest_lane_width == 2
    composed = Settings.from_env({"SCONE_API_KEY": "k", "SCONE_INGEST_CONCURRENCY": "3",
                                  "SCONE_CONVERSATIONS_JOURNAL": str(tmp_path / "sessions.db")})
    assert serve.build_app(composed, engine).state.ingest_lane_width == 3, "the composed host carries the same lane"
