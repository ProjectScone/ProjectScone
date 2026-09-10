"""Shared command-line and cleanup support for the bounded Qdrant experiments."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import ipaddress
import json
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from qdrant_client import AsyncQdrantClient


@dataclass(frozen=True)
class Options:
    url: str
    output: Path


class Trial(TypedDict):
    repeat: int
    query: int
    elapsed_ms: float
    actual: list[tuple[int, float]]
    exact_reference: list[tuple[int, float]]


class RecallTrial(Trial):
    recall_at_10: float


def arguments(description: str) -> Options:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--url", required=True, help="Existing loopback-only Qdrant HTTP(S) endpoint")
    parser.add_argument("--output", required=True, type=Path, help="New JSON results path, normally under ignored bench-runs/")
    namespace = parser.parse_args()
    try:
        parsed = urlsplit(namespace.url)
        hostname = parsed.hostname
        loopback = hostname == "localhost" or (hostname is not None and ipaddress.ip_address(hostname).is_loopback)
        if (not loopback or parsed.scheme not in ("http", "https") or parsed.username is not None
                or parsed.password is not None or parsed.query or parsed.fragment):
            raise ValueError("URL must be loopback HTTP(S) without credentials, query, or fragment")
        _ = parsed.port
    except ValueError as error:
        parser.error(f"invalid --url: {error}")
    output = namespace.output.expanduser().resolve()
    if output.exists():
        parser.error("--output already exists; choose a new path to preserve previous measurements")
    return Options(namespace.url, output)


async def finish(client: AsyncQdrantClient, collection: str, creation_started: bool,
                 report: dict[str, object], output: Path) -> None:
    """Preserve partial measurements even when cleanup fails; never delete other names."""
    cleanup_error: Exception | None = None
    report["creation_attempted"] = creation_started
    try:
        if creation_started:
            if await client.collection_exists(collection):
                await client.delete_collection(collection)
            report["deleted"] = not await client.collection_exists(collection)
            if report["deleted"] is not True:
                raise RuntimeError("owned Qdrant collection remains after cleanup")
    except Exception as error:
        report["cleanup_error"] = f"{type(error).__name__}: {error}"
        cleanup_error = error
    finally:
        try:
            await client.close()
        finally:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as destination:
                destination.write(json.dumps(report, indent=2) + "\n")
            print("results", output, "cleanup", report.get("deleted"), flush=True)
    if cleanup_error is not None:
        raise cleanup_error
