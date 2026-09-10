"""Typing guards preserve valid native objects accepted by public boundaries."""
from decimal import Decimal

from scone_memory.observability.payload import integer, number
from scone_memory.pipeline.voice import _closing_stream
from scone_memory.retrieval.filters import Condition


async def test_voice_stream_proxy_closes_its_underlying_generator():
    closed = []

    async def generate():
        try:
            yield "reply"
            yield "unused"
        finally:
            closed.append(True)

    class Proxy:
        def __init__(self, stream):
            self.stream = stream

        def __aiter__(self):
            return self

        async def __anext__(self):
            return await anext(self.stream)

        def __getattr__(self, name):
            return getattr(self.stream, name)

    async with _closing_stream(Proxy(generate())) as stream:
        assert await anext(stream) == "reply"
    assert closed == [True]


def test_native_list_filter_matches_and_keeps_parameterized_sql():
    condition = Condition("team", "in", ["blue", "red"])
    assert condition.matches({"team": "blue"})
    assert not condition.matches({"team": "green"})
    sql, parameters = condition.to_sql("metadata")
    assert "IN (?, ?)" in sql
    assert parameters == ["team", "blue", "red"]


def test_decimal_event_values_keep_native_numeric_conversion():
    assert number(Decimal("1.25")) == 1.25
    assert integer(Decimal("12")) == 12
