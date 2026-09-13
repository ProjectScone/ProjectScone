"""The reply an agent is writing, delivered to a reader as it is written.

The evidence loop asks the model for turns until one comes back without
tool calls; that turn's content is the answer. With a public-text sink,
the loop hands the model the sink for each turn, so a streaming model's
content deltas reach the reader as they arrive; text streamed in a turn
that then called tools was not the answer and is withdrawn; a model that
cannot stream, or a structured answer, delivers the answer as one delta
once it is known. Nothing but content is ever delivered.
"""

from __future__ import annotations

import pytest

from scone_memory.agents.evidence_loop import EvidenceToolLoop, ToolCall, ToolStep
from scone_memory.integrations.scoped_tools import ScopedMemoryTools
from scone_memory.retrieval.recall_scope import RecallScope

pytestmark = pytest.mark.asyncio


class Streaming:
    """A model that writes its content through the sink it is handed."""

    def __init__(self, *turns: tuple[list[str], ToolStep]) -> None:
        self.turns = list(turns)
        self.handed: list[bool] = []

    async def complete(self, messages, tools, *, on_public_text=None):
        self.handed.append(on_public_text is not None)
        pieces, step = self.turns.pop(0)
        if on_public_text is not None:
            for piece in pieces:
                await on_public_text(piece)
        return step


class Plain:
    """A model with no notion of streaming."""

    def __init__(self, *steps: ToolStep) -> None:
        self.steps = list(steps)

    async def complete(self, messages, tools):
        return self.steps.pop(0)


class Recorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, str | None]] = []

    async def append(self, text: str) -> None:
        self.events.append(('text', text))

    def withdraw(self) -> None:
        self.events.append(('withdraw', None))


def binding(engine):
    return ScopedMemoryTools(engine, 'alpha', scope=RecallScope.validated(where={'team': 'blue'}),
                             exclude_session_id='current')


QUESTION = [{'role': 'user', 'content': 'What did we decide?'}]
SEARCH = ToolStep(calls=(ToolCall(id='s1', name='search_memory', arguments={'query': 'decision'}),))


async def test_a_streaming_model_delivers_the_answer_as_it_is_written(engine):
    sink = Recorder()
    model = Streaming((['Hel', 'lo'], ToolStep(content='Hello')))
    result = await EvidenceToolLoop(model, binding(engine), public_text=sink).run(QUESTION)
    assert result.text == 'Hello'
    assert sink.events == [('text', 'Hel'), ('text', 'lo')]
    assert model.handed == [True]


async def test_text_written_before_a_tool_turn_is_withdrawn_and_the_answer_follows(engine):
    sink = Recorder()
    model = Streaming((['Looking that up.'], SEARCH), (['We ', 'decided.'], ToolStep(content='We decided.')))
    result = await EvidenceToolLoop(model, binding(engine), public_text=sink).run(QUESTION)
    assert result.text == 'We decided.'
    assert sink.events == [('text', 'Looking that up.'), ('withdraw', None), ('text', 'We '), ('text', 'decided.')]


async def test_a_tool_turn_that_wrote_nothing_withdraws_nothing(engine):
    sink = Recorder()
    model = Streaming(([], SEARCH), (['Done.'], ToolStep(content='Done.')))
    await EvidenceToolLoop(model, binding(engine), public_text=sink).run(QUESTION)
    assert sink.events == [('text', 'Done.')]


async def test_a_model_that_cannot_stream_delivers_the_answer_once_it_is_known(engine):
    sink = Recorder()
    result = await EvidenceToolLoop(Plain(SEARCH, ToolStep(content='Hello')), binding(engine), public_text=sink).run(QUESTION)
    assert result.text == 'Hello' and sink.events == [('text', 'Hello')]


async def test_without_a_sink_a_streaming_model_is_not_handed_one(engine):
    model = Streaming((['x'], ToolStep(content='Hello')))
    result = await EvidenceToolLoop(model, binding(engine)).run(QUESTION)
    assert result.text == 'Hello' and model.handed == [False]


async def test_a_sink_that_fails_fails_the_turn_and_nothing_more_is_delivered(engine):
    class Failing(Recorder):
        async def append(self, text: str) -> None:
            raise RuntimeError('observation closed')

    sink = Failing()
    model = Streaming((['Hel', 'lo'], ToolStep(content='Hello')))
    with pytest.raises(RuntimeError):
        await EvidenceToolLoop(model, binding(engine), public_text=sink).run(QUESTION)
    assert sink.events == []


async def test_a_sink_that_fails_on_the_one_delta_path_fails_the_turn(engine):
    class Failing(Recorder):
        async def append(self, text: str) -> None:
            raise RuntimeError('observation closed')

    with pytest.raises(RuntimeError):
        await EvidenceToolLoop(Plain(ToolStep(content='Hello')), binding(engine), public_text=Failing()).run(QUESTION)


async def test_what_is_delivered_is_the_reply_and_only_the_reply(engine):
    """The sum of deltas is the accepted text, byte for byte."""
    sink = Recorder()
    pieces = ['A ', 'reply ', 'in ', 'pieces.']
    result = await EvidenceToolLoop(Streaming((pieces, ToolStep(content=''.join(pieces)))), binding(engine), public_text=sink).run(QUESTION)
    assert ''.join(text for kind, text in sink.events if kind == 'text' and text) == result.text


async def test_a_bound_agent_hands_the_sink_to_its_loop(engine):
    from scone_memory.agents.catalog import AgentCatalog, AgentDefinition, AgentModel

    sink = Recorder()
    catalog = AgentCatalog(
        models=[AgentModel('local', 'Local model', '1', lambda: Streaming((['We ', 'decided.'], ToolStep(content='We decided.'))))],
        agents=[AgentDefinition(agent_id='research', instructions='Answer from evidence.', models=('local',),
                                default_model='local', initial_search=False)])
    result = await catalog.bind('research', model_id='local').run('What did we decide?', tools=binding(engine), public_text=sink)
    assert result.output.text == 'We decided.'
    assert sink.events == [('text', 'We '), ('text', 'decided.')]


async def test_a_sink_that_is_not_one_is_refused_before_any_model_call(engine):
    with pytest.raises(ValueError):
        EvidenceToolLoop(Plain(ToolStep(content='x')), binding(engine), public_text=object())  # type: ignore[arg-type]
