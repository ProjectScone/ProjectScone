import pytest
from scone_memory.core.errors import InvalidInput
from scone_memory.runtime.config import Settings


def test_capture_mode_defaults_inline_and_environment_opts_in():
    assert Settings.from_env({}).conversations_capture == 'inline'
    assert Settings.from_env({'SCONE_CONVERSATIONS_CAPTURE':'deferred'}).conversations_capture == 'deferred'


@pytest.mark.parametrize('mode', ['', 'background', 'DEFERRED'])
def test_invalid_capture_mode_is_refused(mode):
    with pytest.raises(InvalidInput, match='SCONE_CONVERSATIONS_CAPTURE'):
        Settings.from_env({'SCONE_CONVERSATIONS_CAPTURE':mode})


def test_deferred_capture_refuses_tool_mode():
    with pytest.raises(InvalidInput, match='capture'):
        Settings(conversations_capture='deferred', conversations_tool_mode='native')

async def test_standard_serve_requires_journal_for_deferred_capture():
    from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
    from scone_memory.api.__main__ import build_app
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    try:
        with pytest.raises(ValueError, match='SCONE_CONVERSATIONS_JOURNAL'):
            build_app(Settings.from_env({'SCONE_API_KEY':'fixture-key','SCONE_CONVERSATIONS_CAPTURE':'deferred'}), engine)
    finally:
        await engine.close()
