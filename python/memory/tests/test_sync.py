"""The sync wrapper runs an engine on its own loop in a thread. A store
whose client binds to the loop it was opened on (pymongo, psycopg's pool)
must therefore be opened on that loop, not on a throwaway one."""

from __future__ import annotations

import asyncio
import threading

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine, SyncMemoryEngine
from scone_memory.backends.memory import InMemoryDocumentStore as Reference


class LoopBound(Reference):
    """Behaves like a client that refuses to work on any loop but the one
    it was opened on."""

    name = "loop-bound"

    async def open(self):
        self.loop = asyncio.get_running_loop()
        return self

    async def counts(self, space):
        if asyncio.get_running_loop() is not self.loop:
            raise RuntimeError("cannot use this client in a different event loop")
        return await super().counts(space)


def test_from_env_builds_on_the_loop_that_serves(monkeypatch):
    import scone_memory.runtime.config as config

    async def fake_build(settings):
        documents = await LoopBound().open()
        return await MemoryEngine(documents, InMemoryVectorIndex(), HashEmbedder()).open()

    monkeypatch.setattr(config, "build_engine", fake_build)
    with SyncMemoryEngine.from_env({}) as memory:
        assert memory.status("default").episodes == 0, "the store answers because it lives on the wrapper's loop"


def test_a_prebuilt_engine_still_works_and_a_closed_wrapper_refuses():
    memory = SyncMemoryEngine(MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()))
    memory.remember("default", "kept")
    assert memory.recall("default", "kept").items[0].episode_id == 1
    memory.close()
    with pytest.raises(RuntimeError, match="closed"):
        memory.status("default")


def test_close_while_a_call_is_pending_raises_instead_of_hanging():
    """A caller blocked in a call must not wait forever when another thread
    closes the wrapper mid-flight."""
    import time

    entered = threading.Event()
    cleaned = threading.Event()

    class Slow(InMemoryDocumentStore):
        async def counts(self, space):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0)  # resource cleanup needs another loop turn
                cleaned.set()

    memory = SyncMemoryEngine(MemoryEngine(Slow(), InMemoryVectorIndex(), HashEmbedder()))
    outcome: dict[str, object] = {}

    def caller():
        try:
            memory.status("default")
            outcome["result"] = "returned"
        except RuntimeError as e:
            outcome["result"] = str(e)

    thread = threading.Thread(target=caller, daemon=True)  # daemon: a hang must fail the test, not pin the process
    thread.start()
    assert entered.wait(2), "the call did not enter the store"
    started = time.monotonic()
    memory.close()
    thread.join(timeout=3)
    assert not thread.is_alive(), "the caller is still blocked after close"
    assert "closed while this call was pending" in str(outcome["result"])
    assert cleaned.is_set(), "close returned before pending async cleanup finished"
    assert time.monotonic() - started < 3


def test_timed_out_close_keeps_cleanup_alive_and_rejects_new_calls():
    entered, cleaning = threading.Event(), threading.Event()
    cleaned = threading.Event()
    release = {}

    class Gated(InMemoryDocumentStore):
        async def counts(self, space):
            loop = asyncio.get_running_loop()
            permit = asyncio.Event()
            release["cleanup"] = lambda: loop.call_soon_threadsafe(permit.set)
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                await permit.wait()
                cleaned.set()

    memory = SyncMemoryEngine(MemoryEngine(Gated(), InMemoryVectorIndex(), HashEmbedder()))
    outcome = []

    def caller():
        try:
            memory.status("default")
        except RuntimeError as error:
            outcome.append(str(error))

    caller_thread = threading.Thread(target=caller, daemon=True)
    caller_thread.start()
    try:
        assert entered.wait(2)
        with pytest.raises(TimeoutError, match="shutdown"):
            memory.close(timeout=0.05)
        assert cleaning.is_set()
        assert not cleaned.is_set()
        with pytest.raises(RuntimeError, match="closed|closing"):
            memory.remember("default", "must not be accepted")
        release["cleanup"]()
        memory.close(timeout=2)
        assert cleaned.is_set()
        memory.close()  # idempotent after the first timed-out attempt
        caller_thread.join(2)
        assert not caller_thread.is_alive()
        assert len(outcome) == 1 and "closed while" in outcome[0]
    finally:
        if not cleaned.is_set() and "cleanup" in release:
            try:
                release["cleanup"]()
            except RuntimeError:
                pass
        memory.close()
        caller_thread.join(2)


@pytest.mark.parametrize("fail_open", [False, True])
def test_constructor_failure_closes_its_loop_and_worker(fail_open):
    captured = {}

    class Broken(MemoryEngine):
        async def open(self):
            raise ValueError("opening failed")

    async def builder():
        captured["loop"] = asyncio.get_running_loop()
        captured["thread"] = threading.current_thread()
        if not fail_open:
            raise ValueError("building failed")
        return Broken(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder())

    try:
        with pytest.raises(ValueError, match="failed"):
            SyncMemoryEngine(builder)
        assert not captured["thread"].is_alive(), "failed construction leaked the worker"
        assert captured["loop"].is_closed(), "failed construction leaked the loop"
    finally:
        # Cleanup also makes the regression safe against the pre-fix implementation.
        if captured["thread"].is_alive():
            captured["loop"].call_soon_threadsafe(captured["loop"].stop)
            captured["thread"].join(2)
        if not captured["loop"].is_closed():
            captured["loop"].close()


@pytest.mark.parametrize("operation", ["status", "close"])
def test_calls_from_the_wrappers_loop_are_rejected_without_deadlock(operation):
    wrapper = {}

    class Reentrant(InMemoryDocumentStore):
        async def counts(self, space):
            if operation == "status":
                wrapper["memory"].status(space)
            else:
                wrapper["memory"].close()

    memory = SyncMemoryEngine(MemoryEngine(Reentrant(), InMemoryVectorIndex(), HashEmbedder()))
    wrapper["memory"] = memory
    try:
        with pytest.raises(RuntimeError, match="own event loop"):
            memory.status("default")
        memory.remember("default", "The wrapper still accepts unrelated work after rejection.")
    finally:
        memory.close()


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True, "5"])
def test_invalid_close_timeout_does_not_begin_shutdown(timeout):
    with SyncMemoryEngine(MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder())) as memory:
        with pytest.raises(ValueError, match="timeout"):
            memory.close(timeout=timeout)
        assert memory.status("default").episodes == 0


def test_loop_finalizes_async_generators_and_waits_for_executor_work():
    generator_cleaned = threading.Event()
    executor_entered, executor_release, executor_finished = (threading.Event() for _ in range(3))
    captured = {}

    async def stream():
        try:
            yield "retained iterator"
        finally:
            await asyncio.sleep(0)
            generator_cleaned.set()

    def worker():
        executor_entered.set()
        executor_release.wait(5)
        executor_finished.set()

    async def builder():
        captured["loop"] = asyncio.get_running_loop()
        captured["generator"] = stream()
        await anext(captured["generator"])
        captured["worker"] = asyncio.create_task(asyncio.to_thread(worker))
        return MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder())

    memory = SyncMemoryEngine(builder)
    try:
        assert executor_entered.wait(2)
        with pytest.raises(TimeoutError, match="shutdown"):
            memory.close(timeout=0.05)
        assert generator_cleaned.is_set()
        assert not executor_finished.is_set()
        assert not captured["loop"].is_closed()
        executor_release.set()
        memory.close(timeout=2)
        assert executor_finished.is_set()
        assert captured["loop"].is_closed()
    finally:
        executor_release.set()
        memory.close()


def test_concurrent_closers_share_one_drain_without_recancelling_cleanup():
    entered, cleaning, released, cleaned = (threading.Event() for _ in range(4))

    class Gated(InMemoryDocumentStore):
        async def counts(self, space):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                while not released.is_set():
                    await asyncio.sleep(0.001)
                cleaned.set()

    memory = SyncMemoryEngine(MemoryEngine(Gated(), InMemoryVectorIndex(), HashEmbedder()))
    outcomes = []

    def caller():
        try:
            memory.status("default")
        except RuntimeError:
            pass

    def closer():
        try:
            memory.close(timeout=2)
            outcomes.append("closed")
        except Exception as error:
            outcomes.append(error)

    pending = threading.Thread(target=caller, daemon=True)
    closers = [threading.Thread(target=closer, daemon=True) for _ in range(2)]
    pending.start()
    try:
        assert entered.wait(2)
        with pytest.raises(TimeoutError):
            memory.close(timeout=0.05)
        assert cleaning.is_set()
        for thread in closers:
            thread.start()
        released.set()
        for thread in closers:
            thread.join(3)
        assert outcomes == ["closed", "closed"]
        assert cleaned.is_set(), "a repeated close interrupted cleanup with another cancellation"
    finally:
        released.set()
        memory.close()
        pending.join(2)


def test_executor_finalization_cannot_leave_new_loop_tasks_behind():
    executor_entered, executor_release, late_entered, late_cleaned = (threading.Event() for _ in range(4))
    captured = {}

    async def late_task():
        late_entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            late_cleaned.set()

    def worker():
        executor_entered.set()
        executor_release.wait(5)
        captured["loop"].call_soon_threadsafe(asyncio.create_task, late_task())
        late_entered.wait(2)

    async def builder():
        captured["loop"] = asyncio.get_running_loop()
        captured["worker"] = asyncio.create_task(asyncio.to_thread(worker))
        return MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder())

    memory = SyncMemoryEngine(builder)
    try:
        assert executor_entered.wait(2)
        with pytest.raises(TimeoutError):
            memory.close(timeout=0.05)
        executor_release.set()
        memory.close(timeout=2)
        assert late_entered.is_set()
        assert late_cleaned.is_set(), "successful close abandoned a task created during executor finalization"
        assert not asyncio.all_tasks(captured["loop"])
    finally:
        executor_release.set()
        memory.close()


def test_a_completed_task_delivers_its_result_before_the_loop_stops(monkeypatch):
    """Queue a fast call and shutdown in one tick; Task completion precedes
    the callback that delivers its result to the blocked synchronous caller."""
    captured = {}
    paused, release, accepted = (threading.Event() for _ in range(3))
    outcome = []

    async def builder():
        captured["loop"] = asyncio.get_running_loop()
        return MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder())

    memory = SyncMemoryEngine(builder)
    submit = asyncio.run_coroutine_threadsafe

    def observe_submission(coro, loop):
        future = submit(coro, loop)
        captured["future"] = future
        accepted.set()
        return future

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", observe_submission)

    def pause_loop():
        paused.set()
        release.wait(3)

    def caller():
        try:
            outcome.append(memory.status("default").episodes)
        except RuntimeError as error:
            outcome.append(str(error))

    pending = threading.Thread(target=caller, daemon=True)
    try:
        captured["loop"].call_soon_threadsafe(pause_loop)
        assert paused.wait(2)
        pending.start()
        assert accepted.wait(2)
        with pytest.raises(TimeoutError):
            memory.close(timeout=0.05)
        release.set()
        memory.close(timeout=2)
        pending.join(0.5)
        assert not pending.is_alive(), "completed call lost its result-delivery callback"
        assert outcome == [0], "an already completed call must retain its actual result"
    finally:
        release.set()
        memory.close()
        future = captured.get("future")
        if future is not None and not future.done():
            # Release a caller stranded by the unfixed implementation, without
            # hiding the assertion above or leaking a test thread.
            future.set_exception(RuntimeError("test cleanup after lost callback"))
        if pending.ident is not None:
            pending.join(2)
