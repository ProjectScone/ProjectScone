"""``scone-memory`` / ``python -m scone_memory.api``: serve the engine
described by the environment. Refuses to start without a key, because
an unauthenticated memory server is a leak waiting for a port scan."""

from __future__ import annotations

import asyncio
import sys
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..runtime.agent_runtime import AgentRuntime
    from ..ingestion.document_media import DocumentMedia
    from ..ingestion.document_ocr import DocumentOcr
    from ..ingestion.import_service import DocumentImportService
    from ..ingestion.directory_service import DirectorySyncService
    from ..realtime.catalog import PersonaCatalog
    from ..runtime.model_runtime import DynamicLocalCatalog

from ..runtime.config import Settings, build_engine, build_worker
from ..runtime.model_connections import ModelConnectionError
from ..agents.workflow import WorkflowError
from .app import create_app


def build_app(settings: Settings, engine, *, document_media: DocumentMedia | None = None):
    from ..runtime.agent_runtime import load_agent_runtime
    from ..runtime.document_jobs import load_document_imports
    from ..runtime.directory_sync import load_directory_sync
    from ..runtime.document_ocr import build_document_ocr

    agents = load_agent_runtime(settings.agents_config, engine) if settings.agents_config else None
    imports = None
    directory_sync = None
    try:
        if document_media is None and settings.document_media_config:
            from ..runtime.document_media import load_document_media
            document_media = load_document_media(settings.document_media_config)
        ocr = build_document_ocr(settings)
        if settings.document_jobs_config:
            imports = load_document_imports(settings.document_jobs_config, engine, document_ocr=ocr, document_media=document_media,
                ocr_identity=f'{settings.document_ocr_executable}:{settings.document_ocr_language}:{settings.document_ocr_psm}')
        if settings.directory_sync_config:
            directory_sync = load_directory_sync(settings.directory_sync_config, engine,
                document_ocr=ocr, document_media=document_media,
                ocr_identity=f'{settings.document_ocr_executable}:{settings.document_ocr_language}:{settings.document_ocr_psm}')
        app = _build_app(settings, engine, agents, document_ocr=ocr, document_import_service=imports,
                         document_media=document_media, directory_sync_service=directory_sync)
        return agents.own(app) if agents is not None else app
    except BaseException:
        try:
            if directory_sync is not None:
                directory_sync.close_idle()
        finally:
            try:
                if imports is not None:
                    imports.close_idle()
            finally:
                if agents is not None:
                    agents.close_idle()
        raise


def _build_app(settings: Settings, engine, agents: AgentRuntime | None = None, *,
               document_ocr: DocumentOcr | None = None,
               document_import_service: DocumentImportService | None = None,
               directory_sync_service: DirectorySyncService | None = None,
               document_media: DocumentMedia | None = None):
    """The app ``serve`` runs: the memory API alone, or the conversation
    service composed over it on the same origin when the settings name a
    journal (SCONE_CONVERSATIONS_JOURNAL). Raises ValueError for a journal
    or model factory the operator got wrong, before anything is served."""
    from ..runtime.diagnostics import install_http_diagnostics
    store = None
    model_management = False
    vision_available = None
    if settings.model_connections:
        from ..runtime.model_connections import ModelConnectionStore
        from ..runtime.model_runtime import LocalModelWorker, connection_defaults, local_admin_enabled

        store = ModelConnectionStore(settings.model_connections, connection_defaults(settings))
        worker = LocalModelWorker(engine, settings, store)
        model_management = local_admin_enabled(settings)
        vision_available = lambda: store.get('vision') is not None
    else:
        worker = build_worker(engine, settings, settings.keys.values())

    def finish(app):
        if store is not None:
            from .model_connections import mount_model_connection_routes
            from ..runtime.model_runtime import authorize_local_admin

            app.state.model_connections = store
            target = getattr(app.state, 'memory_app', app)
            target.state.model_connections = store
            mount_model_connection_routes(target, store, authorize_local_admin(settings), on_change=worker.refresh)
            from .image_understanding import mount_image_understanding_routes
            from ..runtime.model_runtime import authorize_image_write, local_vision_factory

            mount_image_understanding_routes(target, engine, authorize_image_write(settings, engine),
                                             local_vision_factory(store))
        install_http_diagnostics(app)
        return app

    if not settings.conversations_journal:
        return finish(create_app(engine, settings.keys, worker=worker,
                          document_ocr=document_ocr, document_import_service=document_import_service, document_media=document_media,
                          directory_sync_service=directory_sync_service,
                          agent_catalog=agents.catalog if agents else None,
                          agent_plan_store=agents.plans if agents else None,
                          agent_run_service=agents.service if agents else None,
                          ingest_concurrency=settings.ingest_concurrency, roles=settings.roles,
                          model_connections_available=model_management, vision_available=vision_available))
    from .conversation_server import journal_path, load_model_factory
    from .conversations import create_conversation_app
    from ..runtime.conversation_review import build_conversation_review
    from ..runtime.conversation_retrieval import build_adaptive_retrieval
    from ..runtime.conversation_tools import build_conversation_tools

    journal = journal_path(settings, settings.conversations_journal)
    answer_review = build_conversation_review(settings)
    adaptive_retriever = build_adaptive_retrieval(settings, engine)
    conversation_tools = build_conversation_tools(settings)
    catalog: PersonaCatalog | DynamicLocalCatalog | None = None
    if settings.conversations_personas:
        if not settings.conversations_registry:
            raise ValueError("SCONE_CONVERSATIONS_REGISTRY is required with a persona catalog")
        from ..realtime.catalog import bind_catalog, load_personas
        from ..realtime.providers import ProviderRegistry

        # The same trust rule as a model factory; this one is called once,
        # here, and may close over the operator's provider credentials.
        registry = load_model_factory(settings.conversations_registry)()
        if not isinstance(registry, ProviderRegistry):
            raise ValueError("SCONE_CONVERSATIONS_REGISTRY must return a ProviderRegistry")
        catalog = bind_catalog(load_personas(settings.conversations_personas), registry)
    elif store is not None:
        from ..runtime.model_runtime import DynamicLocalCatalog

        catalog = DynamicLocalCatalog(store, think=settings.chat_think, tools=conversation_tools)
    scoped = None
    runtime_available = None
    if settings.conversations_model_factory:
        factory = load_model_factory(settings.conversations_model_factory)
        from ..realtime.text import TextConversation

        def scoped(space, sid, scope, **conversation_options):
            return TextConversation(engine, space, sid, factory, turn_timeout=settings.chat_timeout,
                                    **scope.kwargs(), **conversation_options)
    elif store is not None:
        from ..runtime.model_runtime import local_text_runtime

        scoped = local_text_runtime(engine, store, think=settings.chat_think, tools=conversation_tools)
        runtime_available = lambda: store.get('chat') is not None
    return finish(create_conversation_app(engine, settings.keys, journal, None, scoped_runtime_factory=scoped,
                                   public_text_streaming=scoped is not None or catalog is not None,
                                   worker=worker, catalog=catalog,
                                   document_ocr=document_ocr, document_import_service=document_import_service, document_media=document_media,
                                   directory_sync_service=directory_sync_service,
                                   agent_catalog=agents.catalog if agents else None,
                                   agent_plan_store=agents.plans if agents else None,
                                   agent_run_service=agents.service if agents else None,
                                   ingest_concurrency=settings.ingest_concurrency, roles=settings.roles,
                                   runtime_available=runtime_available,
                                   model_connections_available=model_management, vision_available=vision_available,
                                   answer_review=answer_review, adaptive_retriever=adaptive_retriever,
                                   tool_retrieval=conversation_tools))


def build_server(settings: Settings, app):
    """The uvicorn server for ``app``. A composed host must tell the
    conversation service to end its open streams before uvicorn waits for
    open responses, or a reader holding a stream holds shutdown; the
    memory-only app has no streams to end."""
    import uvicorn

    if settings.conversations_journal:
        from .conversation_server import create_server

        return create_server(app, host=settings.host, port=settings.port)
    return uvicorn.Server(uvicorn.Config(app, host=settings.host, port=settings.port, log_level="warning"))


def main(settings: Optional[Settings] = None) -> None:
    settings = settings or Settings.from_env()
    if not settings.keys:
        print("refusing to serve without a key: set SCONE_API_KEY or SCONE_API_KEYS", file=sys.stderr)
        sys.exit(2)
    try:
        import uvicorn
    except ImportError:
        print("serving needs uvicorn: pip install 'scone-memory[api]'", file=sys.stderr)
        sys.exit(2)
    async def run() -> None:
        from ..runtime.diagnostics import configure_diagnostics

        configure_diagnostics(settings.log_path)
        # The engine is built on the loop that serves it. Async database
        # clients (pymongo, psycopg's pool) bind to the loop they were
        # opened on; building on a throwaway loop and serving on uvicorn's
        # made every Mongo request fail with "Cannot use AsyncMongoClient
        # in different event loop" (seen in the compose smoke test).
        engine = await build_engine(settings)
        app = None
        try:
            try:
                app = build_app(settings, engine)
            except (ValueError, OSError, ImportError, AttributeError, TypeError, ModelConnectionError, WorkflowError) as error:
                print(f"refusing to serve: {error}", file=sys.stderr)
                sys.exit(2)
            worker = app.state.worker
            consolidation_model = settings.chat_model
            connections = getattr(app.state, 'model_connections', None)
            if connections is not None:
                extraction = connections.get('extraction')
                consolidation_model = extraction.model if extraction is not None else None
            print(
                f"scone-memory on http://{settings.host}:{settings.port} "
                f"documents={engine.documents.name} vectors={engine.vectors.name} embedder={engine.embedder.id} "
                f"spaces={sorted(set(settings.keys.values()))} "
                + (f"consolidation={consolidation_model} every {settings.distill_interval_s:g}s"
                   if worker is not None and getattr(worker, 'distiller', None) is not None else "consolidation=off")
                + (f" conversations={settings.conversations_journal}" if settings.conversations_journal else ""),
                file=sys.stderr,
            )
            await build_server(settings, app).serve()
        finally:
            try:
                agent_runtime = getattr(getattr(app, 'state', None), 'agent_runtime', None)
                if agent_runtime is not None:
                    await agent_runtime.aclose()
            finally:
                try:
                    target = getattr(getattr(app, 'state', None), 'memory_app', app)
                    imports = getattr(getattr(target, 'state', None), 'document_import_service', None)
                    if imports is not None:
                        await imports.aclose()
                finally:
                    try:
                        target = getattr(getattr(app, 'state', None), 'memory_app', app)
                        directory_sync = getattr(getattr(target, 'state', None), 'directory_sync_service', None)
                        if directory_sync is not None:
                            await directory_sync.aclose()
                    finally:
                        await engine.close()

    from ._signals import termination_unwinds
    with termination_unwinds():
        try:
            asyncio.run(run())
        except KeyboardInterrupt:
            # Uvicorn hands the interrupt back once it has stopped gracefully.
            sys.exit(130)


if __name__ == "__main__":
    main()
