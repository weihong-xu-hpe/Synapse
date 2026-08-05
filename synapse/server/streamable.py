"""Forward-compatible Streamable MCP runtime entrypoints for Synapse.

This module now targets the native Streamable HTTP runtime and no longer wraps
the removed REST/SSE/stdio compatibility paths.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Callable

from fastapi import FastAPI

from synapse.lifecycle.scheduler import DreamerScheduler
from synapse.server.decider import LocalLLMDecider
from synapse.server.app import create_app as create_streamable_http_app
from synapse.server.streamable_runtime import (
    StreamableSessionManager,
    StreamableToolOrchestrator,
    create_streamable_orchestrator,
    create_streamable_session_manager,
)


STREAMABLE_ARCHITECTURE_DOC = "docs/design/streamable-mcp-single-path-architecture.md"
STREAMABLE_RUNTIME_MODE = "native-streamable-http"


@dataclass(slots=True)
class StreamableRuntime:
    """Minimal Streamable MCP runtime skeleton backed by the current FastAPI internals."""

    config: Any
    runtime_paths: Any = None
    logger: Any = None
    sampling_client: Any = None
    _session_manager: StreamableSessionManager | None = None
    _orchestrator: StreamableToolOrchestrator | None = None
    _dreamer_scheduler: DreamerScheduler | None = None

    @property
    def execution_layer(self) -> Any:
        """Expose the canonical service layer used behind the Streamable runtime."""

        return self.create_orchestrator().service

    def create_session_manager(self) -> StreamableSessionManager:
        """Create or reuse the native Streamable session manager skeleton."""

        if self._session_manager is None:
            self._session_manager = create_streamable_session_manager()
        return self._session_manager

    def create_orchestrator(self) -> StreamableToolOrchestrator:
        """Create or reuse the native Streamable tool orchestrator skeleton."""

        if self._orchestrator is None:
            self._orchestrator = create_streamable_orchestrator(
                self.config,
                runtime_paths=self.runtime_paths,
                logger=self.logger,
                sampling_client=self.sampling_client,
            )
        return self._orchestrator

    def create_app(self) -> FastAPI:
        """Build the current native Streamable HTTP server app."""

        lifespan_context = None
        app_sampling_client = self.sampling_client
        if self.config.dreamer.enabled:
            scheduler_sampling_client = self.sampling_client or LocalLLMDecider(self.config.decider)
            scheduler = DreamerScheduler(
                self.config,
                runtime_paths=self.runtime_paths,
                logger=self.logger,
                sampling_client=scheduler_sampling_client,
            )
            self._dreamer_scheduler = scheduler
            if self.config.decider.provider == "local_llm":
                app_sampling_client = scheduler_sampling_client

            @asynccontextmanager
            async def dreamer_lifespan(_app: FastAPI):
                scheduler.start()
                try:
                    yield
                finally:
                    scheduler.stop()

            lifespan_context = dreamer_lifespan

        app = create_streamable_http_app(
            self.config,
            runtime_paths=self.runtime_paths,
            logger=self.logger,
            sampling_client=app_sampling_client,
            lifespan=lifespan_context,
        )
        app.state.streamable_runtime = self
        app.state.streamable_runtime_mode = STREAMABLE_RUNTIME_MODE
        app.state.streamable_architecture_doc = STREAMABLE_ARCHITECTURE_DOC
        app.state.dreamer_scheduler = self._dreamer_scheduler
        if getattr(app.state, "streamable_session_manager", None) is None:
            app.state.streamable_session_manager = self.create_session_manager()
        else:
            self._session_manager = app.state.streamable_session_manager
        if getattr(app.state, "streamable_orchestrator", None) is None:
            app.state.streamable_orchestrator = self.create_orchestrator()
        else:
            self._orchestrator = app.state.streamable_orchestrator
        return app

    def run(
        self,
        *,
        host: str | None = None,
        port: int | None = None,
        log_level: str = "info",
        uvicorn_runner: Callable[..., object] | None = None,
    ) -> None:
        """Launch the current Streamable-oriented server runtime."""

        runner = uvicorn_runner
        if runner is None:
            import uvicorn

            runner = uvicorn.run

        runner(
            self.create_app(),
            host=host or self.config.server.host,
            port=port or self.config.server.port,
            log_level=log_level,
        )


def create_streamable_runtime(
    config,
    *,
    runtime_paths=None,
    logger=None,
    sampling_client=None,
) -> StreamableRuntime:
    """Create the Streamable MCP runtime skeleton."""

    return StreamableRuntime(
        config=config,
        runtime_paths=runtime_paths,
        logger=logger,
        sampling_client=sampling_client,
    )


def create_streamable_app(
    config,
    *,
    runtime_paths=None,
    logger=None,
    sampling_client=None,
) -> FastAPI:
    """Create the current Streamable-oriented FastAPI app."""

    return create_streamable_runtime(
        config,
        runtime_paths=runtime_paths,
        logger=logger,
        sampling_client=sampling_client,
    ).create_app()


def run_streamable_server(
    config,
    *,
    runtime_paths=None,
    logger=None,
    sampling_client=None,
    host: str | None = None,
    port: int | None = None,
    log_level: str = "info",
    uvicorn_runner: Callable[..., object] | None = None,
) -> None:
    """Launch Synapse through the Streamable-oriented runtime entrypoint."""

    create_streamable_runtime(
        config,
        runtime_paths=runtime_paths,
        logger=logger,
        sampling_client=sampling_client,
    ).run(
        host=host,
        port=port,
        log_level=log_level,
        uvicorn_runner=uvicorn_runner,
    )