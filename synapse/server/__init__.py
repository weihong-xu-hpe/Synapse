"""Server package exports for Synapse."""

from synapse.server.mcp import SynapseMCPServer
from synapse.server.service import NodeNotFoundError, SynapseServerService, SynapseServiceError
from synapse.server.streamable import (
    STREAMABLE_ARCHITECTURE_DOC,
    STREAMABLE_RUNTIME_MODE,
    StreamableRuntime,
    create_streamable_app,
    create_streamable_runtime,
    run_streamable_server,
)
from synapse.server.streamable_runtime import (
	StreamableSession,
	StreamableSessionManager,
	StreamableToolOrchestrator,
	StreamableTransportRuntime,
	create_streamable_orchestrator,
	create_streamable_session_manager,
)

create_app = create_streamable_app

__all__ = [
	"NodeNotFoundError",
	"STREAMABLE_ARCHITECTURE_DOC",
	"STREAMABLE_RUNTIME_MODE",
	"SynapseMCPServer",
	"SynapseServerService",
	"SynapseServiceError",
	"StreamableSession",
	"StreamableSessionManager",
	"StreamableToolOrchestrator",
	"StreamableTransportRuntime",
	"StreamableRuntime",
	"create_app",
	"create_streamable_orchestrator",
	"create_streamable_app",
	"create_streamable_session_manager",
	"create_streamable_runtime",
	"run_streamable_server",
]
