"""
HTTP Server for SimpleMem MCP

Provides:
- User registration and authentication (/api/auth/*)
- MCP over Streamable HTTP (2025-03-26 spec) (/mcp)
- Legacy MCP over SSE for backwards compatibility (/mcp/sse)
- Static frontend for configuration (/)

Streamable HTTP Transport:
- Single endpoint /mcp supporting POST, GET, DELETE
- Authentication via Authorization: Bearer <token> header
- Session management via Mcp-Session-Id header
- Supports both JSON and SSE response formats
"""

import asyncio
import json
import os
import uuid
import secrets
from typing import Optional
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import FastAPI, HTTPException, Depends, Query, Request, Header, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel

from .auth.token_manager import TokenManager
from .auth.models import User
from .database.user_store import UserStore
from .database.vector_store import MultiTenantVectorStore
from .integrations.openrouter import OpenRouterClient, OpenRouterClientManager
from .integrations.ollama import OllamaClient, OllamaClientManager
from .mcp_handler import MCPHandler

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import get_settings


# === Pydantic Models ===

class RegisterRequest(BaseModel):
    openrouter_api_key: str


class RegisterResponse(BaseModel):
    success: bool
    token: Optional[str] = None
    user_id: Optional[str] = None
    mcp_endpoint: Optional[str] = None
    error: Optional[str] = None


# === Session Management ===

@dataclass
class MCPSession:
    """Represents an active MCP session"""
    session_id: str
    user_id: str
    user: User
    api_key: str
    handler: MCPHandler
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_active: datetime = field(default_factory=datetime.utcnow)
    initialized: bool = False
    # Server-to-client message queue for GET requests
    message_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    # Track active SSE streams
    active_streams: set = field(default_factory=set)
    # Event ID counter for resumability
    event_counter: int = 0

    def next_event_id(self) -> str:
        """Generate next SSE event ID"""
        self.event_counter += 1
        return f"{self.session_id}-{self.event_counter}"

    def touch(self):
        """Update last active timestamp"""
        self.last_active = datetime.utcnow()


# Session expiry time (30 minutes of inactivity)
SESSION_EXPIRY_MINUTES = 30


# === Global Instances ===

settings = get_settings()
user_store = UserStore(settings.user_db_path)
vector_store = MultiTenantVectorStore(settings.lancedb_path, settings.embedding_dimension)
token_manager = TokenManager(
    secret_key=settings.jwt_secret_key,
    encryption_key=settings.encryption_key,
    expiration_days=settings.jwt_expiration_days,
)

# Initialize client manager based on provider
if settings.llm_provider == "ollama":
    client_manager = OllamaClientManager(
        base_url=settings.ollama_base_url,
        llm_model=settings.llm_model,
        embedding_model=settings.embedding_model,
    )
else:  # Default to OpenRouter
    client_manager = OpenRouterClientManager(
        base_url=settings.openrouter_base_url,
        llm_model=settings.llm_model,
        embedding_model=settings.embedding_model,
    )

# Store active MCP handlers per user (legacy)
_mcp_handlers: dict[str, MCPHandler] = {}

# Store active sessions by session_id
_sessions: dict[str, MCPSession] = {}

# Lock for session operations
_session_lock = asyncio.Lock()


# === Session Helper Functions ===

async def cleanup_expired_sessions():
    """Remove expired sessions"""
    async with _session_lock:
        now = datetime.utcnow()
        expired = [
            sid for sid, session in _sessions.items()
            if (now - session.last_active).total_seconds() > SESSION_EXPIRY_MINUTES * 60
        ]
        for sid in expired:
            del _sessions[sid]
        if expired:
            print(f"Cleaned up {len(expired)} expired sessions")


async def session_cleanup_task():
    """Background task to clean up expired sessions"""
    while True:
        await asyncio.sleep(60)  # Check every minute
        await cleanup_expired_sessions()


def generate_session_id() -> str:
    """Generate a cryptographically secure session ID"""
    return secrets.token_urlsafe(32)


async def get_or_create_session(user: User, api_key: str, session_id: Optional[str] = None) -> MCPSession:
    """Get existing session or create new one"""
    async with _session_lock:
        if session_id and session_id in _sessions:
            session = _sessions[session_id]
            session.touch()
            return session

        # Create new session
        new_session_id = generate_session_id()
        handler = MCPHandler(
            user=user,
            api_key=api_key,
            vector_store=vector_store,
            client_manager=client_manager,
            settings=settings,
        )
        session = MCPSession(
            session_id=new_session_id,
            user_id=user.user_id,
            user=user,
            api_key=api_key,
            handler=handler,
        )
        _sessions[new_session_id] = session
        return session


async def get_session(session_id: str) -> Optional[MCPSession]:
    """Get session by ID"""
    async with _session_lock:
        session = _sessions.get(session_id)
        if session:
            session.touch()
        return session


async def delete_session(session_id: str) -> bool:
    """Delete a session"""
    async with _session_lock:
        if session_id in _sessions:
            del _sessions[session_id]
            return True
        return False


# === Authentication Helper ===

async def verify_bearer_token(authorization: Optional[str]) -> tuple[User, str]:
    """
    Verify Bearer token from Authorization header.
    Returns (user, api_key) tuple.
    Raises HTTPException on failure.
    """
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Invalid Authorization header format. Use: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = parts[1]
    is_valid, payload, error = token_manager.verify_token(token)
    if not is_valid:
        raise HTTPException(
            status_code=401,
            detail=f"Invalid token: {error}",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = user_store.get_user(payload.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    api_key = token_manager.decrypt_api_key(user.openrouter_api_key_encrypted)
    return user, api_key


# === Lifecycle ===

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the app"""
    print("SimpleMem MCP Server started")
    print(f"  - LLM Model: {settings.llm_model}")
    print(f"  - Embedding Model: {settings.embedding_model}")
    print(f"  - Window Size: {settings.window_size}")
    print(f"  - Transport: Streamable HTTP (2025-03-26)")

    # Start session cleanup task
    cleanup_task = asyncio.create_task(session_cleanup_task())

    yield

    # Cancel cleanup task
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

    print("SimpleMem MCP Server stopped")


# === FastAPI App ===

app = FastAPI(
    title="SimpleMem MCP Server",
    description="Multi-tenant Memory Service for LLM Agents",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === Authentication Endpoints ===

@app.post("/api/auth/register", response_model=RegisterResponse)
async def register(request: RegisterRequest):
    """Register a new user with API key (or placeholder for Ollama)"""
    try:
        api_key = request.openrouter_api_key

        # For Ollama, we don't need a real API key, use a placeholder
        if settings.llm_provider == "ollama":
            if not api_key or api_key == "":
                # Use a placeholder key for Ollama
                api_key = "ollama-placeholder-key"

            # Verify Ollama is accessible
            client = OllamaClient(base_url=settings.ollama_base_url)
            is_valid, error = await client.verify_api_key()
            await client.close()

            if not is_valid:
                return RegisterResponse(
                    success=False,
                    error=f"Cannot connect to Ollama: {error}",
                )
        else:
            # For OpenRouter, validate the API key
            client = OpenRouterClient(
                api_key=api_key,
                base_url=settings.openrouter_base_url,
            )
            is_valid, error = await client.verify_api_key()
            await client.close()

            if not is_valid:
                return RegisterResponse(
                    success=False,
                    error=f"Invalid OpenRouter API key: {error}",
                )

        # Create user
        user = User()
        user.openrouter_api_key_encrypted = token_manager.encrypt_api_key(api_key)

        # Save user
        user_store.create_user(user)

        # Generate token
        token = token_manager.generate_token(user)

        # Base URL for MCP endpoint (can be overridden via env)
        base_url = os.getenv("MCP_BASE_URL", "")
        mcp_endpoint = f"{base_url}/mcp" if base_url else "/mcp"

        return RegisterResponse(
            success=True,
            token=token,
            user_id=user.user_id,
            mcp_endpoint=mcp_endpoint,  # Streamable HTTP endpoint
        )

    except Exception as e:
        return RegisterResponse(
            success=False,
            error=str(e),
        )


@app.get("/api/auth/verify")
async def verify_token(token: str = Query(..., description="Token to verify")):
    """Verify token is valid"""
    is_valid, payload, error = token_manager.verify_token(token)
    if not is_valid:
        raise HTTPException(status_code=401, detail=error)

    user = user_store.get_user(payload.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return {
        "valid": True,
        "user_id": user.user_id,
    }


@app.post("/api/auth/refresh")
async def refresh_token(token: str = Query(..., description="Token to refresh")):
    """Refresh authentication token"""
    is_valid, payload, error = token_manager.verify_token(token)
    if not is_valid:
        raise HTTPException(status_code=401, detail=error)

    user = user_store.get_user(payload.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    new_token = token_manager.generate_token(user)
    return {
        "success": True,
        "token": new_token,
    }


# === Health & Info ===

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
    }


@app.get("/api/server/info")
async def server_info():
    """Get server information"""
    return {
        "name": "SimpleMem MCP Server",
        "version": "1.0.0",
        "protocol_version": "2025-03-26",
        "transport": "Streamable HTTP",
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
        "window_size": settings.window_size,
        "active_sessions": len(_sessions),
        "total_users": user_store.count_users(),
    }


# === MCP Protocol Endpoints (Streamable HTTP - 2025-03-26 spec) ===

def _get_mcp_handler(user: User, api_key: str) -> MCPHandler:
    """Get or create MCP handler for user (legacy)"""
    if user.user_id not in _mcp_handlers:
        _mcp_handlers[user.user_id] = MCPHandler(
            user=user,
            api_key=api_key,
            vector_store=vector_store,
            client_manager=client_manager,
            settings=settings,
        )
    return _mcp_handlers[user.user_id]


def _is_initialize_request(data: dict | list) -> bool:
    """Check if the message is an initialize request"""
    if isinstance(data, list):
        return any(
            isinstance(item, dict) and item.get("method") == "initialize"
            for item in data
        )
    return isinstance(data, dict) and data.get("method") == "initialize"


def _is_notification_or_response_only(data: dict | list) -> bool:
    """Check if message contains only notifications or responses (no requests)"""
    items = data if isinstance(data, list) else [data]
    for item in items:
        if not isinstance(item, dict):
            continue
        # Has 'method' but no 'id' -> notification
        # Has 'result' or 'error' -> response
        # Has both 'method' and 'id' -> request
        if "method" in item and "id" in item:
            return False  # This is a request
    return True


@app.post("/mcp")
async def mcp_post_endpoint(
    request: Request,
    authorization: Optional[str] = Header(None),
    mcp_session_id: Optional[str] = Header(None, alias="Mcp-Session-Id"),
):
    """
    Streamable HTTP POST endpoint (MCP 2025-03-26 spec).

    Handles JSON-RPC 2.0 messages from clients.

    Headers:
    - Authorization: Bearer <token> (required)
    - Accept: application/json, text/event-stream (required)
    - Mcp-Session-Id: <session-id> (required after initialization)

    Request body: JSON-RPC request, notification, response, or array of them.

    Response:
    - For notifications/responses only: 202 Accepted
    - For requests: JSON response or SSE stream
    """
    # Validate Accept header
    accept = request.headers.get("accept", "")
    if "application/json" not in accept and "text/event-stream" not in accept:
        raise HTTPException(
            status_code=406,
            detail="Accept header must include application/json or text/event-stream",
        )

    # Authenticate
    user, api_key = await verify_bearer_token(authorization)
    user_store.update_last_active(user.user_id)

    # Parse request body
    try:
        body = await request.body()
        data = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as e:
        return JSONResponse(
            status_code=400,
            content={
                "jsonrpc": "2.0",
                "error": {"code": -32700, "message": f"Parse error: {e}"},
            },
        )

    # Handle initialization (creates new session)
    if _is_initialize_request(data):
        session = await get_or_create_session(user, api_key)
        session.initialized = True

        # Process the initialize request
        response_str = await session.handler.handle_message(json.dumps(data))
        response_data = json.loads(response_str)

        # Return with Mcp-Session-Id header
        return JSONResponse(
            content=response_data,
            headers={"Mcp-Session-Id": session.session_id},
        )

    # For non-initialization requests, session ID is required
    if not mcp_session_id:
        raise HTTPException(
            status_code=400,
            detail="Mcp-Session-Id header required for non-initialization requests",
        )

    # Get existing session
    session = await get_session(mcp_session_id)
    if not session:
        # Auto-create new session for expired sessions (MCP client compatibility)
        # This allows MCP clients to seamlessly recover from expired sessions
        session = await get_or_create_session(user, api_key)
        session.initialized = True
        # Optionally log the session recovery
        print(f"Auto-recovered expired session for user {user.user_id}, new session_id: {session.session_id}")

    # Verify session belongs to this user
    if session.user_id != user.user_id:
        raise HTTPException(status_code=403, detail="Session does not belong to this user")

    # If only notifications or responses, return 202 Accepted
    if _is_notification_or_response_only(data):
        # Still process them (e.g., initialized notification)
        await session.handler.handle_message(json.dumps(data))
        return Response(status_code=202)

    # Process request(s) and return response
    # For now, we return JSON. SSE streaming can be added for long-running tools.
    response_str = await session.handler.handle_message(json.dumps(data))
    response_data = json.loads(response_str)

    return JSONResponse(
        content=response_data,
        headers={"Mcp-Session-Id": session.session_id},
    )


@app.get("/mcp")
async def mcp_get_endpoint(
    request: Request,
    authorization: Optional[str] = Header(None),
    mcp_session_id: Optional[str] = Header(None, alias="Mcp-Session-Id"),
    last_event_id: Optional[str] = Header(None, alias="Last-Event-ID"),
):
    """
    Streamable HTTP GET endpoint for server-to-client SSE stream.

    Used for server-initiated messages (notifications, requests to client).

    Headers:
    - Authorization: Bearer <token> (required)
    - Accept: text/event-stream (required)
    - Mcp-Session-Id: <session-id> (required)
    - Last-Event-ID: <event-id> (optional, for resumability)
    """
    # Validate Accept header
    accept = request.headers.get("accept", "")
    if "text/event-stream" not in accept:
        raise HTTPException(
            status_code=406,
            detail="Accept header must include text/event-stream",
        )

    # Authenticate
    user, api_key = await verify_bearer_token(authorization)

    # Session ID required for GET
    if not mcp_session_id:
        raise HTTPException(
            status_code=400,
            detail="Mcp-Session-Id header required",
        )

    # Get session
    session = await get_session(mcp_session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail="Session not found or expired",
        )

    # Verify session belongs to this user
    if session.user_id != user.user_id:
        raise HTTPException(status_code=403, detail="Session does not belong to this user")

    # Generate unique stream ID
    stream_id = secrets.token_urlsafe(16)
    session.active_streams.add(stream_id)

    async def event_generator():
        """Generate SSE events for server-to-client messages"""
        try:
            # Send initial keepalive
            yield ": keepalive\n\n"

            while True:
                try:
                    # Wait for messages with timeout for keepalive
                    message = await asyncio.wait_for(
                        session.message_queue.get(),
                        timeout=15.0,
                    )

                    # Format as SSE event
                    event_id = session.next_event_id()
                    yield f"id: {event_id}\n"
                    yield f"event: message\n"
                    yield f"data: {json.dumps(message)}\n\n"

                except asyncio.TimeoutError:
                    # Send keepalive comment
                    yield ": keepalive\n\n"

        finally:
            # Remove stream from active set
            session.active_streams.discard(stream_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Mcp-Session-Id": session.session_id,
        },
    )


@app.delete("/mcp")
async def mcp_delete_endpoint(
    authorization: Optional[str] = Header(None),
    mcp_session_id: Optional[str] = Header(None, alias="Mcp-Session-Id"),
):
    """
    Terminate an MCP session.

    Headers:
    - Authorization: Bearer <token> (required)
    - Mcp-Session-Id: <session-id> (required)
    """
    # Authenticate
    user, api_key = await verify_bearer_token(authorization)

    if not mcp_session_id:
        raise HTTPException(status_code=400, detail="Mcp-Session-Id header required")

    # Get session to verify ownership
    session = await get_session(mcp_session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.user_id != user.user_id:
        raise HTTPException(status_code=403, detail="Session does not belong to this user")

    # Delete session
    await delete_session(mcp_session_id)
    return Response(status_code=204)


# === Legacy MCP Endpoints (HTTP+SSE - 2024-11-05 spec) ===
# Kept for backwards compatibility with older clients

@app.get("/mcp/sse")
async def mcp_sse_endpoint_legacy(
    request: Request,
    token: Optional[str] = Query(None, description="Authentication token (legacy)"),
    authorization: Optional[str] = Header(None),
):
    """
    Legacy MCP over Server-Sent Events (SSE) endpoint.

    DEPRECATED: Use Streamable HTTP at /mcp instead.

    Supports both:
    - Query param: ?token=<token> (legacy)
    - Header: Authorization: Bearer <token> (preferred)
    """
    # Try to get token from header first, then query param
    if authorization:
        user, api_key = await verify_bearer_token(authorization)
    elif token:
        is_valid, payload, error = token_manager.verify_token(token)
        if not is_valid:
            raise HTTPException(status_code=401, detail=error)
        user = user_store.get_user(payload.user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        api_key = token_manager.decrypt_api_key(user.openrouter_api_key_encrypted)
    else:
        raise HTTPException(status_code=401, detail="Authentication required")

    # Create session for legacy client
    session = await get_or_create_session(user, api_key)

    # Get base URL
    base_url = os.getenv("MCP_BASE_URL", "")
    message_endpoint = f"{base_url}/mcp/message" if base_url else "/mcp/message"

    async def event_generator():
        """Generate SSE events"""
        # Send endpoint info as first event (legacy format)
        endpoint_url = f"{message_endpoint}?session_id={session.session_id}"
        yield f"event: endpoint\ndata: {endpoint_url}\n\n"

        # Send initial keepalive
        yield ": keepalive\n\n"

        # Keep connection alive
        while True:
            await asyncio.sleep(15)
            yield ": keepalive\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.post("/mcp/message")
async def mcp_message_endpoint_legacy(
    request: Request,
    session_id: Optional[str] = Query(None, description="Session ID"),
    token: Optional[str] = Query(None, description="Authentication token (legacy)"),
    authorization: Optional[str] = Header(None),
    mcp_session_id: Optional[str] = Header(None, alias="Mcp-Session-Id"),
):
    """
    Legacy MCP message endpoint.

    DEPRECATED: Use Streamable HTTP POST to /mcp instead.

    Supports both legacy and new authentication methods.
    """
    # Get session ID from header or query param
    sid = mcp_session_id or session_id

    # Try to authenticate and get session
    if authorization:
        user, api_key = await verify_bearer_token(authorization)
        if sid:
            session = await get_session(sid)
            if session and session.user_id == user.user_id:
                handler = session.handler
            else:
                handler = _get_mcp_handler(user, api_key)
        else:
            handler = _get_mcp_handler(user, api_key)
    elif token:
        is_valid, payload, error = token_manager.verify_token(token)
        if not is_valid:
            raise HTTPException(status_code=401, detail=error)
        user = user_store.get_user(payload.user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        api_key = token_manager.decrypt_api_key(user.openrouter_api_key_encrypted)
        handler = _get_mcp_handler(user, api_key)
    elif sid:
        # Try to get session by ID
        session = await get_session(sid)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        handler = session.handler
        user = session.user
    else:
        raise HTTPException(status_code=401, detail="Authentication required")

    user_store.update_last_active(user.user_id)

    # Process message
    body = await request.body()
    response = await handler.handle_message(body.decode("utf-8"))

    return json.loads(response)


# === Static Files (Frontend) ===

frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve frontend HTML"""
    index_path = os.path.join(frontend_path, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>SimpleMem MCP Server</h1><p>Frontend not found.</p>")


# === Entry Point ===

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the HTTP server"""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
