"""
FastAPI Application for Omni-Memory.

Provides REST API endpoints for the memory system.
"""

import os
import logging
from typing import Optional, List
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from simplemem.multimodal.orchestrator import OmniMemoryOrchestrator
from simplemem.multimodal.core.config import OmniMemoryConfig
from simplemem.multimodal.retrieval.pyramid_retriever import RetrievalLevel
from simplemem.multimodal.core.event import EventLevel
from simplemem.multimodal.utils.logging_config import setup_logging

# Setup logging
logger = setup_logging(log_level=os.getenv("LOG_LEVEL", "INFO"))

# Initialize app
app = FastAPI(
    title="Omni-Memory API",
    description="High-Efficiency Unified Multimodal Memory for Agents",
    version="0.1.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global orchestrator instance
orchestrator: Optional[OmniMemoryOrchestrator] = None


# ==================== Pydantic Models ====================

class TextInput(BaseModel):
    """Text input for memory addition."""
    text: str
    session_id: Optional[str] = None
    tags: Optional[List[str]] = None
    force: bool = False


class QueryInput(BaseModel):
    """Query input for retrieval."""
    query: str
    top_k: int = 10
    auto_expand: bool = False
    token_budget: Optional[int] = None


class AnswerInput(BaseModel):
    """Input for answer generation."""
    question: str
    top_k: int = 10
    include_sources: bool = True


class ExpansionInput(BaseModel):
    """Input for expanding MAUs."""
    mau_ids: List[str]
    level: str = "evidence"


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str


class StatsResponse(BaseModel):
    """System statistics response."""
    mau_count: int
    event_count: int
    vector_count: int
    storage_stats: dict


# ==================== Startup/Shutdown ====================

@app.on_event("startup")
async def startup_event():
    """Initialize orchestrator on startup."""
    global orchestrator

    # Load config from environment or defaults
    config = OmniMemoryConfig()
    config.llm.api_key = os.getenv("OPENAI_API_KEY")
    config.llm.api_base_url = os.getenv("OPENAI_API_BASE")

    data_dir = os.getenv("OMNI_MEMORY_DATA_DIR", "./omni_memory_data")

    orchestrator = OmniMemoryOrchestrator(config=config, data_dir=data_dir)
    logger.info("Omni-Memory API started")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown."""
    global orchestrator
    if orchestrator:
        orchestrator.close()
    logger.info("Omni-Memory API shutdown")


# ==================== Health & Stats ====================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", version="0.1.0")


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    stats = orchestrator.get_stats()
    return StatsResponse(
        mau_count=stats["mau_count"],
        event_count=stats["event_count"],
        vector_count=stats["vector_count"],
        storage_stats=stats["storage_stats"],
    )


# ==================== Session Management ====================

@app.post("/session/start")
async def start_session(session_id: Optional[str] = None):
    """Start a new session."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    session = orchestrator.start_session(session_id)
    return {"session_id": session}


@app.post("/session/end")
async def end_session():
    """End current session."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    orchestrator.end_session()
    return {"status": "session ended"}


# ==================== Memory Addition ====================

@app.post("/memory/text")
async def add_text_memory(input: TextInput):
    """Add text to memory."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    result = orchestrator.add_text(
        input.text,
        session_id=input.session_id,
        tags=input.tags,
        force=input.force,
    )

    return {
        "success": result.success,
        "mau_id": result.mau.id if result.mau else None,
        "skipped": result.skipped,
        "trigger_decision": result.trigger_result.decision.value if result.trigger_result else None,
    }


@app.post("/memory/image")
async def add_image_memory(
    image: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    force: bool = Form(False),
):
    """Add image to memory."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    # Save uploaded file temporarily
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(image.filename).suffix) as tmp:
        content = await image.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        tag_list = tags.split(",") if tags else None
        result = orchestrator.add_image(
            tmp_path,
            session_id=session_id,
            tags=tag_list,
            force=force,
        )

        return {
            "success": result.success,
            "mau_id": result.mau.id if result.mau else None,
            "skipped": result.skipped,
            "trigger_decision": result.trigger_result.decision.value if result.trigger_result else None,
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/memory/audio")
async def add_audio_memory(
    audio: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    force: bool = Form(False),
):
    """Add audio to memory."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio.filename).suffix) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        tag_list = tags.split(",") if tags else None
        result = orchestrator.add_audio(
            tmp_path,
            session_id=session_id,
            tags=tag_list,
            force=force,
        )

        return {
            "success": result.success,
            "mau_id": result.mau.id if result.mau else None,
            "skipped": result.skipped,
            "has_speech": result.metadata.get("has_speech", False) if result.metadata else False,
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/memory/video")
async def add_video_memory(
    video: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    max_frames: int = Form(100),
):
    """Add video to memory."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(video.filename).suffix) as tmp:
        content = await video.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        tag_list = tags.split(",") if tags else None
        result = orchestrator.add_video(
            tmp_path,
            session_id=session_id,
            tags=tag_list,
            max_frames=max_frames,
        )

        return {
            "success": result.success,
            "mau_id": result.mau.id if result.mau else None,
            "frames_processed": result.metadata.get("frames_processed", 0) if result.metadata else 0,
            "frames_skipped": result.metadata.get("frames_skipped", 0) if result.metadata else 0,
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ==================== Retrieval ====================

@app.post("/query")
async def query_memories(input: QueryInput):
    """Query memories with pyramid retrieval."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    result = orchestrator.query(
        input.query,
        top_k=input.top_k,
        auto_expand=input.auto_expand,
        token_budget=input.token_budget,
    )

    return result.to_dict()


@app.post("/expand")
async def expand_memories(input: ExpansionInput):
    """Expand specific MAUs to get full details."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    level_map = {
        "summary": RetrievalLevel.SUMMARY,
        "metadata": RetrievalLevel.METADATA,
        "details": RetrievalLevel.DETAILS,
        "evidence": RetrievalLevel.EVIDENCE,
    }
    level = level_map.get(input.level, RetrievalLevel.EVIDENCE)

    result = orchestrator.expand(input.mau_ids, level=level)
    return result.to_dict()


@app.post("/answer")
async def answer_question(input: AnswerInput):
    """Answer a question using memory retrieval."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    result = orchestrator.answer(
        input.question,
        top_k=input.top_k,
        include_sources=input.include_sources,
    )

    return result


# ==================== Events ====================

@app.get("/events")
async def get_events(
    session_id: Optional[str] = Query(None),
    limit: int = Query(10),
):
    """Get recent events."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    events = orchestrator.get_events(session_id=session_id, limit=limit)
    return {"events": events}


@app.get("/events/{event_id}")
async def get_event_details(
    event_id: str,
    level: str = Query("details"),
):
    """Get event details with MAUs."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    level_map = {
        "summary": EventLevel.SUMMARY,
        "details": EventLevel.DETAILS,
        "evidence": EventLevel.EVIDENCE,
    }
    event_level = level_map.get(level, EventLevel.DETAILS)

    result = orchestrator.get_event_details(event_id, level=event_level)
    if not result:
        raise HTTPException(status_code=404, detail="Event not found")

    return result


# ==================== MAU Operations ====================

@app.get("/mau/{mau_id}")
async def get_mau(mau_id: str):
    """Get a specific MAU."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    mau = orchestrator.mau_store.get(mau_id)
    if not mau:
        raise HTTPException(status_code=404, detail="MAU not found")

    return mau.to_dict()


# Run with: uvicorn omni_memory.app:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
