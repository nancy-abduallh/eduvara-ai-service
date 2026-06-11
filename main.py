"""
EduGenie AI Service
===================
FastAPI server that exposes the video-generation pipeline (edu-genie notebook)
and the MCQ-quiz pipeline (video-script-mcq-quiz-generator notebook) as HTTP
endpoints consumed by the Laravel application.

Run:
    uvicorn main:app --host 0.0.0.0 --port 8001 --workers 1
"""

import os
import sys
import logging
import asyncio
import threading
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.responses import JSONResponse
import httpx

from config import settings
from routers import video, quiz, misconceptions, adaptive, vark   # ← vark added
from pipeline.model_registry import ModelRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("edugenie")


# ── Startup / shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy ML models once at startup; release on shutdown."""
    logger.info("🚀 EduGenie AI Service starting…")
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, ModelRegistry.load_all)
    logger.info("✅ All models loaded — service ready")
    yield
    logger.info("⏹️  Shutting down EduGenie AI Service")
    ModelRegistry.unload_all()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="EduGenie AI Service",
    version="1.0.0",
    description="Video generation + MCQ quiz pipeline API for Eduvara",
    lifespan=lifespan,
)


# ── API-key auth (same key the Laravel AiApiService sends) ────────────────────

def verify_api_key(x_ai_key: str = Header(None), api_key: str = None):
    key = x_ai_key or api_key
    if key != settings.AI_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(
    video.router,
    prefix="/api",
    dependencies=[Depends(verify_api_key)],
)
app.include_router(
    quiz.router,
    prefix="/api",
    dependencies=[Depends(verify_api_key)],
)
app.include_router(
    misconceptions.router,
    prefix="/api",
    dependencies=[Depends(verify_api_key)],
)
app.include_router(
    adaptive.router,
    prefix="/api",
    dependencies=[Depends(verify_api_key)],
)
app.include_router(
    vark.router,
    prefix="/api",
    dependencies=[Depends(verify_api_key)],
)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": ModelRegistry.is_ready()}


# ── Global error handler ──────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url}: {exc}\n{traceback.format_exc()}")
    return JSONResponse(status_code=500, content={"error": str(exc)})