"""
routers/video.py
=================
POST /api/generate-video
  Accepts a video-generation request from Laravel's GenerateVideoJob.
  Returns a job_id immediately; the actual work runs in a background thread
  and posts results back to Laravel via webhook when complete.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from pipeline.job_worker import dispatch_video_job

logger = logging.getLogger("edugenie.routers.video")
router = APIRouter()


class VideoRequest(BaseModel):
    video_id:       int
    topic:          str
    caption:        str
    learning_style: Optional[str] = "R"
    proficiency:    Optional[str] = "beginner"
    language:       Optional[str] = "en"
    script:         Optional[str] = None          # pre-generated script (optional)
    webhook_url:    str


@router.post("/generate-video")
def generate_video(req: VideoRequest):
    """
    Queue a video generation job.
    Returns immediately with a job_id; Laravel polls via GET /videos/{id}/status
    or receives the result through the webhook.
    """
    job_id = dispatch_video_job(req.model_dump())
    logger.info(f"Video job {job_id} queued for video_id={req.video_id}")
    return {"job_id": job_id, "status": "queued"}
