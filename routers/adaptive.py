"""
routers/adaptive.py
====================
POST /api/generate-adaptive-lesson
  Queues a targeted remedial video for a student who failed a quiz.
  The job re-uses the full video pipeline with a misconception-focused topic.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from pipeline.job_worker import dispatch_adaptive_job

logger = logging.getLogger("edugenie.routers.adaptive")
router = APIRouter()


class AdaptiveLessonRequest(BaseModel):
    lesson_id:      int
    misconceptions: List[str]
    user_id:        int
    webhook_url:    str


@router.post("/generate-adaptive-lesson")
def generate_adaptive_lesson(req: AdaptiveLessonRequest):
    """Queue an adaptive lesson video and return the job_id immediately."""
    job_id = dispatch_adaptive_job(req.model_dump())
    logger.info(f"Adaptive lesson job {job_id} queued for lesson_id={req.lesson_id}")
    return {"job_id": job_id, "status": "queued"}
