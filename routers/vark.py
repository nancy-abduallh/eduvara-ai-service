"""
routers/vark.py
================
POST /api/classify-vark
  Classifies a student's VARK learning style from their questionnaire answers.
  Option letters map directly to dimensions: a=Visual, b=Auditory,
  c=Reading/writing, d=Kinesthetic.
"""

import logging
from typing import Dict

from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger("edugenie.routers.vark")
router = APIRouter()

# Each VARK question uses the same a→V, b→A, c→R, d→K option mapping.
_LETTER_TO_DIM: Dict[str, str] = {
    "a": "visual",
    "b": "auditory",
    "c": "reading",
    "d": "kinesthetic",
}


class VarkRequest(BaseModel):
    answers: Dict[str, str]   # { "1": "a", "2": "c", … }


@router.post("/classify-vark")
def classify_vark(req: VarkRequest):
    """
    Score VARK questionnaire answers and return per-dimension counts
    plus the dominant learning-style label.
    """
    scores: Dict[str, int] = {
        "visual": 0,
        "auditory": 0,
        "reading": 0,
        "kinesthetic": 0,
    }

    for answer in req.answers.values():
        dim = _LETTER_TO_DIM.get(str(answer).strip().lower())
        if dim:
            scores[dim] += 1

    dominant = max(scores, key=lambda k: scores[k])
    logger.info(f"VARK result={dominant}  scores={scores}")

    return {**scores, "result": dominant}