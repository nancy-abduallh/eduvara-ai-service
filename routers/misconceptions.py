"""
routers/misconceptions.py
==========================
POST /api/analyze-misconceptions
  Examines a student's wrong quiz answers and returns a list of
  misconception strings describing the gaps in understanding.
"""

import logging
from typing import List, Optional, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from pipeline.model_registry import ModelRegistry

logger = logging.getLogger("edugenie.routers.misconceptions")
router = APIRouter()


class MisconceptionsRequest(BaseModel):
    attempt_id: int
    quiz_id:    int
    answers:    dict   # { question_id (str): chosen_letter (str) }


class MisconceptionsResponse(BaseModel):
    misconceptions: List[str]


@router.post("/analyze-misconceptions", response_model=MisconceptionsResponse)
def analyze_misconceptions(req: MisconceptionsRequest):
    """
    Analyse wrong answers and return a list of topic-level misconceptions.
    Uses Flan-T5-base (tu_model) if available; falls back to a rule-based approach.
    """
    from pipeline.model_registry import ModelRegistry
    import torch

    # We need the quiz answers mapped to question text — the answers dict
    # contains {question_id: chosen_option}.  Without the question DB here
    # we work from keys only (the Laravel caller could pass question texts,
    # but for now we return the question IDs as placeholder topics).
    wrong_ids = list(req.answers.keys())

    if not wrong_ids:
        return {"misconceptions": []}

    tokenizer = ModelRegistry.get("tu_tokenizer")
    model     = ModelRegistry.get("tu_model")
    device    = ModelRegistry.get("tu_device") if "tu_device" in _registry() else ModelRegistry.get("device")

    misconceptions: List[str] = []

    if tokenizer and model:
        # Ask Flan-T5 to summarise misconceptions from question IDs
        # (without question text we produce generic labels)
        prompt = (
            f"A student answered {len(wrong_ids)} questions incorrectly on quiz {req.quiz_id}.\n"
            "List the likely conceptual misconceptions, one per line, max 5:\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=128, num_beams=4, early_stopping=True
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        misconceptions = [
            line.strip().lstrip("•-*0123456789. ")
            for line in response.split("\n")
            if line.strip() and len(line.strip()) > 5
        ][:5]

    if not misconceptions:
        misconceptions = [
            f"Unclear understanding of question area {qid}"
            for qid in wrong_ids[:5]
        ]

    return {"misconceptions": misconceptions}


def _registry():
    from pipeline.model_registry import _models
    return _models
