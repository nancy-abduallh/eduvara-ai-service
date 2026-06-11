"""
routers/quiz.py
================
POST /api/generate-quiz
  Called by Laravel's GenerateQuizJob after a video is ready.
  Receives the video script and returns MCQ questions synchronously.

  The quiz pipeline automatically excludes any questions that were already
  embedded as K-type interactive slides during video generation
  (those questions are passed in `k_slide_question_texts`).
"""

import logging
from typing import Optional, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from pipeline.model_registry import ModelRegistry
from pipeline.quiz_generator  import generate_mcqs_from_script

logger = logging.getLogger("edugenie.routers.quiz")
router = APIRouter()


class QuizRequest(BaseModel):
    video_id:   int
    topic:      str
    script:     str
    language:   Optional[str] = "en"
    # Questions already used in K-type slides — will be excluded from output
    k_slide_question_texts: Optional[List[str]] = []


@router.post("/generate-quiz")
def generate_quiz(req: QuizRequest):
    """
    Generate MCQ questions from the video script.
    Returns a list of question objects ready to be stored by GenerateQuizJob.
    """
    if not req.script or len(req.script.strip()) < 100:
        raise HTTPException(status_code=422, detail="Script too short to generate quiz")

    tokenizer = ModelRegistry.get("quiz_tokenizer")
    model     = ModelRegistry.get("quiz_model")
    device    = ModelRegistry.get("device")

    if model is None:
        logger.error("Quiz model not loaded")
        raise HTTPException(status_code=503, detail="Quiz model unavailable")

    # Build exclude list from K-slide question texts
    exclude = [{"question": q} for q in (req.k_slide_question_texts or [])]

    questions = generate_mcqs_from_script(
        script_text=req.script,
        tokenizer=tokenizer,
        model=model,
        device=device,
        num_questions=None,   # None = max coverage from all script chunks
        exclude_questions=exclude,
    )

    if not questions:
        logger.warning(f"No quiz questions generated for video_id={req.video_id}")
        return {"questions": []}

    # Normalise to the format expected by GenerateQuizJob
    output = []
    for q in questions:
        output.append({
            "question":      q["question"],
            "options":       [
                q["options"].get("A", ""),
                q["options"].get("B", ""),
                q["options"].get("C", ""),
                q["options"].get("D", ""),
            ],
            "correct_answer": q["correct_answer"],
            "explanation":    q.get("difficulty", "") + " — " + q.get("qtype", ""),
        })

    logger.info(f"Generated {len(output)} questions for video_id={req.video_id}")
    return {"questions": output}
