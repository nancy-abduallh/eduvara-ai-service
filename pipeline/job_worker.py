"""
pipeline/job_worker.py
=======================
Runs asynchronous video and adaptive-lesson jobs and posts webhook results back
to Laravel. The video path follows the edu-genie-upgraded notebook:
script -> slides -> style-specific assets/notes -> PPTX -> slide video ->
avatar overlay -> final MP4.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import re
import shutil
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger("edugenie.worker")
_executor = ThreadPoolExecutor(max_workers=1)


def _send_webhook(url: str, payload: dict, secret: str):
    body = json.dumps(payload).encode()
    sig = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    try:
        resp = httpx.post(
            url,
            content=body,
            headers={"Content-Type": "application/json", "X-Webhook-Signature": sig},
            timeout=30,
        )
        logger.info("Webhook -> %s HTTP %s", url, resp.status_code)
    except Exception as exc:
        logger.error("Webhook delivery failed: %s", exc)


def _safe_title(text: str) -> str:
    safe = re.sub(r"[^\w\s-]", "", text or "lecture").strip().replace(" ", "_")
    return re.sub(r"_+", "_", safe)[:60] or "lecture"


def _normalize_learning_style(style: str) -> str:
    """Convert Laravel VARK names into the notebook's R/V/A/K branch codes."""
    value = (style or "reading").strip().lower()
    aliases = {
        "r": "R",
        "reading": "R",
        "read": "R",
        "reading/writing": "R",
        "v": "V",
        "visual": "V",
        "a": "A",
        "auditory": "A",
        "audio": "A",
        "k": "K",
        "kinesthetic": "K",
        "kinaesthetic": "K",
    }
    return aliases.get(value, "R")


def _service_path(path: str) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str(Path(__file__).resolve().parents[1] / p)


def _run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _copy_or_none(src: str, dst: str) -> Optional[str]:
    if src and os.path.exists(src):
        shutil.copy(src, dst)
        return dst
    return None


def run_video_job(payload: dict, job_id: str, secret: str):
    from config import settings
    from pipeline.model_registry import ModelRegistry
    from pipeline.script_generator import generate_script
    from pipeline.slide_generator import (
        build_slides_list,
        create_presentation,
        generate_images_from_concepts,
        generate_speaker_note,
        generate_visual_concepts,
        inject_k_questions,
        merge_visual_speaker_notes,
        summarize_explanations,
        wikipedia_search,
    )
    from pipeline.quiz_generator import generate_mcqs_from_script
    from pipeline.video_assembler import (
        concat_final_video,
        concat_slide_videos,
        generate_avatar_clip,
        generate_slide_audios,
        overlay_avatar_on_slides,
        pptx_to_frames,
        render_slide_video,
        _get_duration,
        _ffmpeg,
    )

    webhook_url = payload.get("webhook_url")
    topic       = payload.get("topic", "")
    caption     = payload.get("caption") or topic
    ls          = _normalize_learning_style(payload.get("learning_style"))
    language    = (payload.get("language") or "en").lower()

    output_root = Path(_service_path(settings.OUTPUT_DIR))
    workdir     = output_root / job_id
    workdir.mkdir(parents=True, exist_ok=True)

    # Resolve asset paths from settings (same config-driven pattern as model paths)
    wav2lip_ckpt = _service_path(settings.WAV2LIP_CHECKPOINT_PATH)
    avatar_stand = _service_path(settings.AVATAR_STAND_PATH)

    try:
        device = ModelRegistry.get("device")

        logger.info("[%s] Generating script for: %s", job_id, topic)
        script = payload.get("script") or generate_script(
            topic,
            tokenizer=ModelRegistry.get("lecture_tokenizer"),
            model=ModelRegistry.get("lecture_model"),
            device=device,
        )

        logger.info("[%s] Building slides JSON", job_id)
        slides_list = build_slides_list(
            script,
            tokenizer=ModelRegistry.get("slide_tokenizer"),
            model=ModelRegistry.get("slide_model"),
            device=device,
        )

        k_slide_questions: list = []
        if ls == "K":
            logger.info("[%s] K-type: generating quiz questions for slide injection", job_id)
            all_mcqs = generate_mcqs_from_script(
                script,
                tokenizer=ModelRegistry.get("quiz_tokenizer"),
                model=ModelRegistry.get("quiz_model"),
                device=device,
                num_questions=None,
            )
            keep_for_slides = max(1, len(all_mcqs) // 2) if all_mcqs else 0
            slides_list, k_slide_questions = inject_k_questions(
                slides_list, all_mcqs[:keep_for_slides]
            )

        if ls != "R":
            slides_list = summarize_explanations(
                slides_list,
                tokenizer=ModelRegistry.get("slide_tokenizer"),
                model=ModelRegistry.get("slide_model"),
            )

        logger.info("[%s] Generating speaker notes", job_id)
        speaker_notes = [
            generate_speaker_note(
                slide,
                tokenizer=ModelRegistry.get("slide_tokenizer"),
                model=ModelRegistry.get("slide_model"),
                device=device,
            )
            for slide in slides_list
        ]

        visuals_list = []
        image_folder = str(workdir / "generated_images")
        if ls == "V" and language.startswith("en"):
            logger.info(
                "[%s] V-type English: generating visual concepts and OpenRouter images", job_id
            )
            visuals_list = generate_visual_concepts(
                script,
                tokenizer=ModelRegistry.get("slide_tokenizer"),
                model=ModelRegistry.get("slide_model"),
                device=device,
            )
            generate_images_from_concepts(
                visuals_list,
                image_folder=image_folder,
                api_key=settings.OPENROUTER_API_KEY,
                model_name=getattr(
                    settings,
                    "OPENROUTER_IMAGE_MODEL",
                    "black-forest-labs/flux.2-flex",   # default image model
                ),
            )
            speaker_notes = merge_visual_speaker_notes(
                slides_list, speaker_notes, visuals_list
            )
        elif ls == "V":
            logger.info(
                "[%s] V-type requested for non-English language; skipping OpenRouter image branch",
                job_id,
            )

        references = wikipedia_search(topic or caption) if ls == "R" else []

        logger.info("[%s] Generating TTS audio (%s notes)", job_id, len(speaker_notes))
        audio_paths = _run_async(generate_slide_audios(speaker_notes, str(workdir)))

        logger.info("[%s] Creating PPTX", job_id)
        pptx_path = str(workdir / "lecture.pptx")
        create_presentation(
            slides_list=slides_list,
            template_path=_service_path(settings.PPTX_TEMPLATE_PATH),
            output_path=pptx_path,
            learning_style=ls,
            visuals_list=visuals_list,
            results=references,
            image_path=image_folder,
            title=caption,
            speaker_notes=speaker_notes,
        )

        logger.info("[%s] Converting PPTX to frames", job_id)
        frame_paths = pptx_to_frames(pptx_path, str(workdir))

        if not frame_paths:
            raise RuntimeError("pptx_to_frames returned no frames — PPTX conversion failed")

        logger.info("[%s] Rendering slide videos (%d frames)", job_id, len(frame_paths))
        # frame_paths[0]   = cover slide  (no speaker note → no audio → silent clip)
        # frame_paths[1..N] = content slides (audio_paths[0..N-1])
        # We render ALL frames so the cover card appears at the start of the video.
        total = len(frame_paths)
        segs, durs = [], []
        for i, frame in enumerate(frame_paths):
            seg_path   = str(workdir / f"seg_{i}.mp4")
            # frame[0] is the cover: no audio (renders as a 5-second silent title card)
            # frame[i≥1]: aligned to audio_paths[i-1]
            audio_file = audio_paths[i - 1] if i >= 1 and (i - 1) < len(audio_paths) else None
            if render_slide_video(frame, audio_file, seg_path, i + 1, total):
                segs.append(seg_path)
                durs.append(_get_duration(seg_path))

        if not segs:
            raise RuntimeError("No slide segments rendered")

        slides_mp4 = str(workdir / "slides.mp4")
        concat_slide_videos(segs, durs, slides_mp4)

        # ── Intro / outro audio ───────────────────────────────────────────────
        intro_text = (
            f"Hello everyone, and welcome to today's lesson on {caption}! "
            "I am excited for our learning journey today. "
            "Get your notebooks ready, and let's dive right in!"
        )
        outro_text = (
            "That's all for today's lesson. "
            "I hope the information was clear and easy to understand. "
            "Don't forget to practice, and I'll see you in the next video. Goodbye!"
        )
        intro_dir = workdir / "intro_audio"
        outro_dir = workdir / "outro_audio"
        intro_dir.mkdir(exist_ok=True)
        outro_dir.mkdir(exist_ok=True)
        intro_audio = _run_async(generate_slide_audios([intro_text], str(intro_dir)))[0]
        outro_audio = _run_async(generate_slide_audios([outro_text], str(outro_dir)))[0]
        intro_mp3 = _copy_or_none(intro_audio, str(workdir / "intro.mp3"))
        outro_mp3 = _copy_or_none(outro_audio, str(workdir / "outro.mp3"))

        # ── Avatar clips — paths come from settings (no hardcoded strings) ───
        avatar_intro = avatar_outro = None
        if os.path.exists(wav2lip_ckpt) and os.path.exists(avatar_stand):
            if intro_mp3:
                avatar_intro = generate_avatar_clip(
                    face_image=avatar_stand,
                    audio_file=intro_mp3,
                    output_file=str(workdir / "avatar_intro.mp4"),
                    wav2lip_checkpoint=wav2lip_ckpt,
                )
            if outro_mp3:
                avatar_outro = generate_avatar_clip(
                    face_image=avatar_stand,
                    audio_file=outro_mp3,
                    output_file=str(workdir / "avatar_outro.mp4"),
                    wav2lip_checkpoint=wav2lip_ckpt,
                )
        else:
            logger.info(
                "[%s] Wav2Lip checkpoint or avatar image not found — skipping avatar clips. "
                "Set WAV2LIP_CHECKPOINT_PATH and AVATAR_STAND_PATH in .env to enable.",
                job_id,
            )

        # ── Body section (slides + optional PiP avatar) ───────────────────────
        body_mp4 = str(workdir / "body.mp4")
        if avatar_intro or avatar_outro:
            try:
                full_audio  = str(workdir / "slides_audio.mp3")
                subprocess.run(
                    [_ffmpeg(), "-y", "-i", slides_mp4, "-vn", "-acodec", "mp3", full_audio],
                    check=True, capture_output=True,
                )
                body_avatar = generate_avatar_clip(
                    face_image=avatar_stand,
                    audio_file=full_audio,
                    output_file=str(workdir / "body_avatar.mp4"),
                    wav2lip_checkpoint=wav2lip_ckpt,
                )
                if body_avatar:
                    overlay_avatar_on_slides(slides_mp4, body_avatar, body_mp4)
                else:
                    shutil.copy(slides_mp4, body_mp4)
            except Exception as exc:
                logger.warning("PiP overlay failed, using plain slides: %s", exc)
                shutil.copy(slides_mp4, body_mp4)
        else:
            shutil.copy(slides_mp4, body_mp4)

        # ── Final assembly ────────────────────────────────────────────────────
        final_mp4 = str(workdir / f"{_safe_title(caption)}.mp4")
        if avatar_intro and avatar_outro:
            concat_final_video(avatar_intro, body_mp4, avatar_outro, final_mp4)
        elif avatar_intro:
            subprocess.run(
                [
                    _ffmpeg(), "-y",
                    "-i", avatar_intro, "-i", body_mp4,
                    "-filter_complex",
                    "[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[vout][aout]",
                    "-map", "[vout]", "-map", "[aout]",
                    "-c:v", "libx264", "-c:a", "aac", final_mp4,
                ],
                check=True, capture_output=True,
            )
        else:
            shutil.copy(body_mp4, final_mp4)

        rel_path   = str(Path(final_mp4).relative_to(output_root))
        thumb_path = _generate_thumbnail(final_mp4, workdir)

        if webhook_url:
            _send_webhook(
                webhook_url,
                {
                    "job_id":    job_id,
                    "status":    "completed",
                    "video_path": rel_path,
                    "thumbnail_path": (
                        str(Path(thumb_path).relative_to(output_root))
                        if thumb_path else None
                    ),
                    "script": script,
                    "k_slide_questions": [
                        {
                            "question":       q.get("question", ""),
                            "options":        q.get("options", {}),
                            "correct":        q.get("correct", ""),
                            "correct_answer": q.get("correct_answer", ""),
                            "explanation":    q.get("explanation", ""),
                        }
                        for q in k_slide_questions
                    ],
                },
                secret,
            )

        logger.info("[%s] Video complete: %s", job_id, final_mp4)

    except Exception as exc:
        import traceback
        logger.error("[%s] Video job failed: %s\n%s", job_id, exc, traceback.format_exc())
        if webhook_url:
            _send_webhook(
                webhook_url,
                {"job_id": job_id, "status": "failed", "error": str(exc)},
                secret,
            )


def run_adaptive_lesson_job(payload: dict, job_id: str, secret: str):
    misconceptions = payload.get("misconceptions", [])
    user_id        = payload.get("user_id")
    webhook_url    = payload.get("webhook_url")
    if not misconceptions:
        if webhook_url:
            _send_webhook(
                webhook_url,
                {"job_id": job_id, "status": "failed", "error": "No misconceptions provided"},
                secret,
            )
        return
    topic = (
        "Targeted remedial lesson addressing these misconceptions: "
        + "; ".join(str(m) for m in misconceptions[:5])
    )
    run_video_job(
        {
            "topic":          topic,
            "caption":        "Adaptive Remedial Lesson",
            "webhook_url":    webhook_url,
            "user_id":        user_id,
            "learning_style": "R",
        },
        job_id,
        secret,
    )


def _generate_thumbnail(video_path: str, workdir: Path) -> Optional[str]:
    thumb = str(workdir / "thumbnail.jpg")
    try:
        from pipeline.video_assembler import _ffmpeg
        subprocess.run(
            [_ffmpeg(), "-y", "-i", video_path,
             "-ss", "00:00:02", "-vframes", "1", "-q:v", "3", thumb],
            capture_output=True, timeout=30,
        )
        return thumb if os.path.exists(thumb) else None
    except Exception:
        return None


def dispatch_video_job(payload: dict) -> str:
    from config import settings
    job_id = str(uuid.uuid4())
    _executor.submit(run_video_job, payload, job_id, settings.WEBHOOK_SECRET)
    return job_id


def dispatch_adaptive_job(payload: dict) -> str:
    from config import settings
    job_id = str(uuid.uuid4())
    _executor.submit(run_adaptive_lesson_job, payload, job_id, settings.WEBHOOK_SECRET)
    return job_id