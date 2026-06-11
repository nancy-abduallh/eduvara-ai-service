"""
config.py
=========
Centralised settings for the EduGenie AI service.
"""

import logging as _logging
from pathlib import Path

from pydantic_settings import BaseSettings

_HERE = Path(__file__).resolve().parent
_ENV_FILE = str(_HERE / ".env")

_pre_log = _logging.getLogger("edugenie.config")
if not (_HERE / ".env").exists():
    _pre_log.error("❌ .env file NOT FOUND at %s", _ENV_FILE)
    _example = _HERE / ".env.example"
    if not _example.exists():
        _example.write_text(
            "# Copy this file to .env and fill in your values\n"
            "AI_API_KEY=\nWEBHOOK_SECRET=\nLARAVEL_URL=http://localhost:8000\n"
            "HF_TOKEN=\nOPENROUTER_API_KEY=\n"
            "OPENROUTER_SCRIPT_MODEL=qwen/qwen-2-7b-instruct:free\n"
            "OPENROUTER_IMAGE_MODEL=black-forest-labs/flux.2-flex\n"
            "SCRIPT_ADAPTER_PATH=./models/script-adapter\n"
            "QUIZ_MODEL_PATH=./models/quiz-model\n"
            "SLIDE_MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2\n"
            "SDXL_MODEL_PATH=\nAVATAR_WAVE_PATH=./assets/avatar_wave.png\n"
            "AVATAR_STAND_PATH=./assets/avatar_stand.png\n"
            "WAV2LIP_CHECKPOINT_PATH=./models/wav2lip/wav2lip_gan.pth\n"
            "PPTX_TEMPLATE_PATH=./assets/template.pptx\n"
            "OUTPUT_DIR=./storage/outputs\nMAX_WORKERS=1\n",
            encoding="utf-8",
        )


class Settings(BaseSettings):
    AI_API_KEY: str = ""
    WEBHOOK_SECRET: str = ""
    LARAVEL_URL: str = "http://localhost:8000"
    HF_TOKEN: str = ""

    SCRIPT_ADAPTER_PATH: str = "./models/script-adapter"
    QUIZ_MODEL_PATH: str = "./models/quiz-model"
    SLIDE_MODEL_NAME: str = "./models/Mistral-7B-Instruct"
    SDXL_MODEL_PATH: str = ""

    OPENROUTER_API_KEY: str = ""
    OPENROUTER_IMAGE_MODEL: str = "black-forest-labs/flux.2-flex"
    OPENROUTER_SCRIPT_MODEL: str = "microsoft/phi-3-mini-128k-instruct:free"
    AVATAR_WAVE_PATH: str = "./assets/avatar_wave.png"
    AVATAR_STAND_PATH: str = "./assets/avatar_stand.png"
    WAV2LIP_CHECKPOINT_PATH: str = "./models/wav2lip/wav2lip_gan.pth"
    PPTX_TEMPLATE_PATH: str = "./assets/template.pptx"
    OUTPUT_DIR: str = "./storage/outputs"
    MAX_WORKERS: int = 1

    model_config = {
        "env_file": _ENV_FILE,
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()
Path(settings.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

_cfg_log = _logging.getLogger("edugenie.config")
_cfg_log.info("Config loaded from: %s", _ENV_FILE)
if not settings.OPENROUTER_API_KEY:
    _cfg_log.warning("⚠️ OPENROUTER_API_KEY missing – using template fallback")