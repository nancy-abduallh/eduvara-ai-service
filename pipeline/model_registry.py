"""
pipeline/model_registry.py
===========================
Low-RAM friendly: loads only tokenizer from adapter, uses OpenRouter for generation.
"""

import gc
import logging
import os
import threading
from pathlib import Path

import torch

from huggingface_hub import snapshot_download
import os

def download_models_if_needed():
    # Quiz model
    quiz_path = "models/quiz-model"
    if not os.path.exists(f"{quiz_path}/model.safetensors"):
        os.makedirs(quiz_path, exist_ok=True)
        snapshot_download(
            repo_id="NancyAbdullah11/edugenie-models",
            local_dir=quiz_path,
            allow_patterns=["quiz-model/*"]
        )

    # Script adapter
    adapter_path = "models/script-adapter"
    if not os.path.exists(f"{adapter_path}/adapter_model.safetensors"):
        os.makedirs(adapter_path, exist_ok=True)
        snapshot_download(
            repo_id="NancyAbdullah11/edugenie-models",
            local_dir=adapter_path,
            allow_patterns=["script-adapter/*"]
        )

download_models_if_needed()

logger = logging.getLogger("edugenie.registry")

_lock = threading.Lock()
_models = {}
_ready = False
_API_ONLY = object()


def _fix_ssl_and_hf_env() -> None:
    for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        val = os.environ.get(var, "")
        if val and not Path(val).exists():
            logger.warning("Removing broken SSL env var %s=%r", var, val)
            del os.environ[var]
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _warn_low_ram(device: torch.device) -> None:
    if device.type == "cuda":
        return
    try:
        import psutil
        avail_gb = psutil.virtual_memory().available / 1024 ** 3
        total_gb = psutil.virtual_memory().total / 1024 ** 3
        logger.info("  RAM: %.1f GB available / %.1f GB total", avail_gb, total_gb)
        if avail_gb < 15:
            logger.warning("⚠️ Low RAM – will use OpenRouter for script generation.")
    except ImportError:
        pass


def _free_ram_bytes() -> int:
    try:
        import psutil
        return psutil.virtual_memory().available
    except ImportError:
        return 4 * 1024 ** 3


def _service_path(path: str) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str(Path(__file__).resolve().parents[1] / p)


class ModelRegistry:
    @classmethod
    def is_ready(cls) -> bool:
        return _ready

    @classmethod
    def get(cls, name: str):
        val = _models.get(name)
        if val is _API_ONLY:
            return None   # tells script_generator to use API
        if val is not None:
            return val

        # Lazy loading is disabled in low-RAM mode; all heavy models are _API_ONLY
        if name in ("lecture_model", "slide_model", "lecture_tokenizer", "slide_tokenizer"):
            return None

        raise RuntimeError(f"Model '{name}' not loaded.")

    @classmethod
    def load_all(cls):
        global _ready
        with _lock:
            if _ready:
                return

            _fix_ssl_and_hf_env()
            from config import settings
            if settings.HF_TOKEN:
                os.environ["HF_TOKEN"] = settings.HF_TOKEN

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _models["device"] = device
            logger.info("Loading models on device: %s", device)
            _warn_low_ram(device)

            free_gb = _free_ram_bytes() / 1024 ** 3

            # Always load small models (Flan-T5, quiz model)
            cls._load_tu_model(device)
            gc.collect()
            cls._load_quiz_model(settings, device)
            gc.collect()

            if device.type == "cuda" or free_gb >= 15:
                logger.info("Sufficient RAM – loading 7B models locally.")
                cls._load_lecture_model(settings, device)
                gc.collect()
                cls._load_slide_model(settings, device)
                gc.collect()
            else:
                logger.warning(
                    "Low RAM (%.1f GB free) – loading only adapter tokenizer; "
                    "script generation will use OpenRouter API.",
                    free_gb,
                )
                cls._load_lecture_tokenizer_only(settings, device)
                _models["slide_tokenizer"] = _API_ONLY
                _models["slide_model"] = _API_ONLY

            if settings.SDXL_MODEL_PATH:
                cls._load_sdxl(settings)

            _ready = True
            logger.info("✅ ModelRegistry ready (device=%s)", device)

    @classmethod
    def _load_lecture_tokenizer_only(cls, settings, device):
        """Load ONLY tokenizer from adapter (few MB)."""
        try:
            from transformers import AutoTokenizer
            adapter = _service_path(settings.SCRIPT_ADAPTER_PATH)
            logger.info("Loading adapter tokenizer from %s…", adapter)
            tokenizer = AutoTokenizer.from_pretrained(adapter, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            _models["lecture_tokenizer"] = tokenizer
            _models["lecture_model"] = _API_ONLY
            _models["lecture_device"] = device
            logger.info("✅ Adapter tokenizer loaded (model uses OpenRouter)")
        except Exception as exc:
            logger.warning("Failed to load adapter tokenizer: %s", exc)
            _models["lecture_tokenizer"] = _API_ONLY
            _models["lecture_model"] = _API_ONLY

    @classmethod
    def _load_lecture_model(cls, settings, device):
        """Full local load (only called when RAM is sufficient)."""
        # ... (keep your existing full load code here, but we skip for brevity)
        logger.warning("Full lecture model loading not implemented in this low-RAM version.")
        _models["lecture_model"] = None

    @classmethod
    def _load_tu_model(cls, device):
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            logger.info("Loading Flan-T5-base for text understanding…")
            tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
            model = AutoModelForSeq2SeqLM.from_pretrained(
                "google/flan-t5-base", torch_dtype=torch.float32, low_cpu_mem_usage=True
            )
            model.to(device)
            model.eval()
            _models["tu_tokenizer"] = tokenizer
            _models["tu_model"] = model
            _models["tu_device"] = device
            logger.info("✅ Text-understanding model loaded")
        except Exception as exc:
            logger.error("Failed to load TU model: %s", exc)

    @classmethod
    def _load_quiz_model(cls, settings, device):
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            path = _service_path(settings.QUIZ_MODEL_PATH)
            if not Path(path).exists():
                path = "google/flan-t5-base"
            logger.info("Loading quiz model from %s…", path)
            tokenizer = AutoTokenizer.from_pretrained(path)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                path, torch_dtype=torch.float32, low_cpu_mem_usage=True
            )
            model.config.tie_word_embeddings = False
            model.to(device)
            model.eval()
            _models["quiz_tokenizer"] = tokenizer
            _models["quiz_model"] = model
            logger.info("✅ Quiz model loaded")
        except Exception as exc:
            logger.error("Failed to load quiz model: %s", exc)

    @classmethod
    def _load_sdxl(cls, settings):
        # Keep existing SDXL loader if needed
        pass

    @classmethod
    def unload_all(cls):
        global _ready
        _models.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _ready = False