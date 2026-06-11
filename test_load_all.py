import logging
import sys

# Reconfigure stdout to use utf-8 to avoid encoding crashes on Windows log redirection
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(name)s — %(message)s"
)
from pipeline.model_registry import ModelRegistry

print("Starting ModelRegistry.load_all()...")
try:
    ModelRegistry.load_all()
    print("SUCCESS: ModelRegistry.load_all() finished successfully!")
except Exception as e:
    print("FAILED:", e)
    import traceback
    traceback.print_exc()
