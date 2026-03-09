import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
AGENT_DIR = PROJECT_ROOT / "agent"

SANDBOX_BASE_DIR = Path(os.getenv("SANDBOX_BASE_DIR", "./sandbox"))
AGENT_EXECUTOR_MODE = os.getenv("AGENT_EXECUTOR_MODE", "iterative")

CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_RELOAD = os.getenv("API_RELOAD", "true").lower() == "true"
