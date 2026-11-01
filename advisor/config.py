"""Paths and API settings. Everything configurable lives in .env (see .env.example)."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

KB = ROOT / "knowledge_base"
CURATED = KB / "curated"
DB_PATH = KB / "advisor.sqlite"
CHUNKS_PATH = KB / "chunks.jsonl"
EMB_PATH = KB / "embeddings.npy"
EMB_CACHE = KB / "embedding_cache.json"

SOURCES = {
    "semester_spread": ROOT / "Semester_Spread_Structures_Sept_2026.xlsx",
    "minors": ROOT / "MinorCoursesforBTech_Students.xlsx",
    "handbook": ROOT / "4. Student Handbook Aug 2026.pdf",
    "sop": ROOT / "SOP STUDENT 17082026 - Final.pdf",
}

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
CHAT_DEPLOYMENT = (os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT") or
                   os.getenv("AZURE_OPENAI_DEPLOYMENT", ""))
EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "")
TRANSCRIBE_DEPLOYMENT = os.getenv("AZURE_OPENAI_TRANSCRIBE_DEPLOYMENT", "")
# Some Azure resources expose transcription on a separate api-version.
TRANSCRIBE_API_VERSION = os.getenv("AZURE_OPENAI_TRANSCRIBE_API_VERSION", AZURE_API_VERSION)

# OpenAI-compatible chat providers. Azure remains the automatic first choice when
# its endpoint, key and deployment are all configured; set LLM_PROVIDER to pin one.
LLM_PROVIDER = (os.getenv("LLM_PROVIDER") or "auto").strip().lower() or "auto"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "").strip() or "gpt-4o-mini"
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "").strip() or "text-embedding-3-small"
OPENAI_TRANSCRIBE_MODEL = os.getenv("OPENAI_TRANSCRIBE_MODEL", "").strip() or "gpt-transcribe"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "").strip() or "llama-3.3-70b-versatile"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "").strip() or "gemini-3.8-flash"


EMB_MODEL_PATH = KB / "embeddings_model.txt"

# AWS Bedrock (embeddings only). boto3 reads AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY from the environment.
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
BEDROCK_EMBED_MODEL = os.getenv("BEDROCK_EMBED_MODEL", "")


def azure_ready(deployment: str) -> bool:
    return bool(AZURE_ENDPOINT and AZURE_API_KEY and deployment)


def chat_provider() -> str | None:
    """Return the configured chat provider, or None when no usable credentials exist."""
    ready = {
        "azure": azure_ready(CHAT_DEPLOYMENT),
        "openai": bool(OPENAI_API_KEY),
        "groq": bool(GROQ_API_KEY),
        "gemini": bool(GEMINI_API_KEY),
    }
    if LLM_PROVIDER != "auto":
        return LLM_PROVIDER if ready.get(LLM_PROVIDER, False) else None
    return next((name for name in ("azure", "openai", "groq", "gemini") if ready[name]), None)


def chat_ready() -> bool:
    return chat_provider() is not None


def transcription_provider() -> str | None:
    if azure_ready(TRANSCRIBE_DEPLOYMENT):
        return "azure"
    if OPENAI_API_KEY:
        return "openai"
    return None


def transcription_ready() -> bool:
    return transcription_provider() is not None


def bedrock_embed_ready() -> bool:
    return bool(BEDROCK_EMBED_MODEL and os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"))
