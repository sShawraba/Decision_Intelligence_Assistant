"""Configuration management"""
import os
from pathlib import Path

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CHROMA_PATH = os.getenv("CHROMA_PATH", str(BASE_DIR / "chroma_data"))
MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "model.pkl"))
LOGS_PATH = os.getenv("LOGS_PATH", str(BASE_DIR / "logs"))

# Create logs directory if it doesn't exist
Path(LOGS_PATH).mkdir(exist_ok=True)

# RAG settings
TOP_K = 3  # Number of documents to retrieve

# Verify required configuration
if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY not set in environment variables")
