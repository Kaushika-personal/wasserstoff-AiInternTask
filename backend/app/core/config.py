import os
from pydantic_settings import BaseSettings # For Pydantic V2
from dotenv import load_dotenv
from typing import List

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')) # Ensure .env is loaded from backend/

class Settings(BaseSettings):
    PROJECT_NAME: str = "Document Research & Theme Identification Chatbot"
    API_V1_STR: str = "/api/v1"
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000", "http://127.0.0.1:8000"]

    GEMINI_API_KEY: str
    
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./data/app.db")
    VECTOR_DB_PATH: str = os.getenv("VECTOR_DB_PATH", "./data/vector_db")
    
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 200))
    
    GEMINI_EMBEDDING_MODEL_NAME: str = os.getenv("GEMINI_EMBEDDING_MODEL_NAME", "models/embedding-001")
    GEMINI_GENERATIVE_MODEL_NAME: str = os.getenv("GEMINI_GENERATIVE_MODEL_NAME", "models/gemini-1.5-flash-latest") # or "gemini-pro"


    OCR_LANG: str = os.getenv("OCR_LANG", "en")
    MAX_THEMES_TO_IDENTIFY: int = int(os.getenv("MAX_THEMES_TO_IDENTIFY", 5))


    class Config:
        case_sensitive = True
        # env_file = ".env" # pydantic-settings loads .env by default if python-dotenv is installed
        # env_file_encoding = 'utf-8'


settings = Settings()

# Create data directory if it doesn't exist for SQLite
if "sqlite:///" in settings.DATABASE_URL:
    db_path = settings.DATABASE_URL.split("sqlite:///")[1]
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)