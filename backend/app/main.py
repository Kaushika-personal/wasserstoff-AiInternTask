from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.endpoints import documents as doc_endpoints, queries as query_endpoints # Aliased
from app.db.base import init_db
import logging
import sys
import os # For singleton initialization

# --- Logging Configuration ---
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO, # Change to DEBUG for more verbose output during development
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)",
)
logger = logging.getLogger(__name__)
# --- End Logging Configuration ---

# Create an instance of the FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    version="0.1.0" # Example version
)

# --- Event Handlers ---
@app.on_event("startup")
async def startup_event():
    logger.info("Application startup: Initializing database...")
    init_db() # Create database tables if they don't exist
    logger.info("Database initialized.")
    
    # Pre-load singleton services to initialize models/connections early
    logger.info("Pre-loading OCR Service...")
    from app.services.ocr_service import get_ocr_service
    get_ocr_service() # Initializes PaddleOCR model
    logger.info("OCR Service pre-loaded.")

    logger.info("Pre-loading Embedding Service...")
    from app.services.embedding_service import get_embedding_service
    get_embedding_service() # Initializes OpenAI Embeddings and Chroma
    logger.info("Embedding Service pre-loaded.")

    logger.info("Pre-loading Theme Service...")
    from app.services.theme_service import get_theme_service
    get_theme_service() # Initializes LLM for theming
    logger.info("Theme Service pre-loaded.")

    logger.info("Pre-loading Query Service...")
    from app.services.query_service import get_query_service
    get_query_service() # Initializes LLM for querying
    logger.info("Query Service pre-loaded.")
    
    logger.info("Application startup complete.")

# --- CORS Middleware ---
# Configure Cross-Origin Resource Sharing (CORS)
if settings.ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin).strip() for origin in settings.ALLOWED_ORIGINS if str(origin).strip()],
        allow_credentials=True,
        allow_methods=["*"], # Allows all standard methods (GET, POST, PUT, DELETE, etc.)
        allow_headers=["*"], # Allows all headers
    )
else:
    # Fallback if no origins are specified (less secure, for very open dev perhaps)
    logger.warning("No ALLOWED_ORIGINS configured. Allowing all origins for CORS (development mode).")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# --- API Routers ---
# Include routers from the API endpoints module
app.include_router(doc_endpoints.router, prefix=f"{settings.API_V1_STR}/documents", tags=["Documents Management"])
app.include_router(query_endpoints.router, prefix=f"{settings.API_V1_STR}/queries", tags=["Document Querying & Theming"])

# --- Root Endpoint ---
@app.get("/", tags=["Root"])
async def read_root():
    return {"message": f"Welcome to {settings.PROJECT_NAME} API. Visit /docs for API documentation."}

# --- Main execution for Uvicorn (if running this file directly) ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server directly from main.py...")
    # Uvicorn expects "module_name:app_instance_name"
    # If main.py is in app/, then it's "app.main:app" from backend/
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")