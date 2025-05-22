from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.models.query import QueryRequest, QueryResponse # Pydantic models
from app.services.query_service import get_query_service, DocumentQueryService
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/", response_model=QueryResponse)
async def create_and_process_query(
    query_input: QueryRequest, # Renamed from request to avoid conflict with FastAPI's Request object
    db: Session = Depends(get_db)
):
    if not query_input.query.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")
    
    doc_query_service: DocumentQueryService = get_query_service()
    try:
        query_response_data = await doc_query_service.process_query_across_documents(db, query_input)
        return query_response_data
    except ValueError as ve: # Specific errors like "no docs processed" or "IDs not found"
        logger.warning(f"Value error during query processing for query '{query_input.query[:50]}...': {ve}")
        # Use 404 if it's about documents not being found/processed, 400 for other input errors
        status_code = 404 if "document" in str(ve).lower() or "processed" in str(ve).lower() else 400
        raise HTTPException(status_code=status_code, detail=str(ve))
    except Exception as e:
        logger.error(f"Unhandled error processing query '{query_input.query[:50]}...': {e}", exc_info=True)
        # Generic 500 for unexpected server errors
        raise HTTPException(status_code=500, detail=f"An internal server error occurred while processing the query.")