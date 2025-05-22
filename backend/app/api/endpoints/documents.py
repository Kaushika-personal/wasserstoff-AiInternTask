from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks, Query, Path
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.crud import crud_document
from app.models.document import DocumentResponse, DocumentCreate # Pydantic models
from app.services.document_processor import DocumentFileProcessor
from app.services.embedding_service import get_embedding_service, EmbeddingService
from typing import List
import logging
import datetime
import hashlib # For initial hash check

router = APIRouter()
logger = logging.getLogger(__name__)

# This function runs in the background
async def background_process_document_and_embed(
    db: Session, 
    doc_id_in_db: str, # The ID of the DB record
    file_bytes: bytes, 
    filename: str
):
    try:
        logger.info(f"Background task started: Processing doc_id: {doc_id_in_db}, filename: {filename}")
        crud_document.update_document_details(db, document_id=doc_id_in_db, status="processing")
        
        doc_file_processor = DocumentFileProcessor()
        # This call processes the file (OCR if needed) and returns structured data
        processed_data = await doc_file_processor.process_uploaded_file(file_bytes, filename, doc_id_in_db)
        
        # Update DB with content_hash and total_pages from processed_data
        crud_document.update_document_details(
            db, 
            document_id=doc_id_in_db, 
            content_hash=processed_data.content_hash,
            total_pages=processed_data.total_pages
        )
        
        # Now, add to vector store
        embedding_service = get_embedding_service()
        embedding_service.add_processed_document_to_vector_store(processed_data)
        
        crud_document.update_document_details(
            db, 
            document_id=doc_id_in_db, 
            status="processed", 
            processed_at=datetime.datetime.now(datetime.timezone.utc)
        )
        logger.info(f"Background task finished: Successfully processed and embedded document: {doc_id_in_db} - {filename}")

    except ValueError as ve: # Specific errors like unsupported file type
        logger.error(f"ValueError during background processing for doc {doc_id_in_db}: {ve}", exc_info=True)
        crud_document.update_document_details(db, document_id=doc_id_in_db, status=f"error_validation: {str(ve)[:150]}")
    except Exception as e:
        logger.error(f"Unhandled error processing document {doc_id_in_db} in background: {e}", exc_info=True)
        crud_document.update_document_details(db, document_id=doc_id_in_db, status=f"error_processing: {str(e)[:150]}")


@router.post("/upload", response_model=DocumentResponse, status_code=202) # 202 Accepted for background tasks
async def upload_new_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided with the upload.")
    
    # Read file content
    file_bytes = await file.read()
    await file.seek(0) # Reset file pointer in case it's used again (FastAPI might handle this)

    # Optional: Check for duplicates by content hash *before* creating DB entry
    # This avoids creating multiple DB entries for the exact same file content if re-uploaded.
    temp_content_hash = hashlib.md5(file_bytes).hexdigest()
    existing_doc_by_hash = crud_document.get_document_by_hash(db, temp_content_hash)
    if existing_doc_by_hash:
        logger.info(f"File '{file.filename}' with hash {temp_content_hash} already exists with ID {existing_doc_by_hash.id} and status {existing_doc_by_hash.status}.")
        if existing_doc_by_hash.status == "processed":
            # If already processed, just return the existing document's info
            return existing_doc_by_hash
        elif existing_doc_by_hash.status == "processing" or existing_doc_by_hash.status == "uploaded":
            # If being processed or just uploaded, inform user (could be an error or just info)
            raise HTTPException(status_code=409, detail=f"This document content is already uploaded (ID: {existing_doc_by_hash.id}) and is currently '{existing_doc_by_hash.status}'.")
        # If status is 'error', could allow re-processing, or make user delete first. For now, treat as conflict.
        # This part of logic can be refined based on desired behavior for duplicates.

    # Create initial DB entry for the document
    doc_pydantic_create = DocumentCreate(
        filename=file.filename,
        file_type=file.content_type or os.path.splitext(file.filename)[1].lower()
        # total_pages and extracted_metadata will be updated after processing
    )
    # Pass the temp_content_hash here if you want it stored immediately
    db_document_record = crud_document.create_document_entry(db, doc_create_data=doc_pydantic_create, content_hash=temp_content_hash)
    
    # Add the processing task to run in the background
    background_tasks.add_task(
        background_process_document_and_embed, 
        db, 
        str(db_document_record.id), # Pass the ID of the created DB record
        file_bytes, 
        file.filename
    )
    
    logger.info(f"Document {db_document_record.id} ('{db_document_record.filename}') upload accepted. Processing scheduled in background.")
    return db_document_record


@router.get("/", response_model=List[DocumentResponse])
def list_all_documents(
    skip: int = Query(0, ge=0), 
    limit: int = Query(default=75, ge=1, le=200), # Default to 75 as per requirement
    db: Session = Depends(get_db)
):
    documents_from_db = crud_document.get_documents(db, skip=skip, limit=limit)
    return documents_from_db

@router.get("/{document_id}", response_model=DocumentResponse)
def get_single_document_details(
    document_id: str = Path(..., title="The ID of the document to retrieve"), 
    db: Session = Depends(get_db)
):
    db_document_record = crud_document.get_document(db, document_id)
    if db_document_record is None:
        raise HTTPException(status_code=404, detail=f"Document with ID '{document_id}' not found.")
    return db_document_record

@router.delete("/{document_id}", response_model=DocumentResponse)
def delete_single_document(
    document_id: str = Path(..., title="The ID of the document to delete"),
    db: Session = Depends(get_db)
):
    db_doc_to_delete = crud_document.get_document(db, document_id)
    if db_doc_to_delete is None:
        raise HTTPException(status_code=404, detail=f"Document with ID '{document_id}' not found, cannot delete.")

    # 1. Delete from vector store
    try:
        embedding_service = get_embedding_service()
        embedding_service.delete_document_vectors(document_id) # Pass the string ID
        logger.info(f"Successfully initiated deletion of vectors for document ID {document_id}.")
    except Exception as e:
        logger.error(f"Error deleting vectors for document {document_id}: {e}. Proceeding with DB deletion.", exc_info=True)
        # Decide if failure to delete from vector store should halt DB deletion.
        # For now, we'll proceed but log the error.

    # 2. Delete from database
    deleted_db_doc = crud_document.delete_document_entry(db, document_id)
    if deleted_db_doc: # Should always be true if found earlier
        logger.info(f"Successfully deleted document record for ID {document_id} from database.")
        return deleted_db_doc
    else:
        # This case should ideally not be reached if the first check passed.
        logger.error(f"Document {document_id} was found but could not be deleted from DB.")
        raise HTTPException(status_code=500, detail="Error deleting document from database after vector store operation.")