from sqlalchemy.orm import Session
from app.schemas.document import DocumentDB as DocumentSchema # DB model
from app.models.document import DocumentCreate # Pydantic model for creation
import uuid
from typing import List, Optional
from datetime import datetime

def get_document(db: Session, document_id: str) -> Optional[DocumentSchema]:
    return db.query(DocumentSchema).filter(DocumentSchema.id == document_id).first()

def get_documents(db: Session, skip: int = 0, limit: int = 100) -> List[DocumentSchema]:
    return db.query(DocumentSchema).order_by(DocumentSchema.created_at.desc()).offset(skip).limit(limit).all()

def get_documents_by_ids(db: Session, document_ids: List[str]) -> List[DocumentSchema]:
    if not document_ids:
        return []
    return db.query(DocumentSchema).filter(DocumentSchema.id.in_(document_ids)).all()

def create_document_entry(db: Session, doc_create_data: DocumentCreate, content_hash: Optional[str] = None) -> DocumentSchema:
    db_doc = DocumentSchema(
        id=str(uuid.uuid4()), # Ensure string UUID
        filename=doc_create_data.filename,
        file_type=doc_create_data.file_type,
        total_pages=doc_create_data.total_pages, # Can be None initially
        extracted_metadata=doc_create_data.extracted_metadata, # Can be None initially
        content_hash=content_hash, # Can be None initially
        status="uploaded" 
    )
    db.add(db_doc)
    db.commit()
    db.refresh(db_doc)
    return db_doc

def update_document_details(
    db: Session, 
    document_id: str, 
    status: Optional[str] = None, 
    processed_at: Optional[datetime] = None, 
    total_pages: Optional[int] = None,
    content_hash: Optional[str] = None,
    extracted_metadata: Optional[dict] = None
) -> Optional[DocumentSchema]:
    db_doc = get_document(db, document_id)
    if db_doc:
        if status is not None:
            db_doc.status = status
        if processed_at is not None:
            db_doc.processed_at = processed_at
        if total_pages is not None:
            db_doc.total_pages = total_pages
        if content_hash is not None:
            db_doc.content_hash = content_hash
        if extracted_metadata is not None:
            db_doc.extracted_metadata = extracted_metadata
        
        db.commit()
        db.refresh(db_doc)
    return db_doc

def get_document_by_hash(db: Session, content_hash: str) -> Optional[DocumentSchema]:
    if not content_hash:
        return None
    return db.query(DocumentSchema).filter(DocumentSchema.content_hash == content_hash).first()

def get_all_processed_document_db_entries(db: Session) -> List[DocumentSchema]:
    return db.query(DocumentSchema).filter(DocumentSchema.status == "processed").all()

def delete_document_entry(db: Session, document_id: str) -> Optional[DocumentSchema]:
    db_doc = get_document(db, document_id)
    if db_doc:
        db.delete(db_doc)
        db.commit()
        return db_doc
    return None