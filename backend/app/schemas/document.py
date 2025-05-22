import uuid
from sqlalchemy import Column, String, DateTime, JSON, Integer, Text
from sqlalchemy.sql import func
from app.db.base import Base # Import your Base

class DocumentDB(Base):
    __tablename__ = "documents"

    # Using String for UUID to be compatible with SQLite easily
    # For PostgreSQL, you might use: from sqlalchemy.dialects.postgresql import UUID
    # id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    filename = Column(String, index=True, nullable=False)
    file_type = Column(String, nullable=True)
    status = Column(String, default="uploaded", index=True) # e.g., uploaded, processing, processed, error
    content_hash = Column(String, unique=True, index=True, nullable=True)
    
    total_pages = Column(Integer, nullable=True)
    # You might want to store the full text in the DB or rely on the vector store + original file
    # For simplicity, we're not storing full text here to avoid DB bloat.
    # full_text_content = Column(Text, nullable=True) 
    
    processed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    # For updated_at, server_default and onupdate work well for auto-timestamps
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Store basic metadata, can be extended
    extracted_metadata = Column(JSON, nullable=True) # e.g., author, creation_date from doc properties