from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid

class DocumentMetadataBase(BaseModel):
    filename: str
    file_type: Optional[str] = None
    total_pages: Optional[int] = None
    extracted_metadata: Optional[Dict[str, Any]] = None

class DocumentCreate(DocumentMetadataBase):
    pass

class DocumentResponse(DocumentMetadataBase): # Renamed from Document to avoid conflict
    id: str # Keep as str to match DB schema if using string UUID
    status: str
    content_hash: Optional[str] = None
    created_at: datetime
    processed_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    model_config = ConfigDict(from_attributes=True) # Replaces orm_mode

# For internal processing by services
class ProcessedPageContent(BaseModel):
    page_number: int
    text_content: str
    # Optional: Add bounding boxes for text if you want fine-grained citation later

class ProcessedDocumentData(BaseModel):
    doc_id: str
    filename: str
    full_text: str # Combined text from all pages
    pages: List[ProcessedPageContent]
    content_hash: str
    total_pages: int