from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import uuid

class Citation(BaseModel):
    document_id: str
    filename: str
    page_number: Optional[int] = None
    # paragraph_number: Optional[int] = None # Extra credit if implemented
    # sentence_number: Optional[int] = None # Extra credit if implemented
    snippet: Optional[str] = Field(None, description="A short relevant text snippet from the document")
    relevance_score: Optional[float] = None

class IndividualDocumentAnswer(BaseModel):
    document_id: str
    filename: str
    extracted_answer: str
    citations: List[Citation] = []
    error_message: Optional[str] = Field(None, description="Error message if processing this document for the query failed")

class Theme(BaseModel):
    theme_id: str = Field(default_factory=lambda: f"theme_{uuid.uuid4().hex[:8]}")
    theme_label: str = Field(..., description="e.g., 'Regulatory Non-Compliance'")
    theme_summary: str = Field(..., description="A brief explanation of the theme")
    supporting_documents: List[Dict[str, str]] = Field(
        ..., description="List of {'document_id': '...', 'filename': '...'}"
    )

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User's query string")
    document_ids: Optional[List[str]] = Field(None, description="Query specific documents by ID, or all if None")
    top_k_chunks_per_doc: int = Field(default=3, ge=1, le=10, description="Number of chunks to retrieve per document")

class QueryResponse(BaseModel):
    query_text: str
    synthesized_overview: Optional[str] = Field(None, description="Overall LLM summary based on themes")
    themes: List[Theme] = []
    individual_document_answers: List[IndividualDocumentAnswer] = []
    
    # For tabular display as per requirements - this could be generated on the frontend too
    # but providing it from backend might be convenient for direct display.
    tabular_individual_answers: List[Dict[str, Any]] = Field(
        default=[], 
        description="Example: [{'Document ID': 'DOC001', 'Extracted Answer': '...', 'Citation': 'Page 4, Para 2'}]"
    )
    model_config = ConfigDict(from_attributes=True)