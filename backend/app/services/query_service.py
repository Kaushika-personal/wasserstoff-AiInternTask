import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI # New
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# from langchain.schema.runnable import RunnablePassthrough # If using LCEL chains
from app.core.config import settings
from app.services.embedding_service import get_embedding_service, EmbeddingService
from app.services.theme_service import get_theme_service, ThemeIdentificationService
from app.models.query import QueryRequest, QueryResponse, IndividualDocumentAnswer, Citation, Theme
from app.schemas.document import DocumentDB as DocumentDBSchema # SQLAlchemy model
from sqlalchemy.orm import Session
from app.crud import crud_document
import logging
from typing import List, Dict, Any, Tuple, Optional
import re

logger = logging.getLogger(__name__)

class DocumentQueryService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.embedding_service: EmbeddingService = get_embedding_service()
            cls._instance.theme_service: ThemeIdentificationService = get_theme_service()
            cls._instance.llm = ChatGoogleGenerativeAI(
                model=settings.GEMINI_GENERATIVE_MODEL_NAME, # e.g., "gemini-pro" or "gemini-1.5-flash-latest"
                google_api_key=settings.GEMINI_API_KEY,
                temperature=0.1, # Adjust as needed
                convert_system_message_to_human=True # Gemini might prefer system messages converted
            )
        return cls._instance

    async def _generate_answer_and_citations_for_single_doc(
        self, 
        user_query: str, 
        doc_db_entry: DocumentDBSchema, 
        top_k_chunks_to_retrieve: int
    ) -> IndividualDocumentAnswer:
        
        doc_id_str = str(doc_db_entry.id)
        doc_filename = doc_db_entry.filename
        
        # 1. Retrieve relevant chunks using embedding service
        # The filter for ChromaDB should be like: {"document_id": doc_id_str}
        # Note: embedding_service.similarity_search_with_scores expects filter_metadata
        # For Chroma, the filter for a specific field value is {"field_name": "value"}
        relevant_chunks_with_scores: List[Tuple[Any, float]] = self.embedding_service.similarity_search_with_scores(
            query=user_query, 
            k=top_k_chunks_to_retrieve, 
            filter_metadata={"document_id": doc_id_str} # Filter for THIS document
        )

        if not relevant_chunks_with_scores:
            return IndividualDocumentAnswer(
                document_id=doc_id_str,
                filename=doc_filename,
                extracted_answer="No relevant information found in this document for the query.",
                citations=[]
            )

        # Prepare context and citation objects
        context_parts = []
        citations_pydantic: List[Citation] = []
        
        for chunk_doc, score in relevant_chunks_with_scores:
            context_parts.append(chunk_doc.page_content)
            citations_pydantic.append(Citation(
                document_id=doc_id_str,
                filename=doc_filename,
                page_number=chunk_doc.metadata.get("page_number"),
                snippet=chunk_doc.page_content[:200] + "...", # Truncate for brevity
                relevance_score=round(1 - score, 4) if score is not None else None # Assuming score is distance, convert to similarity
            ))
        
        context_for_llm = "\n\n---\n\n".join(context_parts) # Separator for clarity

        # 2. Generate answer using LLM based on context
        # Updated prompt to explicitly ask for citations within the answer text itself
        answer_gen_prompt_str = (
            "You are a meticulous AI assistant. Your task is to answer the user's query based *ONLY* on the provided context from a specific document. "
            "If the information is not present in the context, clearly state 'The document does not provide an answer to this query.' "
            "Be concise and directly answer the question. "
            "When you use information from the context, cite the page number(s) directly in your answer using the format [Page X] or [Pages X, Y]. "
            "The context below is from the document named '{doc_filename}'.\n\n"
            "Context from '{doc_filename}':\n\"\"\"\n{context_for_llm}\n\"\"\"\n\n"
            "User Query: {user_query}\n\n"
            "Answer (with inline citations like [Page X] where information is found):"
        )
        prompt = ChatPromptTemplate.from_template(answer_gen_prompt_str)
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            llm_generated_answer_text = await chain.ainvoke({
                "doc_filename": doc_filename,
                "context_for_llm": context_for_llm,
                "user_query": user_query
            })
        except Exception as e:
            logger.error(f"LLM query failed for doc {doc_id_str} ('{doc_filename}'), query '{user_query[:30]}...': {e}", exc_info=True)
            return IndividualDocumentAnswer(
                document_id=doc_id_str,
                filename=doc_filename,
                extracted_answer="Error generating answer from this document due to an LLM issue.",
                citations=citations_pydantic, # Still provide retrieved chunks as citations
                error_message=f"LLM generation error: {str(e)[:100]}"
            )
        
        # (Optional Post-processing for citations if LLM doesn't do it well)
        # For now, we rely on the LLM to embed citations like [Page X].
        # The `citations_pydantic` list contains all *retrieved* chunks' page numbers.

        return IndividualDocumentAnswer(
            document_id=doc_id_str,
            filename=doc_filename,
            extracted_answer=llm_generated_answer_text.strip(),
            citations=citations_pydantic # These are based on retrieved chunks
        )

    async def process_query_across_documents(
        self, 
        db: Session, 
        query_request: QueryRequest
    ) -> QueryResponse:
        
        # 1. Determine which documents to query
        if query_request.document_ids:
            docs_to_query_from_db = crud_document.get_documents_by_ids(db, query_request.document_ids)
            # Filter for only processed documents among the requested ones
            processed_docs_to_query = [doc for doc in docs_to_query_from_db if doc.status == "processed"]
            if not processed_docs_to_query and docs_to_query_from_db:
                 raise ValueError("None of the specified documents are processed and ready for querying.")
            if not docs_to_query_from_db: # If IDs were given but none found
                raise ValueError("Specified document IDs not found in the database.")
        else:
            # Query all documents that are processed
            processed_docs_to_query = crud_document.get_all_processed_document_db_entries(db)
            if not processed_docs_to_query:
                 raise ValueError("No documents have been processed yet. Upload and process documents first.")
        
        if not processed_docs_to_query:
            # This case should ideally be caught above, but as a safeguard:
            return QueryResponse(query_text=query_request.query, synthesized_overview="No documents available or processed to answer this query.", themes=[], individual_document_answers=[])

        logger.info(f"Querying {len(processed_docs_to_query)} documents for: '{query_request.query[:50]}...'")

        # 2. Concurrently generate answers for each document
        answer_generation_tasks = [
            self._generate_answer_and_citations_for_single_doc(
                query_request.query, 
                doc_db_entry, 
                query_request.top_k_chunks_per_doc
            )
            for doc_db_entry in processed_docs_to_query
        ]
        
        individual_doc_answers_results: List[IndividualDocumentAnswer] = await asyncio.gather(*answer_generation_tasks, return_exceptions=True)
        
        # Filter out exceptions and handle them (already handled within _generate_answer_for_doc by setting error_message)
        final_individual_doc_answers: List[IndividualDocumentAnswer] = []
        for result in individual_doc_answers_results:
            if isinstance(result, Exception):
                # This should ideally not happen if _generate_answer_and_citations_for_single_doc catches its own errors
                logger.error(f"Unhandled exception during answer generation for a document: {result}", exc_info=True)
                # Create a placeholder error response if needed, though the called function should do this
            elif result is not None: # Ensure result is not None
                final_individual_doc_answers.append(result)

        # 3. Identify themes from the collected answers
        identified_themes, synthesized_overview_text = await self.theme_service.identify_and_summarize_themes(
            query_request.query,
            final_individual_doc_answers 
        )
        
        # 4. Prepare tabular responses for frontend (as per task requirement example)
        tabular_answers_display: List[Dict[str, Any]] = []
        for ans in final_individual_doc_answers:
            # Create a concise citation string for the table
            citation_str_parts = []
            if ans.citations:
                # Extract page numbers from the LLM's answer text if possible
                llm_page_citations = set(re.findall(r'\[Page\s*(\d+)(?:,\s*\d+)*\]', ans.extracted_answer))
                if llm_page_citations:
                    # Convert found page numbers to int for sorting, then back to str
                    cited_pages = sorted([int(p) for p in llm_page_citations])
                    citation_str_parts.append(f"Page(s) {', '.join(map(str, cited_pages))}")
                else: # Fallback to retrieved chunk page numbers
                    unique_pages_from_chunks = sorted(list(set(c.page_number for c in ans.citations if c.page_number is not None)))
                    if unique_pages_from_chunks:
                         citation_str_parts.append(f"Page(s) {', '.join(map(str, unique_pages_from_chunks))}")
            
            tabular_answers_display.append({
                "Document ID": ans.document_id,
                "Filename": ans.filename,
                "Extracted Answer": ans.extracted_answer,
                "Citation": "; ".join(citation_str_parts) if citation_str_parts else "N/A"
            })

        return QueryResponse(
            query_text=query_request.query,
            synthesized_overview=synthesized_overview_text,
            themes=identified_themes,
            individual_document_answers=final_individual_doc_answers,
            tabular_individual_answers=tabular_answers_display
        )

# Singleton accessor
_query_service_instance = None

def get_query_service():
    global _query_service_instance
    if _query_service_instance is None:
        _query_service_instance = DocumentQueryService()
    return _query_service_instance