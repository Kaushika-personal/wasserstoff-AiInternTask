from langchain_google_genai import GoogleGenerativeAIEmbeddings # New
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document as LangchainDoc # Alias
from app.core.config import settings
from app.models.document import ProcessedDocumentData # Internal Pydantic model
import logging
from typing import List, Dict, Any, Optional, Tuple # <--- ADD Tuple HERE
import os

logger = logging.getLogger(__name__)

class EmbeddingService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Initialize expensive resources here
            cls._instance.embeddings_model = GoogleGenerativeAIEmbeddings(
                model=settings.GEMINI_EMBEDDING_MODEL_NAME, # Use your Gemini embedding model
                google_api_key=str(settings.GEMINI_API_KEY) ,
                task_type="retrieval_document"  # Ensure this is present
            )
            cls._instance.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
                length_function=len,
                add_start_index=True, # Useful for more precise citation later
            )
            # Ensure vector DB path exists before Chroma tries to use it
            os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)
            cls._instance.vector_store = Chroma(
                persist_directory=settings.VECTOR_DB_PATH,
                embedding_function=cls._instance.embeddings_model
            )
            logger.info(f"Chroma vector store initialized/loaded from {settings.VECTOR_DB_PATH}")
        return cls._instance
        
    def add_processed_document_to_vector_store(self, processed_doc_data: ProcessedDocumentData):
        langchain_docs_to_add: List[LangchainDoc] = []
        
        for page_content in processed_doc_data.pages:
            # Split page text into chunks
            page_chunks = self.text_splitter.split_text(page_content.text_content)
            
            for i, chunk_text in enumerate(page_chunks):
                metadata = {
                    "document_id": processed_doc_data.doc_id,
                    "filename": processed_doc_data.filename,
                    "page_number": page_content.page_number,
                    "chunk_index_on_page": i, # Index of this chunk within the page
                    # "start_index_in_page": chunk.metadata.get("start_index") # If RecursiveCharacterTextSplitter provides it
                }
                langchain_docs_to_add.append(LangchainDoc(page_content=chunk_text, metadata=metadata))
        
        if langchain_docs_to_add:
            try:
                self.vector_store.add_documents(langchain_docs_to_add)
                # self.vector_store.persist() # Persist is often done automatically by Chroma or on specific calls.
                                          # Check Chroma docs if explicit persist is always needed after add_documents.
                                          # For some versions/configurations, it is.
                logger.info(f"Added {len(langchain_docs_to_add)} chunks for document ID {processed_doc_data.doc_id} ('{processed_doc_data.filename}') to vector store.")
            except Exception as e:
                logger.error(f"Error adding document {processed_doc_data.doc_id} to vector store: {e}", exc_info=True)
                raise
        else:
            logger.warning(f"No text chunks generated for document {processed_doc_data.doc_id} ('{processed_doc_data.filename}'). Document might be empty or too short.")

    def similarity_search_with_scores(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[LangchainDoc, float]]: # <--- This uses Tuple
        """
        Performs similarity search and returns documents with their relevance scores.
        filter_metadata example: {"document_id": "some_doc_id"}
        ChromaDB filter syntax: https://docs.trychroma.com/usage-guide#using-where-filters
        """
        search_kwargs = {'k': k}
        if filter_metadata:
            search_kwargs['filter'] = filter_metadata
        
        try:
            # similarity_search_with_relevance_scores returns (Document, score)
            # Score is cosine distance (lower is more similar) or other depending on collection setting
            # We might want to convert it to similarity (e.g. 1 - distance) if needed
            results_with_scores = self.vector_store.similarity_search_with_relevance_scores(
                query, **search_kwargs
            )
            logger.info(f"Similarity search for '{query[:30]}...' (k={k}, filter={filter_metadata}) found {len(results_with_scores)} chunks.")
            return results_with_scores
        except Exception as e:
            # This can happen if the collection is empty or doesn't exist.
            logger.error(f"Error during similarity search: {e}", exc_info=True)
            return []
            
    def delete_document_vectors(self, document_id: str):
        """
        Deletes all vectors/chunks associated with a given document_id from Chroma.
        Note: This relies on Chroma's ability to delete by metadata.
        """
        try:
            # Chroma's delete method uses 'where' clause for filtering.
            # The LangChain wrapper might abstract this. If direct access is needed:
            # self.vector_store._collection.delete(where={"document_id": document_id})
            # For the LangChain wrapper, we might need to get IDs first if direct metadata deletion is not smooth.
            # A common pattern is to get all IDs for a metadata filter, then delete by those IDs.
            # However, let's try the simpler approach first.
            
            # This is how you'd do it if you had the explicit IDs of the chunks:
            # ids_to_delete = [...] # You'd need to query for these first based on document_id
            # if ids_to_delete: self.vector_store.delete(ids=ids_to_delete)

            # Chroma's LangChain wrapper's delete method might not directly support `where` filters.
            # The most reliable way with the standard LangChain Chroma wrapper is to get the IDs of the chunks
            # belonging to the document and then delete them by their unique IDs.
            # This is more involved than a simple `delete_by_metadata`.
            # For now, we'll log and note this limitation.
            # A workaround is to rebuild the collection without the doc, or handle at application level (mark as deleted).

            # Getting all chunk IDs for a document_id
            collection = self.vector_store._collection # Access underlying chromadb collection
            if collection:
                results = collection.get(where={"document_id": document_id}, include=[]) # include=[] means only IDs
                ids_to_delete = results.get('ids')
                if ids_to_delete:
                    logger.info(f"Found {len(ids_to_delete)} chunk IDs to delete for document_id: {document_id}")
                    collection.delete(ids=ids_to_delete)
                    # self.vector_store.persist() # If needed after deletion
                    logger.info(f"Successfully deleted {len(ids_to_delete)} chunks for document_id {document_id} from vector store.")
                else:
                    logger.info(f"No chunks found for document_id {document_id} to delete.")
            else:
                logger.warning("Chroma collection not available for deletion.")

        except Exception as e:
            logger.error(f"Error deleting vectors for document_id {document_id}: {e}", exc_info=True)


# Singleton accessor
_embedding_service_instance = None

def get_embedding_service():
    global _embedding_service_instance
    if _embedding_service_instance is None:
        _embedding_service_instance = EmbeddingService()
    return _embedding_service_instance