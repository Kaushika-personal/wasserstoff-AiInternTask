# AI Software Intern Task - Document Research & Theme Identification Chatbot

This project implements an interactive system as per the Wasserstoff AI Internship Task Document. It allows users to upload various document types, performs research across these documents based on user queries, identifies common themes, and provides detailed, cited responses.

## Table of Contents
1.  [Project Overview](#project-overview)
2.  [Features](#features)
    *   [Document Upload and Knowledge Base Creation](#document-upload-and-knowledge-base-creation)
    *   [Document Management & Query Processing](#document-management--query-processing)
    *   [Theme Identification & Cross-Document Synthesis](#theme-identification--cross-document-synthesis)
3.  [Technical Stack](#technical-stack)
4.  [Project Structure](#project-structure)
5.  [Setup and Installation](#setup-and-installation)
6.  [Running the Application](#running-the-application)
7.  [API Usage Workflow](#api-usage-workflow)
    *   [API Documentation](#api-documentation)
    *   [Uploading Documents](#uploading-documents)
    *   [Checking Document Status](#checking-document-status)
    *   [Querying Documents](#querying-documents)
8.  [Presentation of Results](#presentation-of-results)
9.  [Future Enhancements / Extra Credit Considerations](#future-enhancements--extra-credit-considerations)

## 1. Project Overview

The core objective of this project is to build an AI-powered system capable of ingesting a large set of documents (minimum 75), processing them to create a searchable knowledge base, and then allowing users to query this knowledge base in natural language. The system extracts relevant information, cites its sources at the page level, and synthesizes common themes found across multiple documents related to the query.

The backend is built using FastAPI, providing a RESTful API for all functionalities. It leverages Large Language Models (LLMs) via the Gemini API for embeddings and generative tasks, and employs various Python libraries for document processing, OCR, and data storage.

## 2. Features

### Document Upload and Knowledge Base Creation

*   **Multi-Format Document Upload:**
    *   Supports uploading documents in various formats including PDF (digital and scanned), DOCX, and common image formats (PNG, JPG, JPEG, TIFF).
    *   Handles a minimum of 75+ documents.
    *   Implemented via the `POST /api/v1/documents/upload` endpoint.
*   **OCR for Scanned Documents:**
    *   Scanned PDFs and image files are automatically processed using Optical Character Recognition (OCR) via PaddleOCR to extract textual content.
    *   This ensures that text from non-digital documents becomes searchable.
*   **Accurate Text Extraction:**
    *   Utilizes PyMuPDF (fitz) for extracting text from digital PDFs and python-docx for DOCX files.
    *   Aims for high fidelity in text extraction to ensure the quality of the knowledge base.
*   **Document Chunking:**
    *   The extracted text from each document is split into smaller, manageable chunks using `RecursiveCharacterTextSplitter` from LangChain. This is essential for effective semantic search and for fitting relevant context into LLM prompts.
    *   Chunk size and overlap are configurable (defaults: 1000 characters, 200 overlap).
*   **Embedding Generation:**
    *   Each text chunk is converted into a numerical vector representation (embedding) using Google's Gemini embedding models (e.g., `models/embedding-001`) via `langchain-google-genai`.
*   **Vector Store Integration:**
    *   Embeddings are stored in a ChromaDB vector database.
    *   ChromaDB allows for efficient similarity searches, enabling the retrieval of chunks most relevant to a user's query.
    *   The vector store is persisted locally (default: `./data/vector_db`).
*   **Background Processing:**
    *   Document parsing, OCR, chunking, and embedding are computationally intensive tasks. These are performed as background tasks using FastAPI's `BackgroundTasks` to keep the API responsive during uploads. The document status is updated as it moves through `uploaded` -> `processing` -> `processed` or `error`.
*   **Metadata Storage:**
    *   Basic metadata for each uploaded document (ID, filename, file type, status, creation/processing timestamps, content hash, total pages) is stored in a relational database (SQLite by default via SQLAlchemy).
    *   Content hashing helps in identifying and potentially skipping reprocessing of duplicate file content.

### Document Management & Query Processing

*   **Document Listing & Retrieval:**
    *   `GET /api/v1/documents/`: Lists all uploaded documents with their metadata and status.
    *   `GET /api/v1/documents/{document_id}`: Retrieves details for a specific document.
*   **Natural Language Queries:**
    *   Users can input queries in natural language via the `POST /api/v1/queries/` endpoint.
*   **Targeted or Broad Search:**
    *   Queries can be targeted at specific document IDs or run across all processed documents in the knowledge base.
*   **Semantic Search for Relevant Chunks:**
    *   The user's query is embedded.
    *   ChromaDB performs a similarity search to find the most relevant text chunks from the specified (or all) documents.
    *   The number of chunks retrieved per document (`top_k_chunks_per_doc`) is configurable.
*   **Contextual Answer Generation:**
    *   The retrieved relevant chunks form the context.
    *   This context, along with the user's query, is passed to a Gemini generative model (e.g., `gemini-1.5-flash-latest`).
    *   The LLM is prompted to answer the query based *only* on the provided context from each document.
*   **Individual Document Responses:**
    *   For each document searched, the system generates an extracted answer.
    *   If the document does not contain relevant information, the LLM is instructed to state so.
*   **Precise Citations (Page Level):**
    *   The LLM is prompted to include page number citations (e.g., `[Page X]`) directly within its generated answer text.
    *   The raw response also includes a list of `Citation` objects for each retrieved chunk, detailing the source document, filename, page number, a text snippet, and relevance score.
    *   This information is used to populate the "Citation" column in the tabular results.

### Theme Identification & Cross-Document Synthesis

*   **Collective Analysis of Responses:**
    *   The extracted answers (text content) from all relevant documents for a given query are analyzed collectively.
*   **Coherent Theme Identification:**
    *   **Text Vectorization:** Textual answers are vectorized using TF-IDF (Term Frequency-Inverse Document Frequency) to represent their content numerically.
    *   **Clustering:** Agglomerative Clustering (from scikit-learn) is applied to the TF-IDF vectors to group similar document answers together. Each cluster represents a potential theme.
    *   The system aims to identify multiple themes if present.
*   **LLM-Powered Theme Labeling and Summarization:**
    *   For each identified cluster (theme), a sample of its constituent document answers is passed to a Gemini generative model.
    *   The LLM is prompted to generate a concise `theme_label` (3-5 words) and a brief `theme_summary` (1-2 sentences) that captures the common topic of that cluster in relation to the original user query.
*   **Final Synthesized Overview:**
    *   An LLM generates a high-level synthesized overview (2-3 sentences) based on all the identified themes and their summaries, providing a concise summary of the collective findings.
*   **Citation Mapping (Document Level for Themes):**
    *   Each identified theme in the API response lists the `document_ids` and `filenames` of the documents that contributed to that theme.

## 3. Technical Stack

*   **Backend Framework:** FastAPI
*   **Programming Language:** Python (3.9+ recommended, tested with 3.12 via Conda)
*   **LLMs & Embeddings:** Google Gemini API (via `langchain-google-genai` and `google-generativeai` SDKs)
    *   Embedding Model: e.g., `models/embedding-001`
    *   Generative Model: e.g., `models/gemini-1.5-flash-latest`
*   **Vector Database:** ChromaDB (local persistence)
*   **OCR Engine:** PaddleOCR (with `paddlepaddle` CPU version)
*   **Document Parsing:**
    *   PDFs: PyMuPDF (fitz)
    *   DOCX: python-docx
    *   Images: Pillow
*   **Data Validation & Serialization:** Pydantic (V2 via `pydantic-settings`)
*   **Database ORM (Metadata):** SQLAlchemy
*   **Database (Metadata):** SQLite (default)
*   **Text Processing & ML:**
    *   LangChain (text splitting, LLM/embedding integrations, vector store abstractions)
    *   scikit-learn (TF-IDF, Agglomerative Clustering)
    *   NumPy
*   **API Interaction (for testing script):** `requests`
*   **Tabular Display (for testing script):** `tabulate`
*   **Asynchronous Server:** Uvicorn

## 4. Project Structure

wasserstoff/
├── backend/ # FastAPI backend application
│ ├── app/ # Core application logic
│ │ ├── api/ # API endpoint definitions (routers)
│ │ │ └── endpoints/ # Specific endpoint files (documents.py, queries.py)
│ │ ├── core/ # Configuration (config.py)
│ │ ├── crud/ # Database Create, Read, Update, Delete operations
│ │ ├── db/ # Database session, SQLAlchemy base
│ │ ├── models/ # Pydantic models for API I/O and internal data structures
│ │ ├── schemas/ # SQLAlchemy ORM models (database table definitions)
│ │ └── services/ # Business logic (document_processor, embedding_service, etc.)
│ ├── data/ # Auto-created for SQLite DB and ChromaDB vector store
│ ├── .env # Environment variables (API keys, paths) - MUST BE CREATED
│ └── requirements.txt # Python dependencies
├── venv/ # Virtual environment folder (Python 3.9, e.g. from python.org)
├── (or venv312/ etc.) # Or Conda environment (Python 3.12, e.g. wasserstoff_py312)
├── query_and_display.py # Optional script to test API and format output
└── README.md # This file



## 5. Setup and Installation

1.  **Clone the repository (if applicable).**
2.  **Install Python:** Python 3.9 or Python 3.12 is recommended. If using Anaconda for Python 3.12, ensure `conda` commands are available.
3.  **Create and Activate a Virtual Environment:**
    *   **Using standard `venv` (e.g., with Python 3.9 from python.org):**
        ```bash
        cd path/to/wasserstoff
        python -m venv venv
        .\venv\Scripts\Activate.ps1  # Windows PowerShell
        # source venv/bin/activate   # macOS/Linux
        ```
        (If PowerShell execution policy error: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process`)
    *   **Using `conda` (e.g., for Python 3.12):**
        Open Anaconda Prompt or a terminal where `conda` is active.
        ```bash
        conda create --name wasserstoff_py312 python=3.12
        conda activate wasserstoff_py312
        cd path/to/wasserstoff
        ```
4.  **Set up Environment Variables:**
    *   Navigate to the `backend/` directory.
    *   Create a file named `.env`.
    *   Add your Gemini API key:
        ```env
        GEMINI_API_KEY="YOUR_ACTUAL_GEMINI_API_KEY_HERE"
        # Optional: Specify different Gemini models if needed
        # GEMINI_EMBEDDING_MODEL_NAME="models/embedding-001"
        # GEMINI_GENERATIVE_MODEL_NAME="models/gemini-1.5-flash-latest"
        ```
        **IMPORTANT: Protect your API key. Do not commit the `.env` file to public repositories.**
5.  **Install Dependencies:**
    *   Navigate to the `backend/` directory (if not already there).
    *   Ensure your chosen virtual environment is active.
    *   Install requirements:
        ```bash
        pip install -r requirements.txt
        ```
        *This may take some time, especially for `paddlepaddle` and its dependencies.*
        *If you encountered issues with SQLite and `chromadb`, ensure your `requirements.txt` reflects the working versions (e.g., an older `chromadb` and `numpy<2.0`). The SQLite patch in `main.py` should be **commented out** if `pysqlite3` could not be installed.*

## 6. Running the Application

1.  **Activate your virtual environment** (e.g., `.\venv\Scripts\Activate.ps1` or `conda activate wasserstoff_py312`).
2.  **Navigate to the `backend/` directory** from your project root.
    ```bash
    # (venv) PS C:\Users\YourUser\Desktop\wasserstoff>
    cd backend
    # (venv) PS C:\Users\YourUser\Desktop\wasserstoff\backend>
    ```
3.  **Start the FastAPI server using Uvicorn:**
    ```bash
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    ```
    *   `--reload` enables auto-reloading on code changes (for development).
    *   The server will be accessible at `http://localhost:8000`.

## 7. API Usage Workflow

### API Documentation
Once the server is running, interactive API documentation (Swagger UI) is available at:
**`http://localhost:8000/docs`**

### Uploading Documents
1.  Go to `/docs`, expand `POST /api/v1/documents/upload`.
2.  Click "Try it out".
3.  Click "Choose File" and select a document (PDF, DOCX, PNG, JPG, etc.).
4.  Click "Execute".
5.  You will receive a `202 Accepted` response with the document's initial details, including its `id`. The document processing (OCR, embedding) happens in the background.
6.  Monitor the terminal running Uvicorn for logs related to background processing. PaddleOCR will download models on its first run for each model type.

### Checking Document Status
1.  Go to `/docs`, expand `GET /api/v1/documents/{document_id}`.
2.  Click "Try it out".
3.  Enter the `id` of a document you uploaded.
4.  Click "Execute".
5.  The response will show the document's details, including its `status` (e.g., "uploaded", "processing", "processed", "error").

### Querying Documents
1.  Ensure at least one document has a status of "processed".
2.  Go to `/docs`, expand `POST /api/v1/queries/`.
3.  Click "Try it out".
4.  Edit the "Request body" JSON:
    *   `"query"`: Your natural language question.
    *   `"document_ids"`:
        *   `null` or `[]` or omit the field: To query all processed documents.
        *   `["id1", "id2", ... ]`: To query specific document IDs.
    *   `"top_k_chunks_per_doc"`: (Optional, default 3) Number of relevant text chunks to retrieve per document for context.
    ```json
    // Example: Query all documents
    {
      "query": "What are the main safety protocols discussed?",
      "document_ids": null,
      "top_k_chunks_per_doc": 3
    }
    ```
    ```json
    // Example: Query specific document
    {
      "query": "Summarize document abc-123.",
      "document_ids": ["abc-123"],
      "top_k_chunks_per_doc": 3
    }
    ```
5.  Click "Execute".
6.  The response will be a JSON object containing the `query_text`, `synthesized_overview`, a list of `themes`, `individual_document_answers`, and `tabular_individual_answers`.

## 8. Presentation of Results

The API returns structured JSON data. For the presentation formats specified in the task document:

*   **Individual document responses in tabular format:**
    The `tabular_individual_answers` field in the query response provides data ready for tabular display. A frontend or a separate script (like the provided `query_and_display.py` example) would render this.
    *   Current Citation: Page-level (e.g., "Page(s) 1, 3"). Achieving "Page X, Para Y" requires significant enhancements in document parsing and LLM prompting.

*   **Final synthesized response in chat format:**
    The `synthesized_overview` and the `themes` list (with `theme_label`, `theme_summary`, and `supporting_documents`) provide the data for this. A frontend or script would format this into a chat-like presentation.

The provided `query_and_display.py` script (if used) offers a console-based example of how this data can be formatted.

## 9. Future Enhancements / Extra Credit Considerations

*   **Enhanced Granularity of Citations:** Implement paragraph or sentence-level citations. This requires more sophisticated document parsing to identify these structures and store their locations, as well as advanced LLM prompting and output parsing.
*   **Visual Representation/Mapping Interface:** A frontend could visually link citations to document sections or map themes to documents.
*   **Advanced Filtering Options:** Allow filtering of documents for querying based on metadata like date, author, document type (requires storing this metadata first).
*   **Targeted Querying UI:** Frontend selection/deselection of specific documents for querying.
*   **Robust Error Handling:** More specific error handling and retry mechanisms for external API calls (e.g., Gemini) and background tasks.
*   **Scalability:**
    *   Replace FastAPI's `BackgroundTasks` with a dedicated task queue like Celery with Redis/RabbitMQ for more robust background processing.
    *   Use a production-grade database like PostgreSQL instead of SQLite for metadata.
*   **Improved Theme Identification:** Experiment with different clustering algorithms, text representations (e.g., sentence embeddings for answer clustering), or more advanced LLM-based topic modeling techniques.
*   **User Authentication & Authorization:** If exposing this beyond a local tool.
*   **Comprehensive Testing:** Unit and integration tests for all components.
*   **Containerization:** Dockerize the application for easier deployment and environment consistency.
  
