import requests
import json
from tabulate import tabulate # For pretty-printing tables

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1" # Your FastAPI base URL
QUERY_ENDPOINT = f"{API_BASE_URL}/queries/"

def make_query(query_text: str, document_ids: list | None = None, top_k: int = 3):
    """Makes a POST request to the query endpoint."""
    payload = {
        "query": query_text,
        "document_ids": document_ids,
        "top_k_chunks_per_doc": top_k
    }
    headers = {
        "Content-Type": "application/json"
    }
    try:
        print(f"\nSending query: {query_text}")
        if document_ids:
            print(f"Targeting document IDs: {document_ids}")
        else:
            print("Targeting all processed documents.")

        response = requests.post(QUERY_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status() # Raises an HTTPError for bad responses (4XX or 5XX)
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        print(f"Response content: {response.text}")
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Connection error occurred: {conn_err} (Is the FastAPI server running at {API_BASE_URL}?)")
    except requests.exceptions.Timeout as timeout_err:
        print(f"Timeout error occurred: {timeout_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"An unexpected error occurred with the request: {req_err}")
    except json.JSONDecodeError:
        print("Error decoding JSON response from the server.")
        print(f"Response content: {response.text}")
    return None

def display_results(query_response):
    """Formats and prints the query response."""
    if not query_response:
        print("No response received from the server.")
        return

    print("\n--- Individual Document Responses (Tabular) ---")
    if query_response.get("tabular_individual_answers"):
        # Prepare data for tabulate
        table_data = []
        headers = ["Document ID", "Filename", "Extracted Answer", "Citation"]
        for item in query_response["tabular_individual_answers"]:
            table_data.append([
                item.get("Document ID", "N/A"),
                item.get("Filename", "N/A"),
                item.get("Extracted Answer", "N/A"),
                item.get("Citation", "N/A")
            ])
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    else:
        print("No tabular individual answers found in the response.")

    print("\n\n--- Final Synthesized Response (Chat Format) ---")
    synthesized_overview = query_response.get("synthesized_overview")
    if synthesized_overview:
        print(f"Overall Summary: {synthesized_overview}\n")
    else:
        print("No synthesized overview provided.\n")

    themes = query_response.get("themes", [])
    if themes:
        for i, theme in enumerate(themes):
            theme_label = theme.get("theme_label", f"Unnamed Theme {i+1}")
            theme_summary = theme.get("theme_summary", "No summary.")
            supporting_docs = theme.get("supporting_documents", [])
            
            doc_ids_str = ", ".join([doc.get("document_id", "Unknown_ID") for doc in supporting_docs])
            
            print(f"Theme {i+1} – {theme_label}:")
            # Original example format: Documents (DOC001, DOC002) highlight ...
            # We can try to mimic this with the summary
            # For a simpler display now:
            print(f"  Summary: {theme_summary}")
            if supporting_docs:
                print(f"  Supporting Document IDs: ({doc_ids_str})")
            print("-" * 20) # Separator
    else:
        print("No themes were identified in the response.")

if __name__ == "__main__":
    # --- Example Usage ---
    # 1. Make sure your FastAPI server is running.
    # 2. Make sure you have uploaded and processed some documents.
    # 3. Get the ID of a processed document (e.g., from Swagger UI or your DB).

    # Example Query 1: Query all processed documents
    # my_query = "What are the main topics discussed?"
    # response_data = make_query(query_text=my_query, document_ids=None) # or document_ids=[]

    # Example Query 2: Query a specific document
    # Replace with an ACTUAL ID of a document you have processed
    specific_doc_id = "YOUR_PROCESSED_DOCUMENT_ID_HERE" # <--- IMPORTANT: REPLACE THIS
    my_query_specific = "Summarize this document."
    
    if specific_doc_id == "YOUR_PROCESSED_DOCUMENT_ID_HERE":
        print("Please update 'specific_doc_id' in the script with an actual document ID.")
        print("Attempting query across all documents for demonstration instead.")
        response_data = make_query(query_text="What information is available?", document_ids=None)
    else:
        response_data = make_query(query_text=my_query_specific, document_ids=[specific_doc_id])

    if response_data:
        # You can print the raw JSON too for debugging:
        # print("\nRaw JSON Response:")
        # print(json.dumps(response_data, indent=2))
        
        display_results(response_data)