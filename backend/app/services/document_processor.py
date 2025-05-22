import io
import fitz  # PyMuPDF
import docx
from typing import List, Tuple
from app.models.document import ProcessedPageContent, ProcessedDocumentData
from app.services.ocr_service import get_ocr_service
import logging
import hashlib
import os

logger = logging.getLogger(__name__)

class DocumentFileProcessor:
    def __init__(self):
        self.ocr_service = get_ocr_service()

    async def process_uploaded_file(self, file_bytes: bytes, filename: str, doc_id_in_db: str) -> ProcessedDocumentData:
        content_hash = hashlib.md5(file_bytes).hexdigest()
        file_extension = os.path.splitext(filename)[1].lower()
        
        pages_content: List[ProcessedPageContent] = []
        full_text_parts: List[str] = []
        
        if file_extension == ".pdf":
            pages_content, full_text_parts = await self._process_pdf(file_bytes, filename)
        elif file_extension == ".docx":
            pages_content, full_text_parts = self._process_docx(file_bytes, filename)
        elif file_extension in [".png", ".jpg", ".jpeg", ".tiff", ".tif"]:
            pages_content, full_text_parts = await self._process_image(file_bytes, filename)
        else:
            logger.error(f"Unsupported file type: {file_extension} for file {filename}")
            raise ValueError(f"Unsupported file type: {file_extension}")
            
        full_text = "\n\n".join(full_text_parts) # Join page texts with double newline
        
        return ProcessedDocumentData(
            doc_id=doc_id_in_db,
            filename=filename,
            full_text=full_text,
            pages=pages_content,
            content_hash=content_hash,
            total_pages=len(pages_content)
        )

    async def _process_pdf(self, file_bytes: bytes, filename: str) -> Tuple[List[ProcessedPageContent], List[str]]:
        pages_data: List[ProcessedPageContent] = []
        full_text_parts: List[str] = []
        try:
            pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                text = page.get_text("text")
                
                if not text.strip():  # If no text extracted, try OCR
                    logger.info(f"Page {page_num + 1} of PDF '{filename}' has no extractable text, attempting OCR.")
                    pix = page.get_pixmap(dpi=200) # Higher DPI for better OCR
                    img_bytes = pix.tobytes("png")
                    text = await self.ocr_service.extract_text_from_image_bytes(img_bytes)
                    if text.strip():
                        logger.info(f"OCR successful for page {page_num + 1} of PDF '{filename}'.")
                    else:
                        logger.warning(f"OCR yielded no text for page {page_num + 1} of PDF '{filename}'.")
                
                pages_data.append(ProcessedPageContent(page_number=page_num + 1, text_content=text))
                full_text_parts.append(text)
            pdf_document.close()
        except Exception as e:
            logger.error(f"Error processing PDF {filename}: {e}", exc_info=True)
            raise # Re-raise to be caught by background task handler
        return pages_data, full_text_parts

    def _process_docx(self, file_bytes: bytes, filename: str) -> Tuple[List[ProcessedPageContent], List[str]]:
        pages_data: List[ProcessedPageContent] = []
        full_text_parts: List[str] = []
        try:
            doc = docx.Document(io.BytesIO(file_bytes))
            current_page_text_parts = []
            # DOCX doesn't have explicit page breaks easily accessible.
            # We can treat the whole document as one "page" or try heuristics (e.g., paragraph count).
            # For simplicity here, treating as one page.
            for para in doc.paragraphs:
                current_page_text_parts.append(para.text)
            
            page_text = "\n".join(current_page_text_parts)
            pages_data.append(ProcessedPageContent(page_number=1, text_content=page_text))
            full_text_parts.append(page_text)
        except Exception as e:
            logger.error(f"Error processing DOCX {filename}: {e}", exc_info=True)
            raise
        return pages_data, full_text_parts

    async def _process_image(self, file_bytes: bytes, filename: str) -> Tuple[List[ProcessedPageContent], List[str]]:
        pages_data: List[ProcessedPageContent] = []
        full_text_parts: List[str] = []
        try:
            text = await self.ocr_service.extract_text_from_image_bytes(file_bytes)
            if text.strip():
                 logger.info(f"OCR successful for image '{filename}'.")
            else:
                logger.warning(f"OCR yielded no text for image '{filename}'.")
            pages_data.append(ProcessedPageContent(page_number=1, text_content=text))
            full_text_parts.append(text)
        except Exception as e:
            logger.error(f"Error processing image {filename} with OCR: {e}", exc_info=True)
            raise
        return pages_data, full_text_parts