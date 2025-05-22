import asyncio
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import numpy as np
from paddleocr import PaddleOCR # type: ignore
from app.core.config import settings
import logging
import io
import os
logger = logging.getLogger(__name__)

class OCRService:
    _instance = None
    _ocr_engine = None
    # Run OCR in a thread pool to avoid blocking asyncio event loop
    _executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 1) 

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OCRService, cls).__new__(cls)
            try:
                # This is the line causing the error:
                cls._ocr_engine = PaddleOCR(use_angle_cls=True, lang=settings.OCR_LANG) # OLD LINE
                logger.info(f"PaddleOCR engine initialized for language: {settings.OCR_LANG}.")
            except Exception as e:
                logger.error(f"Failed to initialize PaddleOCR engine: {e}", exc_info=True)
                cls._ocr_engine = None
        return cls._instance

    async def _run_ocr_in_executor(self, img_array: np.ndarray) -> list:
        if not self._ocr_engine:
            return []
        loop = asyncio.get_event_loop()
        # The actual OCR call is blocking, so run it in executor
        ocr_results = await loop.run_in_executor(self._executor, self._ocr_engine.ocr, img_array, False) # cls=False for text detection + recognition
        return ocr_results if ocr_results else []


    async def extract_text_from_image_bytes(self, image_bytes: bytes) -> str:
        if not self._ocr_engine:
            logger.warning("OCR engine not available. Skipping OCR.")
            return ""
        try:
            # Convert bytes to PIL Image, then to numpy array
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img_array = np.array(image)
            
            ocr_results = await self._run_ocr_in_executor(img_array)
            
            text_lines = []
            # PaddleOCR returns a list of lists, even for a single image.
            # Each inner list contains [bbox, (text, confidence_score)]
            if ocr_results and ocr_results[0] is not None: 
                for line_item in ocr_results[0]: # Process lines from the first (and only) image result
                    text_lines.append(line_item[1][0]) # text content is at index 1, then 0
            return "\n".join(text_lines)
        except Exception as e:
            logger.error(f"Error during OCR processing: {e}", exc_info=True)
            return ""

# Singleton accessor
_ocr_service_instance = None

def get_ocr_service():
    global _ocr_service_instance
    if _ocr_service_instance is None:
        _ocr_service_instance = OCRService()
    return _ocr_service_instance