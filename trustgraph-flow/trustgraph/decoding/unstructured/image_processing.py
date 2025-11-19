"""
Image-specific helpers: content-type guard and OCR.
"""

from __future__ import annotations

import logging
from io import BytesIO
from typing import List, Optional

from .content_type_detection import IMAGE_KIND_TO_CONTENT_TYPE

logger = logging.getLogger(__name__)


# // ---> Processor._is_image_content_type > [is_image_content_type] > guards OCR branch
def is_image_content_type(content_type: Optional[str]) -> bool:
    if not content_type:
        return False
    lowered = content_type.strip().lower()
    return lowered in IMAGE_KIND_TO_CONTENT_TYPE.values()


# // ---> Processor._ocr_image > [ocr_image] > returns textual segments from OCR
def ocr_image(blob: bytes) -> List[str]:
    """
    Perform OCR on an image blob using pytesseract if available.
    Falls back to empty list if library or system dependency is missing.
    """
    try:
        from PIL import Image  # type: ignore
        import pytesseract  # type: ignore
    except Exception as import_error:
        logger.warning(
            "pytesseract/Pillow not available for image OCR: %s; skipping OCR.",
            import_error,
        )
        return []

    try:
        image = Image.open(BytesIO(blob))
    except Exception as open_error:
        logger.warning("Unable to open image for OCR: %s", open_error)
        return []

    try:
        text = pytesseract.image_to_string(image, lang="eng")
    except Exception as ocr_error:
        logger.warning("pytesseract OCR failed: %s", ocr_error)
        return []

    cleaned = (text or "").strip()
    return [cleaned] if cleaned else []


