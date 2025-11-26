"""
OCR decoders for various document types.
Each decoder is imported safely so missing dependencies don't break other decoders.
"""

import logging

logger = logging.getLogger(__name__)

# Import pdf_decoder (always available - uses pdf2image)
try:
    from . pdf_decoder import *
except ImportError as e:
    logger.warning(f"pdf_decoder not available: {e}")

# Import docx_decoder (requires Spire.Doc)
try:
    from . docx_decoder import *
except ImportError as e:
    logger.warning(f"docx_decoder not available: {e}")

# Import pptx_decoder (requires Spire.Presentation) - optional
try:
    from . pptx_decoder import *
except ImportError as e:
    logger.debug(f"pptx_decoder not available: {e}")
