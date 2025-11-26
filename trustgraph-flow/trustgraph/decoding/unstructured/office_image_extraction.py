"""
Office document image extraction utilities for DOCX and DOC files.
Extracts embedded images for processing by vision models using Spire.Doc.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

# Content types for DOCX and DOC
DOCX_CONTENT_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
DOC_CONTENT_TYPE = "application/msword"

# Content types that support image extraction
IMAGE_EXTRACTABLE_CONTENT_TYPES = {DOCX_CONTENT_TYPE, DOC_CONTENT_TYPE}


# // ---> unstructured_decoder.on_message > [is_image_extractable_content_type] > check if images can be extracted
def is_image_extractable_content_type(content_type: Optional[str]) -> bool:
    """Check if the content type supports image extraction."""
    if not content_type:
        return False
    return content_type.strip().lower() in IMAGE_EXTRACTABLE_CONTENT_TYPES


# // ---> unstructured_decoder.on_message > [extract_images_from_docx] > extract embedded images from DOCX using Spire.Doc
def extract_images_from_docx(blob: bytes) -> List[Tuple[bytes, str]]:
    """
    Extract all embedded images from a DOCX document using Spire.Doc.
    
    Args:
        blob: Raw bytes of the DOCX file
        
    Returns:
        List of tuples: (image_bytes, extension)
    """
    try:
        from spire.doc import Document, DocPicture
    except ImportError:
        logger.warning("spire-doc not installed; cannot extract images from DOCX")
        return []
    
    images = []
    temp_file = None
    
    try:
        # Spire.Doc requires a file path, so write to temp file
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
            f.write(blob)
            temp_file = f.name
        
        # Load document with Spire.Doc
        document = Document()
        document.LoadFromFile(temp_file)
        
        # Iterate over sections
        for s in range(document.Sections.Count):
            section = document.Sections[s]
            
            # Iterate over paragraphs
            for p in range(section.Paragraphs.Count):
                paragraph = section.Paragraphs[p]
                
                # Iterate over child objects
                for c in range(paragraph.ChildObjects.Count):
                    obj = paragraph.ChildObjects[c]
                    # Extract image data
                    if isinstance(obj, DocPicture):
                        try:
                            picture = obj
                            # Get image bytes
                            dataBytes = picture.ImageBytes
                            if dataBytes:
                                # Detect format from magic bytes
                                ext = _detect_image_format(dataBytes)
                                if not ext:
                                    ext = "png"  # Default to PNG
                                images.append((dataBytes, ext))
                        except Exception as e:
                            logger.warning(f"Failed to extract image from paragraph: {e}")
                            continue
            
            # Also check tables for images
            for t in range(section.Tables.Count):
                table = section.Tables[t]
                for r in range(table.Rows.Count):
                    row = table.Rows[r]
                    for cell_idx in range(row.Cells.Count):
                        cell = row.Cells[cell_idx]
                        for cp in range(cell.Paragraphs.Count):
                            cell_para = cell.Paragraphs[cp]
                            for co in range(cell_para.ChildObjects.Count):
                                obj = cell_para.ChildObjects[co]
                                if isinstance(obj, DocPicture):
                                    try:
                                        picture = obj
                                        dataBytes = picture.ImageBytes
                                        if dataBytes:
                                            ext = _detect_image_format(dataBytes)
                                            if not ext:
                                                ext = "png"
                                            images.append((dataBytes, ext))
                                    except Exception as e:
                                        logger.warning(f"Failed to extract image from table cell: {e}")
                                        continue
        
        # Close the document
        try:
            document.Close()
        except Exception:
            pass
                    
    except Exception as e:
        logger.error(f"Failed to load DOCX for image extraction: {e}")
        return []
    finally:
        # Clean up temp file
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except Exception:
                pass
    
    logger.debug(f"Extracted {len(images)} images from DOCX using Spire.Doc")
    return images


# // ---> unstructured_decoder.on_message > [extract_images_from_doc] > extract embedded images from DOC using Spire.Doc
def extract_images_from_doc(blob: bytes) -> List[Tuple[bytes, str]]:
    """
    Extract all embedded images from a legacy DOC document using Spire.Doc.
    
    Args:
        blob: Raw bytes of the DOC file
        
    Returns:
        List of tuples: (image_bytes, extension)
    """
    try:
        from spire.doc import Document, DocPicture
    except ImportError:
        logger.warning("spire-doc not installed; cannot extract images from DOC")
        return []
    
    images = []
    temp_file = None
    
    try:
        # Spire.Doc requires a file path, so write to temp file
        with tempfile.NamedTemporaryFile(suffix=".doc", delete=False) as f:
            f.write(blob)
            temp_file = f.name
        
        # Load document with Spire.Doc
        document = Document()
        document.LoadFromFile(temp_file)
        
        # Iterate over sections
        for s in range(document.Sections.Count):
            section = document.Sections[s]
            
            # Iterate over paragraphs
            for p in range(section.Paragraphs.Count):
                paragraph = section.Paragraphs[p]
                
                # Iterate over child objects
                for c in range(paragraph.ChildObjects.Count):
                    obj = paragraph.ChildObjects[c]
                    # Extract image data
                    if isinstance(obj, DocPicture):
                        try:
                            picture = obj
                            dataBytes = picture.ImageBytes
                            if dataBytes:
                                ext = _detect_image_format(dataBytes)
                                if not ext:
                                    ext = "png"
                                images.append((dataBytes, ext))
                        except Exception as e:
                            logger.warning(f"Failed to extract image from paragraph: {e}")
                            continue
            
            # Also check tables for images
            for t in range(section.Tables.Count):
                table = section.Tables[t]
                for r in range(table.Rows.Count):
                    row = table.Rows[r]
                    for cell_idx in range(row.Cells.Count):
                        cell = row.Cells[cell_idx]
                        for cp in range(cell.Paragraphs.Count):
                            cell_para = cell.Paragraphs[cp]
                            for co in range(cell_para.ChildObjects.Count):
                                obj = cell_para.ChildObjects[co]
                                if isinstance(obj, DocPicture):
                                    try:
                                        picture = obj
                                        dataBytes = picture.ImageBytes
                                        if dataBytes:
                                            ext = _detect_image_format(dataBytes)
                                            if not ext:
                                                ext = "png"
                                            images.append((dataBytes, ext))
                                    except Exception as e:
                                        logger.warning(f"Failed to extract image from table cell: {e}")
                                        continue
        
        # Close the document
        try:
            document.Close()
        except Exception:
            pass
                    
    except Exception as e:
        logger.warning(f"Failed to extract images from DOC: {e}")
        return []
    finally:
        # Clean up temp file
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except Exception:
                pass
    
    logger.debug(f"Extracted {len(images)} images from DOC using Spire.Doc")
    return images


# // ---> unstructured_decoder.on_message > [extract_images_from_office] > unified extraction interface
def extract_images_from_office(blob: bytes, content_type: str) -> List[Tuple[bytes, str]]:
    """
    Extract images from an Office document based on content type.
    
    Args:
        blob: Raw bytes of the document
        content_type: MIME type of the document
        
    Returns:
        List of tuples: (image_bytes, extension)
    """
    lowered = (content_type or "").strip().lower()
    
    if lowered == DOCX_CONTENT_TYPE:
        return extract_images_from_docx(blob)
    elif lowered == DOC_CONTENT_TYPE:
        return extract_images_from_doc(blob)
    else:
        return []


# // ---> extract_images_from_docx > [_detect_image_format] > detect image format from magic bytes
def _detect_image_format(data: bytes) -> Optional[str]:
    """Detect image format from magic bytes."""
    if len(data) < 8:
        return None
    
    # PNG signature
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        return "png"
    
    # JPEG signature
    if data[:2] == b'\xff\xd8':
        return "jpg"
    
    # GIF signature
    if data[:6] in (b'GIF87a', b'GIF89a'):
        return "gif"
    
    # BMP signature
    if data[:2] == b'BM':
        return "bmp"
    
    # TIFF signatures (little and big endian)
    if data[:4] in (b'II*\x00', b'MM\x00*'):
        return "tiff"
    
    # WebP signature
    if data[:4] == b'RIFF' and len(data) >= 12 and data[8:12] == b'WEBP':
        return "webp"
    
    # EMF signature
    if data[:4] == b'\x01\x00\x00\x00':
        return "emf"
    
    # WMF signature
    if data[:4] == b'\xd7\xcd\xc6\x9a':
        return "wmf"
    
    return None
