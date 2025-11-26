"""
DOCX decoder: Extracts text, tables, and images from DOCX documents using python-docx.
Images are sent to vLLM for description. All content is output as structured text segments
suitable for RAG processing.

Key features:
- Uses python-docx (free, open-source, no license required)
- Fallback to zipfile extraction when python-docx fails (handles corrupt files)
- Concurrent vLLM requests for faster image processing
- Image deduplication by MD5 hash to avoid redundant LLM calls
- Robust error handling that continues processing even if parts fail
- Extracts text, tables, headers/footers, and images

Reference: https://python-docx.readthedocs.io/
"""

import asyncio
import base64
import concurrent.futures
import hashlib
import logging
import os
import re
import zipfile
from io import BytesIO
from pathlib import Path
import requests
import json
from typing import Dict, List, Any, Optional, Set, Tuple

from ... schema import Document as TGDocument, TextDocument
from ... base import FlowProcessor, ConsumerSpec, ProducerSpec

# Module logger
logger = logging.getLogger(__name__)

# Try to import python-docx
try:
    from docx import Document as DocxDocument
    from docx.opc.exceptions import PackageNotFoundError
    from docx.shared import Inches, Pt
    DOCX_AVAILABLE = True
except ImportError as e:
    DOCX_AVAILABLE = False
    DocxDocument = None
    logger.warning(f"python-docx not available: {e}")

# Try to import PIL for image handling
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Enable DEBUG level for this module when VLLM_LOGGING_LEVEL=DEBUG is set
if os.environ.get("VLLM_LOGGING_LEVEL", "").upper() == "DEBUG":
    logger.setLevel(logging.DEBUG)

default_ident = "docx-decoder"

# Performance tuning constants
MAX_CONCURRENT_VLLM_REQUESTS = 4  # Max parallel vLLM requests
VLLM_TIMEOUT_SECONDS = 180  # Timeout for vLLM requests
MAX_IMAGE_SIZE_MB = 10  # Skip images larger than this


class Processor(FlowProcessor):

    def __init__(self, **params):
        id = params.get("id", default_ident)
        self.vllm_api_url = params.get(
            "vllm_api_url",
            "http://vllm:8000/v1/chat/completions"
        )
        self.vllm_model = params.get(
            "vllm_model",
            "Qwen/Qwen3-VL-4B-Instruct"
        )
        self.files_base_dir = params.get(
            "files_base_dir",
            "/root/files_to_parse"
        )

        super(Processor, self).__init__(
            **params | {"id": id}
        )

        self.register_specification(
            ConsumerSpec(
                name="input",
                schema=TGDocument,
                handler=self.on_message,
            )
        )

        self.register_specification(
            ProducerSpec(
                name="output",
                schema=TextDocument,
            )
        )

        logger.info("DOCX decoder initialized (python-docx + zipfile fallback, concurrent vLLM)")

    # // ---> Pulsar consumer(input) > [on_message] > extract text/tables/images, vLLM describe -> flow('output')
    async def on_message(self, msg, consumer, flow):
        logger.info("DOCX message received for extraction")

        v = msg.value()
        logger.info(f"Processing {v.metadata.id}...")

        blob = base64.b64decode(v.data)

        # Check if this is a DOCX file
        content_type = getattr(v, "content_type", None)
        if not self._is_docx(blob, content_type):
            logger.info(f"Skipping non-DOCX file: {v.metadata.id} (content_type: {content_type})")
            return

        # Prepare output directories
        doc_dir = os.path.join(self.files_base_dir, self._safe_id(v.metadata.id))
        os.makedirs(doc_dir, exist_ok=True)
        images_dir = os.path.join(doc_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        # Save DOCX to file
        temp_docx_path = os.path.join(doc_dir, "source.docx")
        try:
            with open(temp_docx_path, "wb") as f:
                f.write(blob)
            logger.debug(f"Saved DOCX to file: {temp_docx_path}")
        except Exception as e:
            logger.error(f"Failed to save DOCX to file: {e}")
            await self._process_with_fallback(blob, images_dir, v, flow)
            return

        # Track processed image hashes for deduplication
        processed_hashes: Set[str] = set()
        all_images: List[Dict[str, Any]] = []
        structured_data = None
        docx_success = False

        # Try python-docx extraction
        if DOCX_AVAILABLE:
            docx_success, structured_data, all_images = self._try_docx_extraction(
                temp_docx_path, blob, images_dir, processed_hashes
            )

        # Fallback to zipfile extraction if python-docx failed
        if not docx_success:
            logger.info("Using zipfile fallback extraction")
            fallback_images = self._extract_images_from_zip(blob, images_dir, processed_hashes)
            for ix, (image_path, original_name) in enumerate(fallback_images):
                all_images.append({
                    "path": image_path,
                    "context": f"Image {ix + 1} ({original_name})",
                    "paragraph_index": None
                })
            structured_data = {
                "document": {
                    "paragraphs": [],
                    "tables": [],
                    "properties": {},
                    "note": "Extracted via zipfile fallback"
                }
            }
            logger.info(f"Fallback: Found {len(fallback_images)} images")

        # Send document header first
        if structured_data:
            filename = getattr(v.metadata, 'id', 'document') + '.docx'
            header = self._format_document_header(structured_data, filename)
            if header.strip():
                logger.info(f"Sending document header to RAG ({len(header)} chars)")
                r = TextDocument(
                    metadata=v.metadata,
                    text=header.encode("utf-8"),
                )
                await flow("output").send(r)

        # Send text content in segments
        if structured_data and structured_data["document"]["paragraphs"]:
            text_content = self._format_document_text(structured_data)
            if text_content.strip():
                logger.info(f"Sending document text to RAG ({len(text_content)} chars)")
                r = TextDocument(
                    metadata=v.metadata,
                    text=text_content.encode("utf-8"),
                )
                await flow("output").send(r)

        # Send tables if any
        if structured_data and structured_data["document"]["tables"]:
            tables_text = self._format_tables(structured_data["document"]["tables"])
            if tables_text.strip():
                logger.info(f"Sending tables to RAG ({len(tables_text)} chars)")
                r = TextDocument(
                    metadata=v.metadata,
                    text=tables_text.encode("utf-8"),
                )
                await flow("output").send(r)

        # Process images with concurrent vLLM requests
        if all_images:
            logger.info(f"Processing {len(all_images)} unique images with concurrent vLLM requests")
            image_descriptions = await self._describe_images_concurrent(all_images)

            # Send image descriptions
            for idx, (img_info, description) in enumerate(zip(all_images, image_descriptions)):
                if description and description.strip():
                    image_text = f"{img_info['context']}\nDescription:\n{description}\n"
                    logger.info(f"Sending image {idx+1}/{len(all_images)} description to RAG")
                    r = TextDocument(
                        metadata=v.metadata,
                        text=image_text.encode("utf-8"),
                    )
                    await flow("output").send(r)

        logger.info(f"DOCX extraction complete: {len(all_images)} images processed")

    # // ---> on_message > [_try_docx_extraction] > extract using python-docx with fallback
    def _try_docx_extraction(
        self, docx_path: str, blob: bytes, images_dir: str, processed_hashes: Set[str]
    ) -> Tuple[bool, Optional[Dict], List[Dict]]:
        """
        Try to extract content using python-docx.
        Falls back to zipfile if python-docx fails completely.
        """
        all_images = []
        structured_data = None

        try:
            # Try loading from file first
            logger.debug("Trying python-docx load from file")
            doc = DocxDocument(docx_path)
            logger.info(f"python-docx loaded successfully: {len(doc.paragraphs)} paragraphs")

            # Extract all content
            structured_data = self._extract_structured_docx(doc, images_dir, processed_hashes)
            logger.info(f"Extracted {len(structured_data['document']['paragraphs'])} paragraphs, "
                       f"{len(structured_data['document']['tables'])} tables")

            # Collect images
            for img_info in structured_data["document"].get("images", []):
                all_images.append({
                    "path": img_info["path"],
                    "context": f"Document Image - {img_info['filename']}",
                    "paragraph_index": img_info.get("paragraph_index")
                })

            return True, structured_data, all_images

        except Exception as e:
            logger.warning(f"python-docx file load failed: {e}")

            # Try loading from bytes (in-memory)
            try:
                logger.debug("Trying python-docx load from bytes")
                doc = DocxDocument(BytesIO(blob))
                logger.info(f"python-docx loaded from bytes: {len(doc.paragraphs)} paragraphs")

                structured_data = self._extract_structured_docx(doc, images_dir, processed_hashes)

                for img_info in structured_data["document"].get("images", []):
                    all_images.append({
                        "path": img_info["path"],
                        "context": f"Document Image - {img_info['filename']}",
                        "paragraph_index": img_info.get("paragraph_index")
                    })

                return True, structured_data, all_images

            except Exception as e2:
                logger.warning(f"python-docx bytes load also failed: {e2}")
                return False, None, []

    # // ---> on_message > [_process_with_fallback] > handle fallback when file save fails
    async def _process_with_fallback(self, blob: bytes, images_dir: str, v, flow):
        """Fallback processing when file operations fail."""
        processed_hashes: Set[str] = set()
        images = self._extract_images_from_zip(blob, images_dir, processed_hashes)

        if images:
            logger.info(f"Fallback: Extracted {len(images)} images")
            all_images = [
                {"path": p, "context": f"Image {i+1} ({name})", "paragraph_index": None}
                for i, (p, name) in enumerate(images)
            ]
            descriptions = await self._describe_images_concurrent(all_images)

            for img_info, desc in zip(all_images, descriptions):
                if desc:
                    r = TextDocument(
                        metadata=v.metadata,
                        text=f"{img_info['context']}\nDescription:\n{desc}\n".encode("utf-8")
                    )
                    await flow("output").send(r)
        else:
            logger.warning("Fallback extraction found no images")

    @staticmethod
    def add_args(parser):
        FlowProcessor.add_args(parser)

    # // ---> [_safe_id] > sanitize directory name
    def _safe_id(self, value: str) -> str:
        if value is None:
            return "unknown"
        return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))

    # // ---> [_is_docx] > check if blob is a DOCX file
    def _is_docx(self, blob: bytes, content_type: Optional[str] = None) -> bool:
        """Check if blob is a DOCX file by content_type or magic bytes."""
        if content_type:
            ct_lower = content_type.strip().lower()
            if ct_lower in (
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/msword"
            ):
                return True

        # DOCX files are ZIP archives starting with PK
        if len(blob) >= 4 and blob[:4] == b'PK\x03\x04':
            # Further verify it's a DOCX by checking for word/ directory
            try:
                with zipfile.ZipFile(BytesIO(blob), 'r') as zf:
                    names = zf.namelist()
                    if any(name.startswith('word/') for name in names):
                        return True
            except Exception:
                pass

        return False

    # // ---> [_compute_image_hash] > MD5 hash for deduplication
    def _compute_image_hash(self, image_data: bytes) -> str:
        return hashlib.md5(image_data).hexdigest()

    # // ---> [_extract_images_from_zip] > fallback extraction via zipfile
    def _extract_images_from_zip(
        self, blob: bytes, images_dir: str, processed_hashes: Set[str]
    ) -> List[Tuple[str, str]]:
        """
        Extract images directly from DOCX ZIP archive.
        Returns list of (saved_path, original_name) tuples.
        Works even when python-docx can't parse the file.
        """
        saved = []
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'}
        # EMF/WMF are Windows metafiles - we'll try to handle them
        metafile_extensions = {'.emf', '.wmf'}

        try:
            with zipfile.ZipFile(BytesIO(blob), 'r') as zf:
                for name in zf.namelist():
                    # DOCX stores media in word/media/
                    if 'media/' not in name.lower():
                        continue

                    ext = Path(name).suffix.lower()
                    original_name = Path(name).name

                    if ext in image_extensions or ext in metafile_extensions:
                        try:
                            image_data = zf.read(name)

                            # Skip very large images
                            if len(image_data) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
                                logger.debug(f"Skipping large image: {name}")
                                continue

                            img_hash = self._compute_image_hash(image_data)
                            if img_hash in processed_hashes:
                                continue
                            processed_hashes.add(img_hash)

                            # Try to convert EMF/WMF if PIL is available
                            if ext in metafile_extensions and PIL_AVAILABLE:
                                converted = self._try_convert_metafile(image_data)
                                if converted:
                                    image_data = converted
                                    ext = '.png'
                                else:
                                    # Skip unconvertible metafiles
                                    logger.debug(f"Skipping unconvertible metafile: {name}")
                                    continue

                            filename = f"zip_{len(saved)}{ext}"
                            path = os.path.join(images_dir, filename)
                            with open(path, 'wb') as f:
                                f.write(image_data)
                            saved.append((path, original_name))
                            logger.debug(f"Extracted from ZIP: {original_name}")

                        except Exception as e:
                            logger.debug(f"Failed to extract {name}: {e}")

        except zipfile.BadZipFile:
            logger.warning("Not a valid ZIP archive")
        except Exception as e:
            logger.warning(f"ZIP extraction failed: {e}")

        return saved

    # // ---> [_try_convert_metafile] > convert EMF/WMF to PNG
    def _try_convert_metafile(self, data: bytes) -> Optional[bytes]:
        """Try to convert EMF/WMF metafile to PNG using PIL."""
        if not PIL_AVAILABLE:
            return None
        try:
            img = Image.open(BytesIO(data))
            output = BytesIO()
            img.save(output, format='PNG')
            return output.getvalue()
        except Exception:
            return None

    # // ---> [_extract_structured_docx] > parse DOCX with python-docx
    def _extract_structured_docx(
        self, doc, images_dir: str, processed_hashes: Set[str]
    ) -> Dict[str, Any]:
        """Extract all content from DOCX using python-docx."""
        structured_data = {
            "document": {
                "paragraphs": [],
                "tables": [],
                "images": [],
                "properties": {}
            }
        }

        # Extract core properties
        try:
            props = doc.core_properties
            structured_data["document"]["properties"] = {
                "title": props.title or "",
                "author": props.author or "",
                "subject": props.subject or "",
                "keywords": props.keywords or "",
                "comments": props.comments or "",
            }
        except Exception as e:
            logger.debug(f"Failed to extract core properties: {e}")

        image_counter = 0

        # Process paragraphs
        for para_idx, paragraph in enumerate(doc.paragraphs):
            try:
                para_data = self._extract_paragraph_data(
                    paragraph, para_idx, images_dir, processed_hashes, image_counter
                )
                if para_data["text"] or para_data.get("images"):
                    structured_data["document"]["paragraphs"].append(para_data)

                # Collect images from paragraph
                for img_info in para_data.get("images", []):
                    img_info["paragraph_index"] = para_idx
                    structured_data["document"]["images"].append(img_info)
                    image_counter += 1

            except Exception as e:
                logger.debug(f"Failed to process paragraph {para_idx}: {e}")

        # Process tables
        for table_idx, table in enumerate(doc.tables):
            try:
                table_data = self._extract_table_data(table, table_idx)
                if table_data:
                    structured_data["document"]["tables"].append(table_data)

                # Extract images from table cells
                for row in table.rows:
                    for cell in row.cells:
                        for para in cell.paragraphs:
                            try:
                                para_images = self._extract_paragraph_images(
                                    para, images_dir, processed_hashes, image_counter
                                )
                                for img_info in para_images:
                                    img_info["table_index"] = table_idx
                                    structured_data["document"]["images"].append(img_info)
                                    image_counter += 1
                            except Exception:
                                pass

            except Exception as e:
                logger.debug(f"Failed to process table {table_idx}: {e}")

        # Also extract images from document's inline shapes (relationship-based extraction)
        try:
            doc_images = self._extract_document_images(doc, images_dir, processed_hashes, image_counter)
            for img_info in doc_images:
                structured_data["document"]["images"].append(img_info)
        except Exception as e:
            logger.debug(f"Failed to extract document images: {e}")

        return structured_data

    # // ---> [_extract_paragraph_data] > extract content from a single paragraph
    def _extract_paragraph_data(
        self, paragraph, para_idx: int, images_dir: str,
        processed_hashes: Set[str], image_counter: int
    ) -> Dict[str, Any]:
        """Extract text and images from a paragraph."""
        para_data = {
            "paragraph_index": para_idx,
            "text": "",
            "style": None,
            "images": []
        }

        # Get paragraph style
        try:
            if paragraph.style:
                para_data["style"] = paragraph.style.name
        except Exception:
            pass

        # Get text content
        try:
            text = paragraph.text.strip()
            if text:
                para_data["text"] = text
        except Exception:
            pass

        # Extract inline images from runs
        para_data["images"] = self._extract_paragraph_images(
            paragraph, images_dir, processed_hashes, image_counter
        )

        return para_data

    # // ---> [_extract_paragraph_images] > extract images from paragraph runs
    def _extract_paragraph_images(
        self, paragraph, images_dir: str, processed_hashes: Set[str], counter: int
    ) -> List[Dict]:
        """Extract images from paragraph's inline shapes."""
        images = []

        try:
            # Access inline shapes through the paragraph's XML
            for run in paragraph.runs:
                try:
                    # Check for inline shapes in the run's XML
                    inline_shapes = run._element.findall('.//' + '{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}inline')
                    for inline in inline_shapes:
                        try:
                            # Get the blip element which contains the image reference
                            blip = inline.find('.//' + '{http://schemas.openxmlformats.org/drawingml/2006/main}blip')
                            if blip is not None:
                                # Get the relationship ID
                                embed_attr = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed'
                                rId = blip.get(embed_attr)
                                if rId:
                                    # Get the image part from the document's relationships
                                    try:
                                        part = paragraph.part.related_parts.get(rId)
                                        if part and hasattr(part, 'blob'):
                                            image_bytes = part.blob

                                            # Check size
                                            if len(image_bytes) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
                                                continue

                                            # Check for duplicates
                                            img_hash = self._compute_image_hash(image_bytes)
                                            if img_hash in processed_hashes:
                                                continue
                                            processed_hashes.add(img_hash)

                                            # Determine extension
                                            content_type = getattr(part, 'content_type', 'image/png')
                                            ext = self._content_type_to_ext(content_type)

                                            # Handle metafiles
                                            if ext in ['.emf', '.wmf'] and PIL_AVAILABLE:
                                                converted = self._try_convert_metafile(image_bytes)
                                                if converted:
                                                    image_bytes = converted
                                                    ext = '.png'
                                                else:
                                                    continue

                                            filename = f"para_{counter + len(images)}{ext}"
                                            path = os.path.join(images_dir, filename)

                                            with open(path, 'wb') as f:
                                                f.write(image_bytes)

                                            images.append({
                                                "filename": filename,
                                                "path": path
                                            })
                                            logger.debug(f"Extracted inline image: {filename}")
                                    except Exception as e:
                                        logger.debug(f"Failed to get image part: {e}")
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"Failed to extract paragraph images: {e}")

        return images

    # // ---> [_extract_document_images] > extract all images from document relationships
    def _extract_document_images(
        self, doc, images_dir: str, processed_hashes: Set[str], counter: int
    ) -> List[Dict]:
        """Extract images using document part relationships as fallback."""
        images = []

        try:
            # Access all image parts from document relationships
            for rel in doc.part.rels.values():
                try:
                    if "image" in rel.reltype:
                        part = rel.target_part
                        if hasattr(part, 'blob'):
                            image_bytes = part.blob

                            # Check size
                            if len(image_bytes) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
                                continue

                            # Check for duplicates
                            img_hash = self._compute_image_hash(image_bytes)
                            if img_hash in processed_hashes:
                                continue
                            processed_hashes.add(img_hash)

                            # Determine extension
                            content_type = getattr(part, 'content_type', 'image/png')
                            ext = self._content_type_to_ext(content_type)

                            # Handle metafiles
                            if ext in ['.emf', '.wmf'] and PIL_AVAILABLE:
                                converted = self._try_convert_metafile(image_bytes)
                                if converted:
                                    image_bytes = converted
                                    ext = '.png'
                                else:
                                    continue

                            filename = f"doc_{counter + len(images)}{ext}"
                            path = os.path.join(images_dir, filename)

                            with open(path, 'wb') as f:
                                f.write(image_bytes)

                            images.append({
                                "filename": filename,
                                "path": path
                            })
                            logger.debug(f"Extracted document image: {filename}")

                except Exception as e:
                    logger.debug(f"Failed to extract image from relationship: {e}")

        except Exception as e:
            logger.debug(f"Failed to iterate document relationships: {e}")

        return images

    # // ---> [_content_type_to_ext] > map content type to file extension
    def _content_type_to_ext(self, content_type: str) -> str:
        """Map content type to file extension."""
        ext_map = {
            'image/png': '.png',
            'image/jpeg': '.jpg',
            'image/gif': '.gif',
            'image/bmp': '.bmp',
            'image/tiff': '.tiff',
            'image/webp': '.webp',
            'image/x-emf': '.emf',
            'image/x-wmf': '.wmf',
        }
        return ext_map.get(content_type, '.png')

    # // ---> [_extract_table_data] > extract table cell data
    def _extract_table_data(self, table, table_idx: int) -> Optional[Dict]:
        """Extract table data as structured dict."""
        try:
            rows_data = []
            for row in table.rows:
                row_data = []
                for cell in row.cells:
                    try:
                        # Get text from all paragraphs in cell
                        cell_texts = []
                        for para in cell.paragraphs:
                            para_text = para.text.strip()
                            if para_text:
                                cell_texts.append(para_text)
                        cell_text = " ".join(cell_texts)
                        row_data.append(cell_text)
                    except Exception:
                        row_data.append("")
                rows_data.append(row_data)

            if rows_data:
                return {
                    "table_index": table_idx,
                    "rows": len(rows_data),
                    "columns": len(rows_data[0]) if rows_data else 0,
                    "data": rows_data
                }
            return None

        except Exception as e:
            logger.debug(f"Table extraction failed: {e}")
            return None

    # // ---> [_format_document_header] > format document header
    def _format_document_header(self, structured_data: Dict, filename: str) -> str:
        """Format document header with metadata."""
        parts = []
        doc = structured_data.get("document", {})

        parts.append(f"# Word Document: {filename}")
        parts.append("")
        parts.append(f"Total Paragraphs: {len(doc.get('paragraphs', []))}")
        parts.append(f"Total Tables: {len(doc.get('tables', []))}")
        parts.append(f"Total Images: {len(doc.get('images', []))}")

        # Include core properties if available
        props = doc.get("properties", {})
        if props.get("title"):
            parts.append(f"Title: {props['title']}")
        if props.get("author"):
            parts.append(f"Author: {props['author']}")
        if props.get("subject"):
            parts.append(f"Subject: {props['subject']}")

        parts.append("")
        return "\n".join(parts)

    # // ---> [_format_document_text] > format document paragraphs as text
    def _format_document_text(self, structured_data: Dict) -> str:
        """Format document paragraphs as structured text for RAG."""
        parts = []
        doc = structured_data.get("document", {})

        parts.append("## Document Content:")
        parts.append("")

        current_style = None
        for para in doc.get("paragraphs", []):
            style = para.get("style")
            text = para.get("text", "")

            if not text:
                continue

            # Add style header if changed
            if style and style != current_style:
                if style.startswith("Heading"):
                    # Format headings appropriately
                    level = style.replace("Heading", "").strip()
                    try:
                        level_num = int(level)
                        parts.append("")
                        parts.append("#" * (level_num + 1) + f" {text}")
                        parts.append("")
                        current_style = style
                        continue
                    except ValueError:
                        pass
                current_style = style

            parts.append(text)
            parts.append("")

        return "\n".join(parts)

    # // ---> [_format_tables] > format tables as text
    def _format_tables(self, tables: List[Dict]) -> str:
        """Format tables as structured text."""
        parts = []

        parts.append("## Tables:")
        parts.append("")

        for table in tables:
            parts.append(f"Table {table['table_index'] + 1} ({table['rows']} rows × {table['columns']} columns):")
            parts.append("")
            for row in table.get("data", []):
                parts.append(" | ".join(str(c) if c else "" for c in row))
            parts.append("")

        return "\n".join(parts)

    # // ---> [_image_to_base64] > convert image to base64 data URL
    def _image_to_base64(self, image_path: str) -> Optional[str]:
        """Convert image file to base64 data URL for vLLM."""
        try:
            with open(image_path, "rb") as f:
                data = f.read()

            if len(data) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
                logger.warning(f"Image too large for vLLM: {image_path}")
                return None

            b64 = base64.b64encode(data).decode("utf-8")
            ext = Path(image_path).suffix.lower()
            mime_types = {
                ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp"
            }
            mime = mime_types.get(ext, "image/png")
            return f"data:{mime};base64,{b64}"
        except Exception as e:
            logger.error(f"Failed to encode image: {e}")
            return None

    # // ---> [_describe_images_concurrent] > concurrent vLLM requests
    async def _describe_images_concurrent(self, images: List[Dict]) -> List[str]:
        """
        Describe images using concurrent vLLM requests.
        Processes up to MAX_CONCURRENT_VLLM_REQUESTS in parallel.
        """
        if not images:
            return []

        # Filter valid images
        valid_images = [img for img in images if os.path.exists(img["path"])]
        if not valid_images:
            return [""] * len(images)

        logger.info(f"Processing {len(valid_images)} images concurrently")

        loop = asyncio.get_event_loop()

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_VLLM_REQUESTS) as executor:
            futures = [
                loop.run_in_executor(executor, self._describe_single_image, img["path"])
                for img in valid_images
            ]
            results = await asyncio.gather(*futures, return_exceptions=True)

        # Process results
        descriptions = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Image description failed: {result}")
                descriptions.append("")
            else:
                descriptions.append(result or "")

        return descriptions

    # // ---> [_describe_single_image] > single vLLM request
    def _describe_single_image(self, image_path: str) -> str:
        """Describe a single image using vLLM."""
        try:
            image_url = self._image_to_base64(image_path)
            if not image_url:
                return "Image could not be processed."

            payload = {
                "model": self.vllm_model,
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Describe this image from a document in detail. "
                                "Focus on: 1) Any text or labels visible, 2) Charts/graphs and their data, "
                                "3) Diagrams and their relationships, 4) Key visual elements. "
                                "Be concise but comprehensive."
                            )
                        },
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }]
            }

            # Log the outgoing request (truncate base64 for readability)
            try:
                payload_log = json.loads(json.dumps(payload))
                try:
                    url_val = payload_log["messages"][0]["content"][1]["image_url"]["url"]
                    if url_val.startswith("data:"):
                        payload_log["messages"][0]["content"][1]["image_url"]["url"] = (
                            url_val[:60] + f"...[{len(url_val)} chars total]"
                        )
                except (KeyError, IndexError):
                    pass
                logger.debug(
                    "vLLM request: POST %s payload=%s",
                    self.vllm_api_url,
                    json.dumps(payload_log, ensure_ascii=False),
                )
            except Exception as log_err:
                logger.warning(f"Failed to serialize vLLM payload for logging: {log_err}")

            resp = requests.post(self.vllm_api_url, json=payload, timeout=VLLM_TIMEOUT_SECONDS)

            if resp.status_code != 200:
                logger.error(f"vLLM HTTP {resp.status_code}: {resp.text[:500]}")
                return f"Description unavailable (HTTP {resp.status_code})"

            data = resp.json()
            choice = (data.get("choices") or [{}])[0]
            content = (choice.get("message") or {}).get("content", "")

            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                return "\n".join(c.get("text", "") for c in content if c.get("type") == "text")
            return ""

        except requests.exceptions.Timeout:
            logger.error(f"vLLM request timed out for {image_path}")
            return "Description unavailable (timeout)"
        except requests.exceptions.ConnectionError as e:
            logger.error(f"vLLM connection error for {image_path}: {e}")
            return "Description unavailable (connection error)"
        except Exception as e:
            logger.error(f"vLLM error: {e}")
            return "Description unavailable"


def run():
    Processor.launch(default_ident, __doc__)
