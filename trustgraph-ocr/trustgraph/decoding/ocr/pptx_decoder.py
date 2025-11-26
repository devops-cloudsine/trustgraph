"""
PPTX decoder: Extracts text, tables, and images from PPTX presentations using Spire.Presentation.
Images are sent to vLLM for description. All content is output as structured text segments
suitable for RAG processing.
"""

import base64
import logging
import os
import re
from pathlib import Path
import requests
import json
from typing import Dict, List, Any, Optional

from spire.presentation.common import *
from spire.presentation import Presentation, IShape

from ... schema import Document as TGDocument, TextDocument
from ... base import FlowProcessor, ConsumerSpec, ProducerSpec

# Module logger
logger = logging.getLogger(__name__)

# Enable DEBUG level for this module when VLLM_LOGGING_LEVEL=DEBUG is set
if os.environ.get("VLLM_LOGGING_LEVEL", "").upper() == "DEBUG":
    logger.setLevel(logging.DEBUG)

default_ident = "pptx-decoder"


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
            **params | {
                "id": id,
            }
        )

        self.register_specification(
            ConsumerSpec(
                name = "input",
                schema = TGDocument,
                handler = self.on_message,
            )
        )

        self.register_specification(
            ProducerSpec(
                name = "output",
                schema = TextDocument,
            )
        )

        logger.info("PPTX extraction + vLLM description processor initialized (using Spire.Presentation)")

    # // ---> Pulsar consumer(input) > [on_message] > extract slides/text/images, vLLM describe -> flow('output')
    async def on_message(self, msg, consumer, flow):

        logger.info("PPTX message received for extraction")

        v = msg.value()

        logger.info(f"Processing {v.metadata.id}...")

        blob = base64.b64decode(v.data)

        # Check if this is a PPTX file (content_type or magic bytes)
        content_type = getattr(v, "content_type", None)
        if not self._is_pptx(blob, content_type):
            logger.info(f"Skipping non-PPTX file: {v.metadata.id} (content_type: {content_type})")
            return

        # Prepare output directories for this document
        doc_dir = os.path.join(self.files_base_dir, self._safe_id(v.metadata.id))
        os.makedirs(doc_dir, exist_ok=True)
        images_dir = os.path.join(doc_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        # Write blob to a file for Spire.Presentation to load
        temp_pptx_path = os.path.join(doc_dir, "source.pptx")
        try:
            with open(temp_pptx_path, "wb") as f:
                f.write(blob)
            logger.debug(f"Saved PPTX to file: {temp_pptx_path}")
        except Exception as e:
            logger.error(f"Failed to save PPTX to file: {e}")
            return

        # Load PPTX using Spire.Presentation
        try:
            ppt = Presentation()
            ppt.LoadFromFile(temp_pptx_path)
        except Exception as e:
            logger.error(f"Failed to load PPTX document with Spire.Presentation: {e}")
            return

        # Extract structured data from presentation
        structured_data = self._extract_structured_pptx(ppt, images_dir)
        
        logger.info(f"Extracted {structured_data['presentation']['total_slides']} slides from PPTX")

        # Process each slide and send structured output
        for slide_data in structured_data["presentation"]["slides"]:
            slide_number = slide_data["slide_number"]
            
            # Build structured text output for this slide
            slide_text_parts = []
            
            # Header with slide info
            slide_text_parts.append(f"{'='*60}")
            slide_text_parts.append(f"SLIDE {slide_number}")
            if slide_data.get("layout"):
                slide_text_parts.append(f"Layout: {slide_data['layout']}")
            slide_text_parts.append(f"{'='*60}\n")
            
            # Add text content
            if slide_data["text_content"]:
                slide_text_parts.append("## Text Content:")
                for text_item in slide_data["text_content"]:
                    slide_text_parts.append(f"\n{text_item['text']}")
                slide_text_parts.append("")
            
            # Add tables as structured text
            if slide_data["tables"]:
                slide_text_parts.append("## Tables:")
                for table_idx, table in enumerate(slide_data["tables"]):
                    slide_text_parts.append(f"\nTable {table_idx + 1} ({table['rows']} rows × {table['columns']} columns):")
                    for row in table["data"]:
                        slide_text_parts.append(" | ".join(str(cell) if cell else "" for cell in row))
                slide_text_parts.append("")
            
            # Add shape summary
            if slide_data["shapes"]:
                slide_text_parts.append(f"## Shape Summary: {len(slide_data['shapes'])} shapes")
                for shape in slide_data["shapes"]:
                    shape_info = f"  - Shape {shape['shape_index']}: {shape['shape_type']}"
                    if shape.get("name"):
                        shape_info += f" (Name: {shape['name']})"
                    if shape.get("text"):
                        preview = shape["text"][:50] + "..." if len(shape["text"]) > 50 else shape["text"]
                        shape_info += f" - Text: {preview}"
                    slide_text_parts.append(shape_info)
                slide_text_parts.append("")
            
            # Send slide text content
            slide_text = "\n".join(slide_text_parts)
            if slide_text.strip():
                r = TextDocument(
                    metadata=v.metadata,
                    text=slide_text.encode("utf-8"),
                )
                await flow("output").send(r)
            
            # Process and describe images for this slide
            for img_info in slide_data["images"]:
                image_path = img_info["path"]
                
                if os.path.exists(image_path):
                    # Describe image via vLLM
                    description = self._describe_image_with_vllm(image_path)
                    image_text = (
                        f"Slide {slide_number} - Image: {img_info['filename']}\n"
                        f"Shape Index: {img_info['shape_index']}\n"
                        f"Description:\n{description}\n"
                    )
                    
                    r = TextDocument(
                        metadata=v.metadata,
                        text=image_text.encode("utf-8"),
                    )
                    await flow("output").send(r)
        
        # Also process global images (fallback extraction)
        global_images = self._extract_global_images(ppt, images_dir)
        for ix, image_path in enumerate(global_images):
            if os.path.exists(image_path):
                description = self._describe_image_with_vllm(image_path)
                image_text = (
                    f"Global Image {ix + 1}\n"
                    f"Description:\n{description}\n"
                )
                
                r = TextDocument(
                    metadata=v.metadata,
                    text=image_text.encode("utf-8"),
                )
                await flow("output").send(r)

        # Dispose presentation
        try:
            ppt.Dispose()
        except Exception as e:
            logger.warning(f"Failed to dispose Spire.Presentation: {e}")

        logger.info("PPTX extraction and image descriptions complete")

    @staticmethod
    def add_args(parser):
        FlowProcessor.add_args(parser)

    # // ---> on_message > [_safe_id] > sanitize directory name
    def _safe_id(self, value: str) -> str:
        if value is None:
            return "unknown"
        return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))

    # // ---> on_message > [_is_pptx] > check if blob is a PPTX file
    def _is_pptx(self, blob: bytes, content_type: Optional[str] = None) -> bool:
        """
        Check if the blob is a PPTX file by examining content_type and/or magic bytes.
        PPTX files are ZIP-based Office Open XML files (magic bytes: PK).
        """
        # Check content_type first if available
        if content_type:
            ct_lower = content_type.strip().lower()
            if ct_lower == "application/vnd.openxmlformats-officedocument.presentationml.presentation":
                return True
            # Also handle legacy PPT if we want to try loading it
            if ct_lower == "application/vnd.ms-powerpoint":
                # Legacy PPT - might work with Spire, let's try
                return True

        # Fall back to checking magic bytes
        # PPTX/DOCX/XLSX are all ZIP files starting with "PK" (0x50 0x4B)
        if len(blob) < 4:
            return False

        # ZIP magic bytes check
        if blob[:2] != b'PK':
            return False

        # Additional validation: check for Office Open XML signature
        # ZIP files have "PK\x03\x04" as local file header signature
        if blob[:4] == b'PK\x03\x04':
            # This is a valid ZIP file - could be PPTX, DOCX, or XLSX
            # For now, accept any ZIP file and let Spire.Presentation validate
            # A more thorough check would unzip and look for [Content_Types].xml
            return True

        return False

    # // ---> on_message > [_extract_structured_pptx] > parse PPTX and build structured data dict
    def _extract_structured_pptx(self, ppt: Presentation, images_dir: str) -> Dict[str, Any]:
        """
        Extract structured data from PPTX file including slides, text, images, and shapes.
        Returns a dictionary with hierarchical structure suitable for LLM/RAG processing.
        """
        structured_data = {
            "presentation": {
                "total_slides": ppt.Slides.Count,
                "slides": []
            }
        }

        # Global image counter for unique naming
        global_image_counter = 0

        # Iterate through all slides
        for slide_idx in range(ppt.Slides.Count):
            slide = ppt.Slides[slide_idx]

            slide_data = {
                "slide_number": slide_idx + 1,
                "slide_id": slide.SlideID if hasattr(slide, "SlideID") else None,
                "layout": slide.Layout.Name if hasattr(slide, "Layout") and hasattr(slide.Layout, "Name") else None,
                "shapes": [],
                "text_content": [],
                "images": [],
                "tables": []
            }

            # Iterate through shapes on the slide
            for shape_idx in range(slide.Shapes.Count):
                shape = slide.Shapes[shape_idx]

                shape_data = {
                    "shape_index": shape_idx,
                    "shape_type": str(shape.ShapeType) if hasattr(shape, "ShapeType") else "Unknown",
                    "name": shape.Name if hasattr(shape, "Name") else None,
                    "left": float(shape.Left) if hasattr(shape, "Left") else None,
                    "top": float(shape.Top) if hasattr(shape, "Top") else None,
                    "width": float(shape.Width) if hasattr(shape, "Width") else None,
                    "height": float(shape.Height) if hasattr(shape, "Height") else None,
                    "text": None,
                    "image_reference": None,
                    "table_data": None
                }

                # Extract text from shapes with TextFrame
                if hasattr(shape, "TextFrame") and shape.TextFrame is not None:
                    try:
                        text_content = shape.TextFrame.Text
                        if text_content and text_content.strip():
                            shape_data["text"] = text_content.strip()
                            slide_data["text_content"].append({
                                "shape_index": shape_idx,
                                "text": text_content.strip()
                            })
                    except Exception as e:
                        logger.debug(f"Failed to extract text from shape {shape_idx}: {e}")

                # Extract images - check for PictureFill property
                global_image_counter = self._try_extract_shape_image(
                    shape, shape_data, slide_data, images_dir,
                    slide_idx, global_image_counter
                )

                # Extract tables - check for Table property
                self._try_extract_table(shape, shape_data, slide_data, shape_idx)

                slide_data["shapes"].append(shape_data)

            structured_data["presentation"]["slides"].append(slide_data)

        return structured_data

    # // ---> _extract_structured_pptx > [_try_extract_shape_image] > extract image from shape if present
    def _try_extract_shape_image(
        self, shape, shape_data: Dict, slide_data: Dict,
        images_dir: str, slide_idx: int, global_image_counter: int
    ) -> int:
        """
        Try to extract an image from a shape using multiple methods.
        Returns updated global_image_counter.
        """
        try:
            if hasattr(shape, "PictureFill") and shape.PictureFill is not None:
                if hasattr(shape.PictureFill, "Picture") and shape.PictureFill.Picture is not None:
                    image = shape.PictureFill.Picture.EmbedImage
                    if image is not None:
                        image_filename = f"slide_{slide_idx + 1}_image_{global_image_counter}.png"
                        image_path = os.path.join(images_dir, image_filename)
                        image.Image.Save(image_path)

                        shape_data["image_reference"] = image_filename
                        slide_data["images"].append({
                            "shape_index": shape_data["shape_index"],
                            "filename": image_filename,
                            "path": image_path
                        })
                        global_image_counter += 1
                        logger.debug(f"Extracted image via PictureFill: {image_filename}")
                        return global_image_counter
        except Exception as e:
            logger.debug(f"PictureFill extraction failed: {e}")

        # Try alternative method: check Fill.Picture
        try:
            if hasattr(shape, "Fill") and shape.Fill is not None:
                if hasattr(shape.Fill, "Picture") and shape.Fill.Picture is not None:
                    if hasattr(shape.Fill.Picture, "EmbedImage") and shape.Fill.Picture.EmbedImage is not None:
                        image = shape.Fill.Picture.EmbedImage
                        image_filename = f"slide_{slide_idx + 1}_image_{global_image_counter}.png"
                        image_path = os.path.join(images_dir, image_filename)
                        image.Image.Save(image_path)

                        shape_data["image_reference"] = image_filename
                        slide_data["images"].append({
                            "shape_index": shape_data["shape_index"],
                            "filename": image_filename,
                            "path": image_path
                        })
                        global_image_counter += 1
                        logger.debug(f"Extracted image via Fill.Picture: {image_filename}")
        except Exception as e:
            logger.debug(f"Fill.Picture extraction failed: {e}")

        return global_image_counter

    # // ---> _extract_structured_pptx > [_try_extract_table] > extract table data from shape if present
    def _try_extract_table(self, shape, shape_data: Dict, slide_data: Dict, shape_idx: int) -> None:
        """
        Try to extract table data from a shape.
        """
        try:
            if hasattr(shape, "Table") and shape.Table is not None:
                table = shape.Table
                table_data = []

                for row_idx in range(table.Rows.Count):
                    row = table.Rows[row_idx]
                    row_data = []

                    for col_idx in range(row.Count):
                        cell = row[col_idx]
                        cell_text = ""
                        if hasattr(cell, "TextFrame") and cell.TextFrame is not None:
                            try:
                                cell_text = cell.TextFrame.Text.strip() if cell.TextFrame.Text else ""
                            except Exception:
                                pass
                        row_data.append(cell_text)

                    table_data.append(row_data)

                shape_data["table_data"] = table_data
                slide_data["tables"].append({
                    "shape_index": shape_idx,
                    "rows": len(table_data),
                    "columns": len(table_data[0]) if table_data else 0,
                    "data": table_data
                })
                logger.debug(f"Extracted table with {len(table_data)} rows")
        except Exception as e:
            logger.debug(f"Table extraction failed: {e}")

    # // ---> on_message > [_extract_global_images] > fallback extraction of all presentation images
    def _extract_global_images(self, ppt: Presentation, images_dir: str) -> List[str]:
        """
        Extract all images from the presentation using the global Images collection.
        This is a fallback method to catch any images not extracted from individual shapes.
        Returns list of saved image paths.
        """
        saved_paths = []
        try:
            for i, image in enumerate(ppt.Images):
                image_filename = f"global_image_{i}.png"
                image_path = os.path.join(images_dir, image_filename)
                image.Image.Save(image_path)
                saved_paths.append(image_path)
                logger.debug(f"Extracted global image: {image_filename}")
        except Exception as e:
            logger.debug(f"Global image extraction failed: {e}")
        return saved_paths

    # // ---> on_message > [_image_to_base64_data_url] > reads image file, returns data:image/... URL
    def _image_to_base64_data_url(self, image_path: str) -> str:
        """
        Convert an image file to a base64 data URL.
        This is required because vLLM runs in a separate container and cannot
        access file:// paths from this container's filesystem.
        """
        with open(image_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode("utf-8")

        # Detect MIME type from extension
        ext = Path(image_path).suffix.lower()
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
            ".tiff": "image/tiff",
        }
        mime_type = mime_map.get(ext, "image/png")

        return f"data:{mime_type};base64,{img_data}"

    # // ---> on_message > [_describe_image_with_vllm] > POST to vLLM, returns description text
    def _describe_image_with_vllm(self, image_path: str) -> str:
        try:
            # Convert image to base64 data URL instead of file:// path.
            # CRITICAL: vLLM runs in a separate container and cannot access
            # file:// paths from this container's filesystem.
            logger.debug(f"Converting image to base64: {image_path}")
            try:
                image_url = self._image_to_base64_data_url(image_path)
                logger.debug(f"Base64 image URL created, length: {len(image_url)} chars")
            except Exception as e:
                logger.error(f"Failed to encode image {image_path} to base64: {e}")
                return "Description unavailable (image encoding failed)."

            # NOTE: Do NOT include "response_format" parameter for plain text responses.
            # vLLM v1 has a bug where any response_format triggers structured output validation
            payload = {
                "model": self.vllm_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image from a presentation slide."},
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url},
                            },
                        ],
                    }
                ],
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
                logger.info(
                    "vLLM request: POST %s payload=%s",
                    self.vllm_api_url,
                    json.dumps(payload_log, ensure_ascii=False),
                )
            except Exception as log_err:
                logger.warning(f"Failed to serialize vLLM payload for logging: {log_err}")

            logger.debug(f"Sending request to vLLM at {self.vllm_api_url}")
            resp = requests.post(self.vllm_api_url, json=payload, timeout=120)

            logger.debug(f"vLLM response status: {resp.status_code}")

            if resp.status_code != 200:
                logger.error(
                    f"vLLM returned HTTP {resp.status_code}: {resp.text[:500]}"
                )
                return f"Description unavailable (HTTP {resp.status_code})."

            resp.raise_for_status()
            data = resp.json()

            logger.debug(f"vLLM response JSON keys: {list(data.keys())}")

            # vLLM OpenAI-compatible response
            choice = (data.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            content = msg.get("content")
            if isinstance(content, str):
                return content.strip()
            # Some implementations may return list content parts
            if isinstance(content, list):
                texts = [c.get("text", "") for c in content if c.get("type") == "text"]
                return "\n".join([t for t in texts if t]).strip() or "No description returned."
            return "No description returned."
        except requests.exceptions.Timeout:
            logger.error(f"vLLM request timed out for {image_path}")
            return "Description unavailable (timeout)."
        except requests.exceptions.ConnectionError as e:
            logger.error(f"vLLM connection error for {image_path}: {e}")
            return "Description unavailable (connection error)."
        except Exception as e:
            logger.error(f"vLLM request failed for {image_path}: {type(e).__name__}: {e}")
            return "Description unavailable."


def run():

    Processor.launch(default_ident, __doc__)

