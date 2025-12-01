
"""
Simple decoder, accepts PDF documents on input, outputs pages from the
PDF document as text as separate output objects.

Images are stored in MinIO object storage for persistence and sharing
with vLLM container via mounted volume.
"""

import base64
import logging
import os
import re
from io import BytesIO
from pathlib import Path
from pdf2image import convert_from_bytes
import requests
import json

from ... schema import Document, TextDocument, Metadata
from ... base import FlowProcessor, ConsumerSpec, ProducerSpec
from .minio_storage import MinioStorage, get_minio_storage

# Module logger
logger = logging.getLogger(__name__)

# Enable DEBUG level for this module when VLLM_LOGGING_LEVEL=DEBUG is set
if os.environ.get("VLLM_LOGGING_LEVEL", "").upper() == "DEBUG":
    logger.setLevel(logging.DEBUG)

default_ident = "pdf-ocr"

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
        
        # Initialize MinIO storage for image persistence
        self.minio_storage = get_minio_storage()

        super(Processor, self).__init__(
            **params | {
                "id": id,
            }
        )

        self.register_specification(
            ConsumerSpec(
                name = "input",
                schema = Document,
                handler = self.on_message,
            )
        )

        self.register_specification(
            ProducerSpec(
                name = "output",
                schema = TextDocument,
            )
        )

        logger.info("PDF image extraction + vLLM description processor initialized")

    # // ---> Pulsar consumer(input) > [on_message] > save page images to MinIO, vLLM describe -> flow('output')
    async def on_message(self, msg, consumer, flow):

        logger.info("PDF message received for image extraction")

        v = msg.value()
        doc_id = v.metadata.id

        logger.info(f"Processing {doc_id}...")

        blob = base64.b64decode(v.data)

        # Check if this is a PDF file (magic bytes: %PDF)
        if not self._is_pdf(blob):
            logger.info(f"Skipping non-PDF file: {doc_id}")
            return

        # Initialize MinIO storage
        if not self.minio_storage.initialize():
            logger.error("Failed to initialize MinIO storage - cannot process PDF")
            return

        # Save original PDF document to MinIO
        pdf_filename = f"{self._safe_id(doc_id)}.pdf"
        doc_object_name = self.minio_storage.save_document(
            doc_id=doc_id,
            filename=pdf_filename,
            document_data=blob,
            content_type="application/pdf"
        )
        if doc_object_name:
            logger.info(f"Saved original PDF to MinIO: {doc_object_name}")
        else:
            logger.warning("Failed to save original PDF to MinIO, continuing with processing")

        pages = convert_from_bytes(blob)

        for ix, page in enumerate(pages):
            filename = f"page_{ix + 1}.png"
            
            try:
                # Convert PIL image to bytes
                img_buffer = BytesIO()
                page.save(img_buffer, "PNG")
                img_bytes = img_buffer.getvalue()
                
                # Save to MinIO storage
                object_name = self.minio_storage.save_image(
                    doc_id=doc_id,
                    filename=filename,
                    image_data=img_bytes,
                    content_type="image/png"
                )
                
                if object_name:
                    local_path = self.minio_storage.get_local_path(object_name)
                    logger.debug(f"Saved page image to MinIO: {object_name} (local: {local_path})")
                else:
                    logger.warning(f"Failed saving page image {ix + 1} to MinIO")
                    continue
                    
            except Exception as e:
                logger.warning(f"Failed saving page image {ix + 1}: {e}")
                continue

            # Describe image via vLLM using the image bytes directly
            description = self._describe_image_with_vllm_bytes(img_bytes, "image/png")
            page_text = f"Page {ix + 1} Image Description:\n{description}\n"

            r = TextDocument(
                metadata=v.metadata,
                text=page_text.encode("utf-8"),
            )

            await flow("output").send(r)

        logger.info("PDF page image descriptions complete")

    @staticmethod
    def add_args(parser):
        FlowProcessor.add_args(parser)

    # // ---> on_message > [_is_pdf] > check if blob is a PDF file
    def _is_pdf(self, blob: bytes) -> bool:
        """Check if the blob is a PDF file by examining magic bytes."""
        # PDF files start with %PDF
        return blob[:4] == b'%PDF'

    # // ---> on_message > [_safe_id] > sanitize directory name
    def _safe_id(self, value: str) -> str:
        if value is None:
            return "unknown"
        return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))

    # // ---> on_message > [_bytes_to_base64_data_url] > convert image bytes to data:image/... URL
    def _bytes_to_base64_data_url(self, image_bytes: bytes, content_type: str = "image/png") -> str:
        """
        Convert image bytes to a base64 data URL.
        """
        img_data = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:{content_type};base64,{img_data}"

    # // ---> on_message > [_describe_image_with_vllm_bytes] > POST to vLLM with image bytes, returns description
    def _describe_image_with_vllm_bytes(self, image_bytes: bytes, content_type: str = "image/png") -> str:
        """
        Describe an image using vLLM by sending image bytes as base64.
        
        Args:
            image_bytes: Raw image bytes
            content_type: MIME type of the image
            
        Returns:
            Text description of the image
        """
        try:
            # Convert image bytes to base64 data URL
            logger.debug(f"Converting image bytes to base64 ({len(image_bytes)} bytes)")
            try:
                image_url = self._bytes_to_base64_data_url(image_bytes, content_type)
                logger.debug(f"Base64 image URL created, length: {len(image_url)} chars")
            except Exception as e:
                logger.error(f"Failed to encode image to base64: {e}")
                return "Description unavailable (image encoding failed)."

            # NOTE: Do NOT include "response_format" parameter for plain text responses.
            # vLLM v1 has a bug where any response_format (even {"type": "text"}) triggers
            # structured output validation that fails with "No valid structured output parameter found"
            payload = {
                "model": self.vllm_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image."},
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
            logger.error("vLLM request timed out")
            return "Description unavailable (timeout)."
        except requests.exceptions.ConnectionError as e:
            logger.error(f"vLLM connection error: {e}")
            return "Description unavailable (connection error)."
        except Exception as e:
            logger.error(f"vLLM request failed: {type(e).__name__}: {e}")
            return "Description unavailable."

def run():

    Processor.launch(default_ident, __doc__)

