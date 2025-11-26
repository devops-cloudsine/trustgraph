"""
Unstructured decoder, converts a variety of binary document uploads into textual
segments by delegating parsing to the unstructured open-source library.
Optionally extracts images from Office documents and describes them via vLLM.
"""

from __future__ import annotations

import base64
import logging
import json
from typing import List, Optional

import requests

from ...schema import Document, TextDocument
from ...base import FlowProcessor, ConsumerSpec, ProducerSpec
from .content_type_detection import (
    CONTENT_TYPE_EXTENSION_MAP,
    guess_content_type as _guess_content_type,
    IMAGE_KIND_TO_CONTENT_TYPE,
)
from .image_processing import (
    is_image_content_type as _is_image_content_type_impl,
    ocr_image as _ocr_image_impl,
)
from .partition_processing import (
    partition_document as _partition_document_impl,
    element_text as _element_text_impl,
)
from .fallback_processing import (
    fallback_segments as _fallback_segments_impl,
)
from .office_image_extraction import (
    is_image_extractable_content_type as _is_image_extractable_impl,
    extract_images_from_office as _extract_images_impl,
    DOCX_CONTENT_TYPE,
    DOC_CONTENT_TYPE,
)

# Content types handled by the dedicated docx-decoder (OCR container)
# These should be skipped by unstructured-decoder to avoid duplicate processing
DOCX_OCR_HANDLED_TYPES = {DOCX_CONTENT_TYPE, DOC_CONTENT_TYPE}

try:
    from unstructured.partition.auto import partition
except Exception as import_error:  # pragma: no cover
    partition = None
    PARTITION_IMPORT_ERROR = import_error
else:
    PARTITION_IMPORT_ERROR = None

logger = logging.getLogger(__name__)
default_ident = "unstructured-decoder"

 


class Processor(FlowProcessor):

    def __init__(self, **params):

        # If unstructured is unavailable, enter fallback-only mode instead of crashing
        if partition is None:  # pragma: no cover - handled at runtime
            logger.warning(
                "unstructured.partition.auto.partition is unavailable: %s; "
                "decoder will use fallback text sampling only.",
                PARTITION_IMPORT_ERROR
            )

        id = params.get("id", default_ident)

        # vLLM configuration for image description (optional)
        self.vllm_api_url = params.get("vllm_api_url", None)
        self.vllm_model = params.get(
            "vllm_model",
            "Qwen/Qwen3-VL-4B-Instruct"
        )
        self.enable_image_extraction = params.get("enable_image_extraction", False)

        # Enable image extraction if vLLM URL is provided
        if self.vllm_api_url:
            self.enable_image_extraction = True
            logger.info(
                "vLLM image description enabled: %s (model: %s)",
                self.vllm_api_url,
                self.vllm_model
            )

        super(Processor, self).__init__(
            **params | {
                "id": id,
            }
        )

        self.register_specification(
            ConsumerSpec(
                name="input",
                schema=Document,
                handler=self.on_message,
            )
        )

        self.register_specification(
            ProducerSpec(
                name="output",
                schema=TextDocument,
            )
        )

        logger.info("Unstructured decoder initialized")

    # // ---> Pulsar consumer(input) > [on_message] > flow('output').send(TextDocument)
    async def on_message(self, msg, consumer, flow):

        document: Document = msg.value()
        blob = base64.b64decode(document.data)

        declared_type = self._normalize_content_type(
            getattr(document, "content_type", None)
        )
        guessed_type = declared_type or _guess_content_type(blob)
        content_type = guessed_type or "application/octet-stream"

        # // ---> Skip DOCX/DOC files - they are handled by the dedicated docx-decoder (OCR container)
        if content_type in DOCX_OCR_HANDLED_TYPES:
            logger.info(
                "Skipping %s (content_type=%s) - handled by docx-decoder",
                document.metadata.id,
                content_type,
            )
            return

        filename_hint = self._filename_hint(
            getattr(document, "filename", None),
            content_type,
        )

        logger.debug(
            "Decoding %s as %s with filename hint %s",
            document.metadata.id,
            content_type,
            filename_hint,
        )

        segments: List[str] = []
        is_image = self._is_image_content_type(content_type)
        # For images, prefer unstructured partition with hi_res strategy first per docs
        if is_image:
            if partition is None:
                logger.warning(
                    "Unstructured partition unavailable; attempting OCR for %s",
                    document.metadata.id,
                )
                try:
                    segments = self._ocr_image(blob)
                except Exception as exc:  # pragma: no cover
                    logger.error(
                        "Image OCR failed for %s: %s",
                        document.metadata.id,
                        exc,
                        exc_info=True,
                    )
                    segments = []
            else:
                try:
                    segments = self._partition_document(
                        blob=blob,
                        metadata_filename=filename_hint,
                        content_type=content_type,
                        strategy="hi_res",
                    )
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.error(
                        "unstructured image partition(hi_res) failed for %s: %s",
                        document.metadata.id,
                        exc,
                        exc_info=True,
                    )
                    segments = []
                # If hi_res produced nothing, fall back to OCR to recover text
                if not segments:
                    try:
                        segments = self._ocr_image(blob)
                        if segments:
                            logger.debug(
                                "Image OCR produced %d segment(s) for %s",
                                len(segments),
                                document.metadata.id,
                            )
                    except Exception as exc:  # pragma: no cover
                        logger.error(
                            "Image OCR failed for %s: %s",
                            document.metadata.id,
                            exc,
                            exc_info=True,
                        )
                        segments = []
        else:
            # Non-image content: use unstructured partition if available
            if partition is None:
                logger.warning(
                    "Unstructured partition unavailable; using fallback for %s",
                    document.metadata.id
                )
            else:
                try:
                    segments = self._partition_document(
                        blob=blob,
                        metadata_filename=filename_hint,
                    )
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.error(
                        "unstructured partition failed for %s: %s",
                        document.metadata.id,
                        exc,
                        exc_info=True,
                    )
                    segments = []

            # Extract and describe images from Office documents (DOCX/DOC)
            if self.enable_image_extraction and _is_image_extractable_impl(content_type):
                image_descriptions = await self._extract_and_describe_images(
                    blob=blob,
                    content_type=content_type,
                    doc_id=document.metadata.id,
                )
                segments.extend(image_descriptions)

        if not segments:
            segments = self._fallback_segments(blob, content_type)

        for segment in segments:
            text = segment.strip()
            if not text:
                continue
            payload = TextDocument(
                metadata=document.metadata,
                text=text.encode("utf-8"),
            )
            await flow("output").send(payload)

    def _normalize_content_type(self, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return value.strip().lower()

    def _filename_hint(
        self,
        explicit_filename: Optional[str],
        content_type: Optional[str],
    ) -> Optional[str]:
        if explicit_filename:
            return explicit_filename
        if content_type in CONTENT_TYPE_EXTENSION_MAP:
            return f"document{CONTENT_TYPE_EXTENSION_MAP[content_type]}"
        return None

    # // ---> on_message > [_partition_document] > calls unstructured.partition.auto.partition
    def _partition_document(
        self,
        *,
        blob: bytes,
        metadata_filename: Optional[str],
        content_type: Optional[str] = None,
        strategy: Optional[str] = None,
    ) -> List[str]:
        return _partition_document_impl(
            blob=blob,
            metadata_filename=metadata_filename,
            content_type=content_type,
            strategy=strategy,
        )

    # // ---> on_message > [_is_image_content_type] > guards OCR branch for images
    def _is_image_content_type(self, content_type: Optional[str]) -> bool:
        return _is_image_content_type_impl(content_type)

    # // ---> on_message(image/*) > [_ocr_image] > returns textual segments
    def _ocr_image(self, blob: bytes) -> List[str]:
        return _ocr_image_impl(blob)

    @staticmethod
    def _element_text(element) -> Optional[str]:
        return _element_text_impl(element)

    # // ---> on_message > [_fallback_segments] > ensures downstream processors receive text
    def _fallback_segments(
        self,
        blob: bytes,
        content_type: Optional[str],
    ) -> List[str]:
        return _fallback_segments_impl(blob, content_type)

    # // ---> on_message(DOCX/DOC) > [_extract_and_describe_images] > extract images and get vLLM descriptions
    async def _extract_and_describe_images(
        self,
        blob: bytes,
        content_type: str,
        doc_id: str,
    ) -> List[str]:
        """
        Extract images from Office documents and describe them via vLLM.
        Returns a list of image description text segments.
        """
        if not self.vllm_api_url:
            return []

        descriptions = []
        
        try:
            images = _extract_images_impl(blob, content_type)
            logger.info(
                "Extracted %d images from %s for document %s",
                len(images),
                content_type,
                doc_id
            )
        except Exception as exc:
            logger.error(
                "Image extraction failed for %s: %s",
                doc_id,
                exc,
                exc_info=True,
            )
            return []

        for ix, (image_bytes, image_ext) in enumerate(images):
            try:
                description = self._describe_image_with_vllm(image_bytes, image_ext)
                if description and description.strip():
                    # Format as a clear image description segment
                    formatted = f"[Embedded Image {ix + 1}]\n{description}"
                    descriptions.append(formatted)
                    logger.debug(
                        "Image %d/%d described for %s",
                        ix + 1,
                        len(images),
                        doc_id
                    )
            except Exception as exc:
                logger.warning(
                    "Failed to describe image %d for %s: %s",
                    ix + 1,
                    doc_id,
                    exc
                )
                continue

        return descriptions

    # // ---> _extract_and_describe_images > [_image_bytes_to_base64_data_url] > encode image for vLLM
    def _image_bytes_to_base64_data_url(self, image_bytes: bytes, ext: str) -> str:
        """
        Convert image bytes to a base64 data URL.
        Required because vLLM may run in a separate container.
        """
        img_data = base64.b64encode(image_bytes).decode("utf-8")
        
        mime_map = {
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "gif": "image/gif",
            "webp": "image/webp",
            "bmp": "image/bmp",
            "tiff": "image/tiff",
        }
        mime_type = mime_map.get(ext.lower(), "image/png")
        
        return f"data:{mime_type};base64,{img_data}"

    # // ---> _extract_and_describe_images > [_describe_image_with_vllm] > POST to vLLM, returns description
    def _describe_image_with_vllm(self, image_bytes: bytes, ext: str) -> str:
        """
        Send image to vLLM for description.
        """
        if not self.vllm_api_url:
            return ""

        try:
            image_url = self._image_bytes_to_base64_data_url(image_bytes, ext)
            logger.debug(f"Base64 image URL created, length: {len(image_url)} chars")
        except Exception as e:
            logger.error(f"Failed to encode image to base64: {e}")
            return "Description unavailable (image encoding failed)."

        # NOTE: Do NOT include "response_format" parameter for plain text responses.
        # vLLM v1 has a bug where any response_format triggers structured output validation.
        payload = {
            "model": self.vllm_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image in detail."},
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
            logger.debug(
                "vLLM request: POST %s payload=%s",
                self.vllm_api_url,
                json.dumps(payload_log, ensure_ascii=False),
            )
        except Exception as log_err:
            logger.warning(f"Failed to serialize vLLM payload for logging: {log_err}")

        try:
            resp = requests.post(self.vllm_api_url, json=payload, timeout=120)
            
            logger.debug(f"vLLM response status: {resp.status_code}")
            
            if resp.status_code != 200:
                logger.error(
                    f"vLLM returned HTTP {resp.status_code}: {resp.text[:500]}"
                )
                return f"Description unavailable (HTTP {resp.status_code})."
            
            resp.raise_for_status()
            data = resp.json()
            
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

    @staticmethod
    def add_args(parser):

        FlowProcessor.add_args(parser)

        parser.add_argument(
            '--vllm-api-url',
            default=None,
            help='vLLM API URL for image description (e.g., http://vllm:8000/v1/chat/completions). '
                 'If provided, enables image extraction from Office documents.'
        )

        parser.add_argument(
            '--vllm-model',
            default='Qwen/Qwen3-VL-4B-Instruct',
            help='Vision model name for vLLM image description (default: Qwen/Qwen3-VL-4B-Instruct)'
        )


# // ---> CLI entrypoint > [run] > Processor.launch
def run():

    Processor.launch(default_ident, __doc__)

