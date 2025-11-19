"""
Unstructured decoder, converts a variety of binary document uploads into textual
segments by delegating parsing to the unstructured open-source library.
"""

from __future__ import annotations

import base64
import logging
from typing import List, Optional

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
        # Prefer dedicated OCR for images and skip unstructured partition for image/* entirely
        is_image = self._is_image_content_type(content_type)
        if is_image:
            try:
                # // ---> on_message(image/*) > [_ocr_image] > returns textual segments
                segments = self._ocr_image(blob)
                if segments:
                    logger.debug(
                        "Image OCR produced %d segment(s) for %s",
                        len(segments),
                        document.metadata.id,
                    )
            except Exception as exc:  # pragma: no cover - defensive logging for OCR
                logger.error(
                    "Image OCR failed for %s: %s",
                    document.metadata.id,
                    exc,
                    exc_info=True,
                )
                segments = []

        # Only attempt unstructured partition for non-image content
        if not is_image:
            if partition is None:
                if not segments:
                    logger.warning(
                        "Unstructured partition unavailable; using fallback for %s",
                        document.metadata.id
                    )
            else:
                # Only attempt unstructured partition if we don't already have text
                if not segments:
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
    ) -> List[str]:
        return _partition_document_impl(blob=blob, metadata_filename=metadata_filename)

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

    @staticmethod
    def add_args(parser):

        FlowProcessor.add_args(parser)


# // ---> CLI entrypoint > [run] > Processor.launch
def run():

    Processor.launch(default_ident, __doc__)

