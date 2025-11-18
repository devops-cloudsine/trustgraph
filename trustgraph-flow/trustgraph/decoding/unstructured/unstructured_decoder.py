"""
Unstructured decoder, converts a variety of binary document uploads into textual
segments by delegating parsing to the unstructured open-source library.
"""

from __future__ import annotations

import base64
import imghdr
import logging
import zipfile
from io import BytesIO
from typing import List, Optional

from ...schema import Document, TextDocument
from ...base import FlowProcessor, ConsumerSpec, ProducerSpec

try:
    from unstructured.partition.auto import partition
except Exception as import_error:  # pragma: no cover
    partition = None
    PARTITION_IMPORT_ERROR = import_error
else:
    PARTITION_IMPORT_ERROR = None

logger = logging.getLogger(__name__)
default_ident = "unstructured-decoder"

CONTENT_TYPE_EXTENSION_MAP = {
    "application/pdf": ".pdf",
    "text/csv": ".csv",
    "application/msword": ".doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "image/gif": ".gif",
    "text/html": ".html",
    "text/calendar": ".ics",
    "image/jpeg": ".jpg",
    "application/json": ".json",
    "text/markdown": ".md",
    "image/png": ".png",
    "application/vnd.ms-powerpoint": ".ppt",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "image/webp": ".webp",
    "application/vnd.ms-excel": ".xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "text/plain": ".txt",
}

IMAGE_KIND_TO_CONTENT_TYPE = {
    "jpeg": "image/jpeg",
    "png": "image/png",
    "gif": "image/gif",
    "webp": "image/webp",
}

OLE_SIGNATURE = b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1"
ZIP_SIGNATURE = b"PK\x03\x04"


def _safe_text_sample(blob: bytes, limit: int = 4096) -> Optional[str]:
    snippet = blob[:limit]
    try:
        return snippet.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return snippet.decode("utf-8", errors="ignore")
        except Exception:
            return None


def _looks_textual(sample: str) -> bool:
    if not sample:
        return False
    printable = sum(1 for ch in sample if ch.isprintable() or ch.isspace())
    return printable / max(1, len(sample)) >= 0.6


def _guess_zip_content_type(blob: bytes) -> Optional[str]:
    try:
        with zipfile.ZipFile(BytesIO(blob)) as archive:
            names = archive.namelist()
    except zipfile.BadZipFile:
        return None

    if any(name.startswith("word/") for name in names):
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if any(name.startswith("ppt/") for name in names):
        return "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    if any(name.startswith("xl/") for name in names):
        return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    return None


def _guess_ole_content_type(blob: bytes) -> Optional[str]:
    if not blob.startswith(OLE_SIGNATURE):
        return None
    lowered = blob.lower()
    if b"worddocument" in lowered:
        return "application/msword"
    if b"powerpoint document" in lowered:
        return "application/vnd.ms-powerpoint"
    if b"workbook" in lowered:
        return "application/vnd.ms-excel"
    return None


def _guess_textual_content_type(sample: str) -> str:
    stripped = sample.lstrip()
    lowered = stripped.lower()

    if lowered.startswith("{") or lowered.startswith("["):
        return "application/json"
    if "<html" in lowered or "<!doctype html" in lowered:
        return "text/html"
    if "begin:vcalendar" in lowered:
        return "text/calendar"

    first_line = stripped.splitlines()[0] if stripped else ""
    if ("," in first_line or "\t" in first_line or ";" in first_line) and "\n" in sample:
        return "text/csv"
    if first_line.strip().startswith("#") or "```" in sample:
        return "text/markdown"
    return "text/plain"


def _guess_content_type(blob: bytes) -> Optional[str]:
    if blob.startswith(b"%PDF"):
        return "application/pdf"
    if blob.startswith(ZIP_SIGNATURE):
        guess = _guess_zip_content_type(blob)
        if guess:
            return guess
    if blob.startswith(OLE_SIGNATURE):
        guess = _guess_ole_content_type(blob)
        if guess:
            return guess

    image_kind = imghdr.what(None, blob)
    if image_kind:
        return IMAGE_KIND_TO_CONTENT_TYPE.get(image_kind)

    sample = _safe_text_sample(blob)
    if sample and _looks_textual(sample):
        return _guess_textual_content_type(sample)

    return None


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
        kwargs = {"file": BytesIO(blob)}
        if metadata_filename:
            kwargs["metadata_filename"] = metadata_filename

        elements = partition(**kwargs)
        return [
            text for text in (self._element_text(element) for element in elements)
            if text
        ]

    @staticmethod
    def _element_text(element) -> Optional[str]:
        text = getattr(element, "text", None)
        if not text:
            return None
        cleaned = text.strip()
        return cleaned if cleaned else None

    # // ---> on_message > [_fallback_segments] > ensures downstream processors receive text
    def _fallback_segments(
        self,
        blob: bytes,
        content_type: Optional[str],
    ) -> List[str]:
        sample = _safe_text_sample(blob)
        if sample and sample.strip():
            return [sample]

        logger.warning(
            "Unable to decode %s content, emitting placeholder text.",
            content_type or "unknown",
        )
        return ["[binary document contents elided]"]

    @staticmethod
    def add_args(parser):

        FlowProcessor.add_args(parser)


# // ---> CLI entrypoint > [run] > Processor.launch
def run():

    Processor.launch(default_ident, __doc__)

