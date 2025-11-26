"""
XLSX decoder: Extracts text from XLS/XLSX spreadsheets using the unstructured library.
All content is output as structured text segments suitable for RAG processing.

Key features:
- Uses unstructured.partition.auto for parsing (free, open-source)
- Handles both legacy .xls and modern .xlsx formats
- Extracts cell data, sheet names, and table structures
- No vision model required (spreadsheets are text-based)

Reference: https://docs.unstructured.io/
"""

import base64
import logging
import zipfile
from io import BytesIO
from typing import List, Optional

from ... schema import Document as TGDocument, TextDocument
from ... base import FlowProcessor, ConsumerSpec, ProducerSpec

# Module logger
logger = logging.getLogger(__name__)

# Try to import unstructured
try:
    from unstructured.partition.auto import partition
    UNSTRUCTURED_AVAILABLE = True
except ImportError as e:
    UNSTRUCTURED_AVAILABLE = False
    partition = None
    logger.warning(f"unstructured not available: {e}")

default_ident = "xlsx-decoder"

# Content types for XLS/XLSX files
XLSX_CONTENT_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
XLS_CONTENT_TYPE = "application/vnd.ms-excel"
XLSX_HANDLED_TYPES = {XLSX_CONTENT_TYPE, XLS_CONTENT_TYPE}

# Magic bytes
ZIP_SIGNATURE = b"PK\x03\x04"
OLE_SIGNATURE = b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1"


class Processor(FlowProcessor):

    def __init__(self, **params):
        id = params.get("id", default_ident)

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

        if not UNSTRUCTURED_AVAILABLE:
            logger.warning("XLSX decoder initialized but unstructured library not available")
        else:
            logger.info("XLSX decoder initialized (unstructured)")

    # // ---> Pulsar consumer(input) > [on_message] > extract text via unstructured -> flow('output')
    async def on_message(self, msg, consumer, flow):
        logger.info("XLSX message received for extraction")

        v = msg.value()
        logger.info(f"Processing {v.metadata.id}...")

        blob = base64.b64decode(v.data)

        # Check content type first
        content_type = self._normalize_content_type(
            getattr(v, "content_type", None)
        )

        # If content type is declared and not XLS/XLSX, skip
        if content_type and content_type not in XLSX_HANDLED_TYPES:
            logger.info(f"Skipping non-XLSX file: {v.metadata.id} (content_type: {content_type})")
            return

        # If no content type declared, detect from blob
        if not content_type:
            content_type = self._guess_content_type(blob)

        if not content_type or content_type not in XLSX_HANDLED_TYPES:
            logger.info(f"Skipping non-XLSX file: {v.metadata.id} (detected: {content_type})")
            return

        # Determine filename hint for unstructured
        filename_hint = self._filename_hint(
            getattr(v, "filename", None),
            content_type
        )

        logger.debug(
            "Decoding %s as %s with filename hint %s",
            v.metadata.id,
            content_type,
            filename_hint,
        )

        # Extract text using unstructured
        segments = self._partition_document(
            blob=blob,
            metadata_filename=filename_hint,
            content_type=content_type,
        )

        if not segments:
            logger.warning(f"No text extracted from {v.metadata.id}")
            return

        logger.info(f"Extracted {len(segments)} segments from {v.metadata.id}")

        # Send each segment
        for segment in segments:
            text = segment.strip()
            if not text:
                continue
            r = TextDocument(
                metadata=v.metadata,
                text=text.encode("utf-8"),
            )
            await flow("output").send(r)

        logger.info(f"XLSX extraction complete: {len(segments)} segments sent")

    # // ---> on_message > [_normalize_content_type] > normalize content type string
    def _normalize_content_type(self, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return value.strip().lower()

    # // ---> on_message > [_filename_hint] > generate filename hint for unstructured
    def _filename_hint(
        self,
        explicit_filename: Optional[str],
        content_type: Optional[str],
    ) -> Optional[str]:
        if explicit_filename:
            return explicit_filename
        if content_type == XLSX_CONTENT_TYPE:
            return "document.xlsx"
        if content_type == XLS_CONTENT_TYPE:
            return "document.xls"
        return None

    # // ---> on_message > [_guess_content_type] > detect content type from blob bytes
    def _guess_content_type(self, blob: bytes) -> Optional[str]:
        """Detect XLS/XLSX content type from blob bytes."""
        # XLSX is a ZIP file with xl/ directory
        if blob.startswith(ZIP_SIGNATURE):
            try:
                with zipfile.ZipFile(BytesIO(blob)) as archive:
                    names = archive.namelist()
                    if any(name.startswith("xl/") for name in names):
                        return XLSX_CONTENT_TYPE
            except zipfile.BadZipFile:
                pass

        # XLS is an OLE file with "Workbook" marker
        if blob.startswith(OLE_SIGNATURE):
            if b"workbook" in blob.lower():
                return XLS_CONTENT_TYPE

        return None

    # // ---> on_message > [_partition_document] > use unstructured to parse spreadsheet
    def _partition_document(
        self,
        *,
        blob: bytes,
        metadata_filename: Optional[str],
        content_type: Optional[str] = None,
    ) -> List[str]:
        """
        Parse spreadsheet using unstructured library.
        Returns list of text segments.
        """
        if not UNSTRUCTURED_AVAILABLE or partition is None:
            logger.error("unstructured library not available")
            return []

        try:
            kwargs = {"file": BytesIO(blob)}
            if metadata_filename:
                kwargs["metadata_filename"] = metadata_filename
            if content_type:
                kwargs["content_type"] = content_type

            elements = partition(**kwargs)

            # Extract text from elements
            raw_segments: List[str] = []
            for element in elements:
                text = self._element_text(element)
                if text:
                    raw_segments.append(text)

            # Normalize text for readability
            normalized_segments: List[str] = []
            for seg in raw_segments:
                norm = self._normalize_text(seg)
                if norm:
                    normalized_segments.append(norm)

            return normalized_segments

        except Exception as e:
            logger.error(f"unstructured partition failed: {e}", exc_info=True)
            return []

    # // ---> _partition_document > [_element_text] > extract text from element
    @staticmethod
    def _element_text(element) -> Optional[str]:
        text = getattr(element, "text", None)
        if not text:
            return None
        cleaned = text.strip()
        return cleaned if cleaned else None

    # // ---> _partition_document > [_normalize_text] > clean up text for readability
    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text for human readability."""
        # Normalize newlines
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Remove control chars except tabs/newlines
        text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t")
        # De-hyphenate line-break hyphenations
        text = text.replace("-\n", "")
        # Collapse excessive whitespace
        lines: List[str] = []
        for raw_line in text.split("\n"):
            line = " ".join(raw_line.split())
            lines.append(line)
        # Collapse multiple blank lines
        normalized_lines: List[str] = []
        blank_streak = 0
        for line in lines:
            if line == "":
                blank_streak += 1
                if blank_streak == 1:
                    normalized_lines.append("")
            else:
                blank_streak = 0
                normalized_lines.append(line)
        return "\n".join(normalized_lines).strip()

    @staticmethod
    def add_args(parser):
        FlowProcessor.add_args(parser)


# // ---> CLI entrypoint > [run] > Processor.launch
def run():
    Processor.launch(default_ident, __doc__)

