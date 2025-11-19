"""
Partitioning helpers using unstructured.partition.auto.
"""

from __future__ import annotations

from io import BytesIO
from typing import List, Optional

try:
    # // ---> Processor._partition_document > [partition] > external library call
    from unstructured.partition.auto import partition
except Exception as import_error:  # pragma: no cover
    partition = None
    PARTITION_IMPORT_ERROR = import_error
else:
    PARTITION_IMPORT_ERROR = None

from .content_type_detection import safe_text_sample

# // ---> partition_document > [_normalize_text] > human-readable cleanup
def _normalize_text(text: str) -> str:
    # Normalize newlines and remove control chars except tabs/newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t")
    # De-hyphenate line-break hyphenations
    text = text.replace("-\n", "")
    # Collapse excessive internal whitespace while preserving paragraph breaks
    lines: List[str] = []
    for raw_line in text.split("\n"):
        line = " ".join(raw_line.split())
        lines.append(line)
    # Collapse multiple blank lines to a single blank line
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
    normalized = "\n".join(normalized_lines).strip()
    return normalized


# // ---> _partition_document > [element_text] > normalize element text
def element_text(element) -> Optional[str]:
    text = getattr(element, "text", None)
    if not text:
        return None
    cleaned = text.strip()
    return cleaned if cleaned else None


# // ---> Processor.on_message > [partition_document] > convert blob to text segments
def partition_document(
    *,
    blob: bytes,
    metadata_filename: Optional[str],
    content_type: Optional[str] = None,
    strategy: Optional[str] = None,
) -> List[str]:
    kwargs = {"file": BytesIO(blob)}
    if metadata_filename:
        kwargs["metadata_filename"] = metadata_filename
    if content_type:
        kwargs["content_type"] = content_type
    if strategy:
        kwargs["strategy"] = strategy

    elements = partition(**kwargs)  # type: ignore[misc]
    raw_segments: List[str] = [
        text for text in (element_text(element) for element in elements) if text
    ]
    # Normalize for human readability
    normalized_segments: List[str] = []
    for seg in raw_segments:
        norm = _normalize_text(seg)
        if norm:
            normalized_segments.append(norm)

    # If partition produced nothing readable, attempt a light text sample
    if not normalized_segments:
        sample = safe_text_sample(blob)
        if sample:
            sample_norm = _normalize_text(sample)
            if sample_norm:
                return [sample_norm]

    return normalized_segments


