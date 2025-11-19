"""
Partitioning helpers using unstructured.partition.auto, with modular per-type handlers.
"""

from __future__ import annotations

from typing import List, Optional

# Re-export helpers from the shared common module to preserve public surface
from .processors.common import (  # noqa: F401
    element_text,
    partition_with_auto_and_normalize,
)

# Per-type handlers (logic preserved; they delegate to the common implementation)
from .processors.doc import partition_doc  # noqa: F401
from .processors.docx import partition_docx  # noqa: F401
from .processors.ppt import partition_ppt  # noqa: F401
from .processors.pptx import partition_pptx  # noqa: F401
from .processors.xlsx import partition_xlsx  # noqa: F401


# ---> Processor.on_message > [partition_document] > dispatch to per-type handlers (no logic change)
def partition_document(
    *,
    blob: bytes,
    metadata_filename: Optional[str],
    content_type: Optional[str] = None,
    strategy: Optional[str] = None,
) -> List[str]:
    lowered = (content_type or "").strip().lower()

    # Route well-known Office types to dedicated modules.
    if lowered == "application/msword":
        return partition_doc(
            blob=blob,
            metadata_filename=metadata_filename,
            content_type=content_type,
            strategy=strategy,
        )
    if lowered == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return partition_docx(
            blob=blob,
            metadata_filename=metadata_filename,
            content_type=content_type,
            strategy=strategy,
        )
    if lowered == "application/vnd.ms-powerpoint":
        return partition_ppt(
            blob=blob,
            metadata_filename=metadata_filename,
            content_type=content_type,
            strategy=strategy,
        )
    if lowered == "application/vnd.openxmlformats-officedocument.presentationml.presentation":
        return partition_pptx(
            blob=blob,
            metadata_filename=metadata_filename,
            content_type=content_type,
            strategy=strategy,
        )
    if lowered == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        return partition_xlsx(
            blob=blob,
            metadata_filename=metadata_filename,
            content_type=content_type,
            strategy=strategy,
        )

    # Unknown/other types: preserve existing behavior by using the same auto-partition path.
    return partition_with_auto_and_normalize(
        blob=blob,
        metadata_filename=metadata_filename,
        content_type=content_type,
        strategy=strategy,
    )


