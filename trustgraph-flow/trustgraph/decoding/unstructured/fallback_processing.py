"""
Fallback strategies when structured decoding fails.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from .content_type_detection import safe_text_sample

logger = logging.getLogger(__name__)


# // ---> Processor.on_message > [fallback_segments] > ensures downstream receives some text
def fallback_segments(
    blob: bytes,
    content_type: Optional[str],
) -> List[str]:
    sample = safe_text_sample(blob)
    if sample and sample.strip():
        return [sample]

    logger.warning(
        "Unable to decode %s content, emitting placeholder text.",
        content_type or "unknown",
    )
    return ["[binary document contents elided]"]


