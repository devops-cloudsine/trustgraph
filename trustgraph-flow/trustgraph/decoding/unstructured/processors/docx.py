"""
DOCX (.docx) partition handler (delegates to auto partition to preserve logic).
Reference: `partition_docx` in Unstructured docs.
"""

from __future__ import annotations

from typing import List, Optional

from .common import partition_with_auto_and_normalize


# ---> partition_document(content_type=application/vnd.openxmlformats-officedocument.wordprocessingml.document) > [partition_docx] > common.partition_with_auto_and_normalize
def partition_docx(
	*,
	blob: bytes,
	metadata_filename: Optional[str],
	content_type: Optional[str] = None,
	strategy: Optional[str] = None,
) -> List[str]:
	return partition_with_auto_and_normalize(
		blob=blob,
		metadata_filename=metadata_filename,
		content_type=content_type,
		strategy=strategy,
	)



