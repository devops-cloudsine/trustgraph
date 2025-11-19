"""
Helpers for detecting content types and textuality from raw blobs.
"""

from __future__ import annotations

import imghdr
import zipfile
from io import BytesIO
from typing import Optional

# // ---> Processor._filename_hint > [CONTENT_TYPE_EXTENSION_MAP] > filename hint construction
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

# // ---> _guess_content_type(image) > [IMAGE_KIND_TO_CONTENT_TYPE] > maps imghdr kinds to MIME
IMAGE_KIND_TO_CONTENT_TYPE = {
    "jpeg": "image/jpeg",
    "png": "image/png",
    "gif": "image/gif",
    "webp": "image/webp",
}

OLE_SIGNATURE = b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1"
ZIP_SIGNATURE = b"PK\x03\x04"


# // ---> Processor._fallback_segments > [_safe_text_sample] > provide sample for textual guess
def safe_text_sample(blob: bytes, limit: int = 4096) -> Optional[str]:
    snippet = blob[:limit]
    try:
        return snippet.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return snippet.decode("utf-8", errors="ignore")
        except Exception:
            return None


# // ---> _guess_content_type > [_looks_textual] > heuristics for textuality
def looks_textual(sample: str) -> bool:
    if not sample:
        return False
    printable = sum(1 for ch in sample if ch.isprintable() or ch.isspace())
    return printable / max(1, len(sample)) >= 0.6


# // ---> _guess_content_type(zip) > [_guess_zip_content_type] > OOXML family recognition
def guess_zip_content_type(blob: bytes) -> Optional[str]:
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


# // ---> _guess_content_type(ole) > [_guess_ole_content_type] > legacy MS Office recognition
def guess_ole_content_type(blob: bytes) -> Optional[str]:
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


# // ---> _guess_content_type(text) > [guess_textual_content_type] > MIME guess for text data
def guess_textual_content_type(sample: str) -> str:
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


# // ---> Processor.on_message > [guess_content_type] > detect MIME from blob when unknown
def guess_content_type(blob: bytes) -> Optional[str]:
    if blob.startswith(b"%PDF"):
        return "application/pdf"
    if blob.startswith(ZIP_SIGNATURE):
        guess = guess_zip_content_type(blob)
        if guess:
            return guess
    if blob.startswith(OLE_SIGNATURE):
        guess = guess_ole_content_type(blob)
        if guess:
            return guess

    image_kind = imghdr.what(None, blob)
    if image_kind:
        return IMAGE_KIND_TO_CONTENT_TYPE.get(image_kind)

    sample = safe_text_sample(blob)
    if sample and looks_textual(sample):
        return guess_textual_content_type(sample)

    return None


