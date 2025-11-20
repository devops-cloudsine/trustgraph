#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import base64
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests


def _join_url(base_url: str, path: str) -> str:
    if path.startswith("http://") or path.startswith("https://"):
        return path
    return base_url.rstrip("/") + "/" + path.lstrip("/")


# ---> main > [ext_to_mime mapping] > payload_2['document-metadata']['kind']
EXT_TO_MIME: Dict[str, str] = {
    "txt": "text/plain",
    "csv": "text/csv",
    "html": "text/html",
    "ics": "text/calendar",
    "md": "text/markdown",
    "json": "application/json",
    "pdf": "application/pdf",
    "doc": "application/msword",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "gif": "image/gif",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "ppt": "application/vnd.ms-powerpoint",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "webp": "image/webp",
    "xls": "application/vnd.ms-excel",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


# ---> main > [guess_mime_from_extension] > payload_2['document-metadata']['kind']
def guess_mime_from_extension(file_path: str) -> Tuple[str, str]:
    """
    Returns (extension_without_dot_lower, mime_type)
    Defaults to ('bin', 'application/octet-stream') if unknown.
    """
    ext = Path(file_path).suffix.lower().lstrip(".")
    if not ext:
        return "bin", "application/octet-stream"
    mime = EXT_TO_MIME.get(ext, "application/octet-stream")
    return ext, mime


# ---> main > [encode_file_to_base64] > payload_2['content']
def encode_file_to_base64(file_path: str) -> str:
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("utf-8")


# ---> main > [post_json] > requests.post
def post_json(base_url: str, path: str, payload: Dict[str, Any], timeout: float) -> requests.Response:
    url = _join_url(base_url, path)
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    return response


def _as_curl(method: str, url: str, payload: Dict[str, Any]) -> str:
    data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    return f"curl -sS -X {method.upper()} '{url}' -H 'Content-Type: application/json' -d '{data}'"


# ---> main > [print_response] > stdout logs
def print_response(stage: str, method: str, url: str, resp: Optional[requests.Response]) -> None:
    print("=" * 80)
    print(f"[{stage}] {method.upper()} {url}")
    if resp is None:
        print("No response (request failed before receiving a response).")
        print("=" * 80)
        return
    print(f"Status: {resp.status_code}")
    content_type = resp.headers.get("Content-Type", "")
    body_text: str
    try:
        if "application/json" in content_type.lower():
            parsed = resp.json()
            body_text = json.dumps(parsed, indent=2, ensure_ascii=False)
        else:
            body_text = resp.text
    except Exception:
        body_text = resp.text
    print("Body:")
    print(body_text)
    print("=" * 80)


# ---> main > [request_with_retries] > post_json -> print_response
def request_with_retries(
    base_url: str,
    path: str,
    payload: Dict[str, Any],
    timeout: float,
    stage: str,
    method: str = "POST",
    retries: int = 0,
    retry_wait: float = 1.0,
) -> Optional[requests.Response]:
    """
    Perform a request with best-effort retries to achieve HTTP 200.
    Returns the final response (may still be non-200 if all attempts fail).
    """
    url = _join_url(base_url, path)
    attempt = 0
    last_resp: Optional[requests.Response] = None
    while True:
        try:
            resp = post_json(base_url, path, payload, timeout)
            last_resp = resp
        except Exception as e:
            print_response(stage, method, url, None)
            print(f"Request error: {e}")
            if attempt >= retries:
                return None
            attempt += 1
            time.sleep(retry_wait)
            continue

        print_response(stage, method, url, last_resp)
        if last_resp.status_code == 200:
            return last_resp
        if attempt >= retries:
            return last_resp
        attempt += 1
        time.sleep(retry_wait)


# ---> CLI entry > [main] > sequential HTTP requests
def main() -> int:
    parser = argparse.ArgumentParser(description="Run TrustGraph flow/librarian sequence and print responses.")
    parser.add_argument(
        "--base-url",
        default="http://98.86.194.106:8088",
        help="Base URL of the API service (default: %(default)s)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP request timeout in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--file-path",
        type=str,
        default="",
        help="Path to a local file to upload as document content (base64-encoded).",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="",
        help="Override collection name used in stages 3 and 4. Defaults to file extension.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Number of retries per request when status is not 200 (default: %(default)s).",
    )
    parser.add_argument(
        "--retry-wait",
        type=float,
        default=1.0,
        help="Seconds to wait between retries (default: %(default)s).",
    )
    args = parser.parse_args()

    base_url: str = args.base_url
    timeout: float = args.timeout
    file_path: str = args.file_path.strip()
    retries: int = max(0, int(args.retries))
    retry_wait: float = max(0.0, float(args.retry_wait))

    # Prepare dynamic identifiers
    doc_id: str = f"doc-{uuid.uuid4().hex[:8]}"
    proc_id: str = f"proc-{uuid.uuid4().hex[:8]}"

    # Derive kind and collection from file when provided
    collection_name: str
    kind_value: str
    title_value: str
    content_b64: str

    if file_path:
        if not os.path.isfile(file_path):
            print(f"Provided --file-path does not exist or is not a file: {file_path}")
            return 2
        ext, mime = guess_mime_from_extension(file_path)
        kind_value = mime
        collection_name = args.collection.strip() or ext or "bin"
        title_value = Path(file_path).name
        content_b64 = encode_file_to_base64(file_path)
    else:
        # Fallback to existing HTML sample content
        collection_name = args.collection.strip() or "html"
        kind_value = "text/html"
        title_value = "HTML Sample"
        content_b64 = (
            "PCFET0NUWVBFIGh0bWw+CjxodG1sIGxhbmc9ImVuIj4KPGhlYWQ+CiAgICA8bWV0YSBjaGFyc2V0PSJVVEYtOCI+CiAgICA8dGl0bGU+TXkgSW5mbzwvdGl0bGU+CjwvaGVhZD4KPGJvZHk+CiAgICA8aDE+TXkgQmlydGhkYXk8L2gxPgogICAgPHA+TXkgYmlydGhkYXkgaXMgb24gSmFudWFyeSAxLCAyMDAwLjwvcD4KPC9ib2R5Pgo8L2h0bWw+Cgo="
        )

    try:
        # 1) Start flow
        stage = "1/4 Start flow"
        path = "/api/v1/flow"
        payload_1: Dict[str, Any] = {
            "operation": "start-flow",
            "flow-id": "docproc-1",
            "class-name": "everything",
            "description": "Document processing pipeline",
        }
        url = _join_url(base_url, path)
        print(f"Equivalent curl:\n{_as_curl('POST', url, payload_1)}\n")
        resp1 = request_with_retries(base_url, path, payload_1, timeout, stage, "POST", retries, retry_wait)
        if resp1 is None or resp1.status_code != 200:
            return 1

        # 2) Add document
        stage = "2/4 Add document"
        path = "/api/v1/librarian"
        payload_2: Dict[str, Any] = {
            "operation": "add-document",
            "document-metadata": {
                "id": doc_id,
                "time": 1731500000,
                "kind": kind_value,
                "title": title_value,
                "comments": collection_name,
                "user": "alice",
                "tags": [collection_name, "sample"],
                "metadata": [
                    {
                        "s": {"v": "", "e": True},
                        "p": {"v": "", "e": True},
                        "o": {"v": "", "e": False},
                    }
                ],
            },
            "content": content_b64,
        }
        url = _join_url(base_url, path)
        print(f"Equivalent curl:\n{_as_curl('POST', url, payload_2)}\n")
        resp2 = request_with_retries(base_url, path, payload_2, timeout, stage, "POST", retries, retry_wait)
        if resp2 is None or resp2.status_code != 200:
            return 1

        # 3) Add processing
        stage = "3/4 Add processing"
        path = "/api/v1/librarian"
        payload_3: Dict[str, Any] = {
            "operation": "add-processing",
            "processing-metadata": {
                "id": proc_id,
                "document-id": doc_id,
                "time": 1731507200,
                "flow": "docproc-1",
                "user": "alice",
                "collection": collection_name,
                "tags": [collection_name, "sample"],
            },
        }
        url = _join_url(base_url, path)
        print(f"Equivalent curl:\n{_as_curl('POST', url, payload_3)}\n")
        resp3 = request_with_retries(base_url, path, payload_3, timeout, stage, "POST", retries, retry_wait)
        if resp3 is None or resp3.status_code != 200:
            return 1

        # 4) Document retrieval
        stage = "4/4 Document retrieval"
        path = "/api/v1/flow/docproc-1/service/document-retrieval"
        payload_4: Dict[str, Any] = {
            "query": "what do u know about my birthday?",
            "user": "alice",
            "collection": collection_name,
            "doc-limit": 10,
        }
        url = _join_url(base_url, path)
        print(f"Equivalent curl:\n{_as_curl('POST', url, payload_4)}\n")
        resp4 = request_with_retries(base_url, path, payload_4, timeout, stage, "POST", retries, retry_wait)
        if resp4 is None or resp4.status_code != 200:
            return 1

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 130

    return 0


if __name__ == "__main__":
    sys.exit(main())

