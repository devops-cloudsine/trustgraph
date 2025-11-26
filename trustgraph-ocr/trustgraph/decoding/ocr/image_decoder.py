"""
Image decoder: Accepts image documents on input, uses vLLM vision model 
to generate a text description of the image as output.
Replaces OCR-based image processing with vision model descriptions.
"""

import base64
import logging
import os
import re
import imghdr
import json
from pathlib import Path
import requests

from ... schema import Document, TextDocument
from ... base import FlowProcessor, ConsumerSpec, ProducerSpec

# Module logger
logger = logging.getLogger(__name__)

# Enable DEBUG level for this module when VLLM_LOGGING_LEVEL=DEBUG is set
if os.environ.get("VLLM_LOGGING_LEVEL", "").upper() == "DEBUG":
    logger.setLevel(logging.DEBUG)

default_ident = "image-decoder"

# Supported image content types
IMAGE_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
    "image/bmp",
    "image/tiff",
}

# imghdr kind to MIME mapping
IMAGE_KIND_TO_CONTENT_TYPE = {
    "jpeg": "image/jpeg",
    "png": "image/png",
    "gif": "image/gif",
    "webp": "image/webp",
    "bmp": "image/bmp",
    "tiff": "image/tiff",
}


class Processor(FlowProcessor):

    def __init__(self, **params):

        id = params.get("id", default_ident)
        self.vllm_api_url = params.get(
            "vllm_api_url",
            "http://vllm:8000/v1/chat/completions"
        )
        self.vllm_model = params.get(
            "vllm_model",
            "Qwen/Qwen3-VL-4B-Instruct"
        )
        self.files_base_dir = params.get(
            "files_base_dir",
            "/root/files_to_parse"
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

        logger.info("Image decoder initialized (using vLLM for image descriptions)")

    # // ---> Pulsar consumer(input) > [on_message] > describe image via vLLM -> flow('output')
    async def on_message(self, msg, consumer, flow):

        logger.info("Image message received for processing")

        v = msg.value()

        logger.info(f"Processing {v.metadata.id}...")

        blob = base64.b64decode(v.data)

        # Check if this is an image file
        if not self._is_image(blob):
            logger.info(f"Skipping non-image file: {v.metadata.id}")
            return

        # Detect image format
        image_kind = imghdr.what(None, blob)
        content_type = IMAGE_KIND_TO_CONTENT_TYPE.get(image_kind, "image/png")
        ext = self._get_extension_for_type(content_type)

        # Prepare output directory for this document
        doc_dir = os.path.join(self.files_base_dir, self._safe_id(v.metadata.id))
        os.makedirs(doc_dir, exist_ok=True)

        image_path = os.path.join(doc_dir, f"image{ext}")
        try:
            # Save image to file
            with open(image_path, "wb") as f:
                f.write(blob)
            logger.debug(f"Saved image: {image_path}")
        except Exception as e:
            logger.warning(f"Failed saving image: {e}")
            return

        # Describe image via vLLM
        description = self._describe_image_with_vllm(image_path, content_type)
        image_text = f"Image Description:\n{description}\n"

        r = TextDocument(
            metadata=v.metadata,
            text=image_text.encode("utf-8"),
        )

        await flow("output").send(r)

        logger.info("Image description complete")

    @staticmethod
    def add_args(parser):
        FlowProcessor.add_args(parser)

    # // ---> on_message > [_is_image] > check if blob is an image file
    def _is_image(self, blob: bytes) -> bool:
        """Check if the blob is an image file by examining magic bytes."""
        image_kind = imghdr.what(None, blob)
        return image_kind is not None

    # // ---> on_message > [_get_extension_for_type] > get file extension for content type
    def _get_extension_for_type(self, content_type: str) -> str:
        """Get file extension for a given content type."""
        ext_map = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/gif": ".gif",
            "image/webp": ".webp",
            "image/bmp": ".bmp",
            "image/tiff": ".tiff",
        }
        return ext_map.get(content_type, ".png")

    # // ---> on_message > [_safe_id] > sanitize directory name
    def _safe_id(self, value: str) -> str:
        if value is None:
            return "unknown"
        return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))

    # // ---> on_message > [_image_bytes_to_base64_data_url] > encode image bytes as data URL
    def _image_bytes_to_base64_data_url(self, image_bytes: bytes, content_type: str) -> str:
        """
        Convert image bytes to a base64 data URL.
        Required because vLLM runs in a separate container.
        """
        img_data = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:{content_type};base64,{img_data}"

    # // ---> on_message > [_image_file_to_base64_data_url] > reads image file, returns data:image/... URL
    def _image_file_to_base64_data_url(self, image_path: str) -> str:
        """
        Convert an image file to a base64 data URL.
        This is required because vLLM runs in a separate container and cannot
        access file:// paths from this container's filesystem.
        """
        with open(image_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode("utf-8")

        # Detect MIME type from extension
        ext = Path(image_path).suffix.lower()
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
            ".tiff": "image/tiff",
        }
        mime_type = mime_map.get(ext, "image/png")

        return f"data:{mime_type};base64,{img_data}"

    # // ---> on_message > [_describe_image_with_vllm] > POST to vLLM, returns description text
    def _describe_image_with_vllm(self, image_path: str, content_type: str = None) -> str:
        try:
            # Convert image to base64 data URL instead of file:// path.
            # CRITICAL: vLLM runs in a separate container and cannot access
            # file:// paths from this container's filesystem.
            logger.debug(f"Converting image to base64: {image_path}")
            try:
                image_url = self._image_file_to_base64_data_url(image_path)
                logger.debug(f"Base64 image URL created, length: {len(image_url)} chars")
            except Exception as e:
                logger.error(f"Failed to encode image {image_path} to base64: {e}")
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
                logger.info(
                    "vLLM request: POST %s payload=%s",
                    self.vllm_api_url,
                    json.dumps(payload_log, ensure_ascii=False),
                )
            except Exception as log_err:
                logger.warning(f"Failed to serialize vLLM payload for logging: {log_err}")

            logger.debug(f"Sending request to vLLM at {self.vllm_api_url}")
            resp = requests.post(self.vllm_api_url, json=payload, timeout=120)

            logger.debug(f"vLLM response status: {resp.status_code}")

            if resp.status_code != 200:
                logger.error(
                    f"vLLM returned HTTP {resp.status_code}: {resp.text[:500]}"
                )
                return f"Description unavailable (HTTP {resp.status_code})."

            resp.raise_for_status()
            data = resp.json()

            logger.debug(f"vLLM response JSON keys: {list(data.keys())}")

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
            logger.error(f"vLLM request timed out for {image_path}")
            return "Description unavailable (timeout)."
        except requests.exceptions.ConnectionError as e:
            logger.error(f"vLLM connection error for {image_path}: {e}")
            return "Description unavailable (connection error)."
        except Exception as e:
            logger.error(f"vLLM request failed for {image_path}: {type(e).__name__}: {e}")
            return "Description unavailable."


def run():

    Processor.launch(default_ident, __doc__)

