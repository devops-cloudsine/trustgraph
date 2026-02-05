"""
Image decoder: Accepts image documents on input, uses vLLM vision model 
to generate a text description of the image as output.
Replaces OCR-based image processing with vision model descriptions.
Images are stored in MinIO for persistence.
"""

import base64
import logging
import os
import re
import imghdr
import json
import urllib.parse
from pathlib import Path
import requests

from ... schema import Document, TextDocument
from ... schema import Triples, Triple, Value, Metadata
from ... schema import EntityContext, EntityContexts
from ... base import FlowProcessor, ConsumerSpec, ProducerSpec
from .minio_storage import MinioStorage, get_minio_storage

# ---> rdf.py constants for image context triples
IMAGE_CONTEXT = "http://trustgraph.ai/ns/image-context"
IMAGE_SOURCE = "http://trustgraph.ai/ns/image-source"
TRUSTGRAPH_ENTITIES = "http://trustgraph.ai/e/"

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
        
        # Initialize MinIO storage for image persistence
        self.minio_storage = get_minio_storage()

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

        # ---> on_message > [triples output] > emit IMAGE_CONTEXT/IMAGE_SOURCE triples for Graph RAG
        self.register_specification(
            ProducerSpec(
                name="triples",
                schema=Triples,
            )
        )

        # ---> on_message > [entity-contexts output] > emit EntityContexts for graph embeddings (enables graph-retrieval)
        self.register_specification(
            ProducerSpec(
                name="entity-contexts",
                schema=EntityContexts,
            )
        )

        logger.info("Image decoder initialized (using vLLM for image descriptions)")

    # // ---> Pulsar consumer(input) > [on_message] > save to MinIO, describe image via vLLM -> flow('output')
    async def on_message(self, msg, consumer, flow):

        logger.info("Image message received for processing")

        v = msg.value()
        doc_id = v.metadata.id

        logger.info(f"Processing {doc_id}...")

        blob = base64.b64decode(v.data)

        # Check if this is an image file
        if not self._is_image(blob):
            logger.info(f"Skipping non-image file: {doc_id}")
            return

        # Initialize MinIO storage
        if not self.minio_storage.initialize():
            logger.error("Failed to initialize MinIO storage - cannot process image")
            return

        # Detect image format
        image_kind = imghdr.what(None, blob)
        content_type = IMAGE_KIND_TO_CONTENT_TYPE.get(image_kind, "image/png")
        ext = self._get_extension_for_type(content_type)
        # Use safe doc_id as filename for the original image document
        image_filename = f"{self._safe_id(doc_id)}{ext}"

        # Save original image document to MinIO
        try:
            object_name = self.minio_storage.save_document(
                doc_id=doc_id,
                filename=image_filename,
                document_data=blob,
                content_type=content_type
            )
            if object_name:
                local_path = self.minio_storage.get_local_path(object_name)
                logger.info(f"Saved original image to MinIO: {object_name} (local: {local_path})")
            else:
                logger.warning("Failed saving original image to MinIO")
                return
        except Exception as e:
            logger.warning(f"Failed saving image: {e}")
            return

        # Describe image via vLLM using the image bytes directly
        description = self._describe_image_with_vllm_bytes(blob, content_type)
        image_text = f"Image Description:\n{description}\n"

        r = TextDocument(
            metadata=v.metadata,
            text=image_text.encode("utf-8"),
        )

        await flow("output").send(r)

        # ---> Create IMAGE_CONTEXT and IMAGE_SOURCE triples for Graph RAG
        if description and description.strip() and not description.startswith("Description unavailable"):
            doc_uri = TRUSTGRAPH_ENTITIES + urllib.parse.quote(doc_id)
            image_uri = f"{doc_uri}/image"
            image_uri_value = Value(value=image_uri, is_uri=True)
            
            image_triples = []
            entity_contexts = []  # For graph embeddings
            
            # Add IMAGE_CONTEXT triple
            image_triples.append(Triple(
                s=image_uri_value,
                p=Value(value=IMAGE_CONTEXT, is_uri=True),
                o=Value(value=description, is_uri=False),
            ))
            
            # Add IMAGE_SOURCE triple
            if object_name:
                image_path = f"minio://{object_name}"
                image_triples.append(Triple(
                    s=image_uri_value,
                    p=Value(value=IMAGE_SOURCE, is_uri=True),
                    o=Value(value=image_path, is_uri=False),
                ))
            
            # ---> Create EntityContext for graph embeddings (enables graph-retrieval to find this entity)
            entity_contexts.append(EntityContext(
                entity=image_uri_value,
                context=description,
            ))
            
            # ---> on_message > [triples flow] > emit IMAGE_CONTEXT/IMAGE_SOURCE triples for Graph RAG
            triples_msg = Triples(
                metadata=Metadata(
                    id=doc_id,
                    metadata=[],
                    user=v.metadata.user if hasattr(v.metadata, 'user') and v.metadata.user else "trustgraph",
                    collection=v.metadata.collection if hasattr(v.metadata, 'collection') and v.metadata.collection else "default",
                ),
                triples=image_triples,
            )
            await flow("triples").send(triples_msg)
            logger.info(f"Emitted {len(image_triples)} image context triples for image {doc_id}")

            # ---> on_message > [entity-contexts flow] > emit EntityContexts for graph embeddings (enables graph-retrieval)
            entity_contexts_msg = EntityContexts(
                metadata=Metadata(
                    id=doc_id,
                    metadata=[],
                    user=v.metadata.user if hasattr(v.metadata, 'user') and v.metadata.user else "trustgraph",
                    collection=v.metadata.collection if hasattr(v.metadata, 'collection') and v.metadata.collection else "default",
                ),
                entities=entity_contexts,
            )
            await flow("entity-contexts").send(entity_contexts_msg)
            logger.info(f"Emitted {len(entity_contexts)} entity contexts for graph embeddings for image {doc_id}")

        logger.info("Image description complete")

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            '--vllm-api-url',
            default='http://vllm-vision-server:8000/v1/chat/completions',
            help='vLLM API URL for image descriptions (default: http://vllm:8000/v1/chat/completions)'
        )
        parser.add_argument(
            '--vllm-model',
            default='google/gemma-3-4b-it',
            help='vLLM model name (default: Qwen/Qwen3-VL-4B-Instruct)'
        )
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

    # // ---> on_message > [_bytes_to_base64_data_url] > convert image bytes to data:image/... URL
    def _bytes_to_base64_data_url(self, image_bytes: bytes, content_type: str = "image/png") -> str:
        """
        Convert image bytes to a base64 data URL.
        """
        img_data = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:{content_type};base64,{img_data}"

    # // ---> on_message > [_describe_image_with_vllm_bytes] > POST to vLLM with bytes, returns description text
    def _describe_image_with_vllm_bytes(self, image_bytes: bytes, content_type: str = "image/png") -> str:
        """
        Describe an image using vLLM by sending image bytes as base64.
        
        Args:
            image_bytes: Raw image bytes
            content_type: MIME type of the image
            
        Returns:
            Text description of the image
        """
        try:
            # Convert image bytes to base64 data URL
            logger.debug(f"Converting image bytes to base64 ({len(image_bytes)} bytes)")
            try:
                image_url = self._bytes_to_base64_data_url(image_bytes, content_type)
                logger.debug(f"Base64 image URL created, length: {len(image_url)} chars")
            except Exception as e:
                logger.error(f"Failed to encode image to base64: {e}")
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
                logger.info(f"vLLM image description: {content.strip()}")
                return content.strip()
            # Some implementations may return list content parts
            if isinstance(content, list):
                texts = [c.get("text", "") for c in content if c.get("type") == "text"]
                return "\n".join([t for t in texts if t]).strip() or "No description returned."
            return "No description returned."
        except requests.exceptions.Timeout:
            logger.error("vLLM request timed out")
            return "Description unavailable (timeout)."
        except requests.exceptions.ConnectionError as e:
            logger.error(f"vLLM connection error: {e}")
            return "Description unavailable (connection error)."
        except Exception as e:
            logger.error(f"vLLM request failed: {type(e).__name__}: {e}")
            return "Description unavailable."


def run():

    Processor.launch(default_ident, __doc__)

