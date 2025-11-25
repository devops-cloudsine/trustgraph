
"""
Simple decoder, accepts PDF documents on input, outputs pages from the
PDF document as text as separate output objects.
"""

from pypdf import PdfWriter, PdfReader
from io import BytesIO
import base64
import uuid
import os
import json

from mistralai import Mistral
from mistralai import DocumentURLChunk, ImageURLChunk, TextChunk
from mistralai.models import OCRResponse

from ... schema import Document, TextDocument, Metadata
from ... base import FlowProcessor, ConsumerSpec, ProducerSpec

import logging
import os
import re
import base64
import requests

logger = logging.getLogger(__name__)

default_ident = "pdf-decoder"
default_api_key = os.getenv("MISTRAL_TOKEN")

pages_per_chunk = 5

def chunks(lst, n):
    "Yield successive n-sized chunks from lst."
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def replace_images_in_markdown(markdown_str: str, images_dict: dict) -> str:
    """
    Replace image placeholders in markdown with base64-encoded images.

    Args:
        markdown_str: Markdown text containing image placeholders
        images_dict: Dictionary mapping image IDs to base64 strings

    Returns:
        Markdown text with images replaced by base64 data
    """
    for img_name, base64_str in images_dict.items():
        markdown_str = markdown_str.replace(
            f"![{img_name}]({img_name})", f"![{img_name}]({base64_str})"
        )
    return markdown_str

def get_combined_markdown(ocr_response: OCRResponse) -> str:
    """
    Combine OCR text and images into a single markdown document.

    Args:
        ocr_response: Response from OCR processing containing text and images

    Returns:
        Combined markdown string with embedded images
    """
    markdowns: list[str] = []
    # Extract images from page
    for page in ocr_response.pages:
        image_data = {}
        for img in page.images:
            image_data[img.id] = img.image_base64
        # Replace image placeholders with actual images
        markdowns.append(replace_images_in_markdown(page.markdown, image_data))

    return "\n\n".join(markdowns)

class Processor(FlowProcessor):

    def __init__(self, **params):

        id = params.get("id", default_ident)
        api_key = params.get("api_key", default_api_key)
        # Optional: enable vLLM image descriptions without changing default behavior
        self.use_vllm = params.get("use_vllm", False)
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
                name = "input",
                schema = Document,
                handler = self.on_message,
            )
        )

        self.register_specification(
            ProducerSpec(
                name = "output",
                schema = TextDocument,
            )
        )

        if api_key is None:
            raise RuntimeError("Mistral API key not specified")

        self.mistral = Mistral(api_key=api_key)

        # Used with Mistral doc upload; refreshed per OCR invocation
        self.unique_id = None

        logger.info("Mistral OCR processor initialized")

    # // ---> on_message > [ocr] > Mistral OCR + optional vLLM image descriptions
    def ocr(self, blob):

        logger.debug("Parse PDF...")

        pdfbuf = BytesIO(blob)
        pdf = PdfReader(pdfbuf)

        # Unique per document call
        self.unique_id = str(uuid.uuid4())

        for chunk in chunks(pdf.pages, pages_per_chunk):
            
            logger.debug("Get next pages...")

            part = PdfWriter()
            for page in chunk:
                part.add_page(page)

            buf = BytesIO()
            part.write_stream(buf)

            logger.debug("Upload chunk...")

            uploaded_file = self.mistral.files.upload(
                file={
                    "file_name": self.unique_id,
                    "content": buf.getvalue(),
                },
                purpose="ocr",
            )

            signed_url = self.mistral.files.get_signed_url(
                file_id=uploaded_file.id, expiry=1
            )

            logger.debug("OCR...")

            processed = self.mistral.ocr.process(
                model="mistral-ocr-latest",
                include_image_base64=True,
                document={
                    "type": "document_url",
                    "document_url": signed_url.url,
                }
            )

            logger.debug("Extract markdown...")

            markdown = get_combined_markdown(processed)

            # Optionally save images and append vLLM-generated descriptions
            if self.use_vllm:
                try:
                    save_dir = os.path.join(self.files_base_dir, self._safe_id(self.unique_id))
                    os.makedirs(save_dir, exist_ok=True)
                    desc_md = self._describe_and_store_images(processed, save_dir)
                    if desc_md:
                        markdown = f"{markdown}\n\n{desc_md}"
                except Exception as e:
                    logger.warning("vLLM description step failed: %s", e)

            logger.info("OCR complete.")

            return markdown

    async def on_message(self, msg, consumer, flow):

        logger.debug("PDF message received")

        v = msg.value()

        logger.info(f"Decoding {v.metadata.id}...")

        markdown = self.ocr(base64.b64decode(v.data))

        r = TextDocument(
            metadata=v.metadata,
            text=markdown.encode("utf-8"),
        )

        await flow("output").send(r)

        logger.info("Done.")

    @staticmethod
    def add_args(parser):

        FlowProcessor.add_args(parser)

        parser.add_argument(
            '-k', '--api-key',
            default=default_api_key,
            help=f'Mistral API Key'
        )

    # // ---> ocr > [_safe_id] > sanitize folder names
    def _safe_id(self, value: str) -> str:
        if value is None:
            return "unknown"
        return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))

    # // ---> ocr(use_vllm) > [_describe_and_store_images] > write files and return markdown descriptions
    def _describe_and_store_images(self, ocr_response: OCRResponse, save_dir: str) -> str:
        sections: list[str] = []
        for page_index, page in enumerate(ocr_response.pages):
            lines: list[str] = [f"# Page {page_index + 1} Image Descriptions"]
            for image_index, img in enumerate(page.images):
                try:
                    image_path = self._write_base64_image(img.image_base64, save_dir, page_index, image_index)
                except Exception as write_err:
                    logger.warning("Failed to write image for page %d index %d: %s", page_index + 1, image_index, write_err)
                    lines.append(f"- image_{image_index + 1}: write failed")
                    continue
                desc = self._describe_image_with_vllm(image_path)
                lines.append(f"- {os.path.basename(image_path)}: {desc}")
            if len(lines) > 1:
                sections.append("\n".join(lines))
        return "\n\n".join(sections)

    # // ---> _describe_and_store_images > [_write_base64_image] > save image to /root/files_to_parse
    def _write_base64_image(self, data_url: str, save_dir: str, page_index: int, image_index: int) -> str:
        # Handle data URL like: data:image/png;base64,xxxx
        if isinstance(data_url, str) and data_url.startswith("data:image"):
            try:
                header, b64 = data_url.split(",", 1)
            except ValueError:
                header, b64 = "data:image/png;base64", data_url
            ext = "png"
            if ";base64" in header:
                mime = header.split(";")[0]
                _, mime_type = mime.split(":", 1)
                if "/" in mime_type:
                    ext = mime_type.split("/")[1].lower() or "png"
        else:
            # Assume raw base64 without header
            b64 = data_url
            ext = "png"
        raw = base64.b64decode(b64)
        filename = f"page_{page_index + 1}_image_{image_index + 1}.{ext}"
        path = os.path.join(save_dir, filename)
        with open(path, "wb") as f:
            f.write(raw)
        return path

    # // ---> _describe_and_store_images > [_describe_image_with_vllm] > POST to vLLM, returns text
    def _describe_image_with_vllm(self, image_path: str) -> str:
        try:
            # NOTE: Do NOT include "response_format" parameter for plain text responses.
            # vLLM v0.11.0+ (V1 engine) has a bug where any response_format (even {"type": "text"})
            # triggers structured output validation that fails with:
            # "ValueError: No valid structured output parameter found"
            # Plain text is the default, so omitting response_format works correctly.
            payload = {
                "model": self.vllm_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image."},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"file://{image_path}"},
                            },
                        ],
                    }
                ],
            }
            # Log the exact outgoing request to vLLM
            try:
                logger.info(
                    "vLLM request: POST %s payload=%s",
                    self.vllm_api_url,
                    json.dumps(payload, ensure_ascii=False),
                )
            except Exception as log_err:
                logger.warning("Failed to serialize vLLM payload for logging: %s", log_err)
            resp = requests.post(self.vllm_api_url, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            choice = (data.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            content = msg.get("content")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                texts = [c.get("text", "") for c in content if c.get("type") == "text"]
                return "\n".join([t for t in texts if t]).strip() or "No description returned."
            return "No description returned."
        except Exception as e:
            logger.warning("vLLM request failed for %s: %s", image_path, e)
            return "Description unavailable."

def run():

    Processor.launch(default_ident, __doc__)

