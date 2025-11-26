"""
DOCX/DOC decoder: Extracts text and images from DOCX and DOC documents using Spire.Doc.
Images are sent to vLLM for description and output as text segments.
"""

import base64
import logging
import os
import re
from pathlib import Path
import requests
import json

from spire.doc import Document, DocPicture

from ... schema import Document as TGDocument, TextDocument
from ... base import FlowProcessor, ConsumerSpec, ProducerSpec

# Module logger
logger = logging.getLogger(__name__)

# Enable DEBUG level for this module when VLLM_LOGGING_LEVEL=DEBUG is set
if os.environ.get("VLLM_LOGGING_LEVEL", "").upper() == "DEBUG":
	logger.setLevel(logging.DEBUG)

default_ident = "docx-decoder"


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
				name = "input",
				schema = TGDocument,
				handler = self.on_message,
			)
		)

		self.register_specification(
			ProducerSpec(
				name = "output",
				schema = TextDocument,
			)
		)

		logger.info("DOCX/DOC decoder initialized (using Spire.Doc + vLLM for image descriptions)")

	# // ---> Pulsar consumer(input) > [on_message] > extract text/images, vLLM describe -> flow('output')
	async def on_message(self, msg, consumer, flow):

		logger.info("DOCX/DOC message received for processing")

		v = msg.value()

		logger.info(f"Processing {v.metadata.id}...")

		blob = base64.b64decode(v.data)

		# Check if this is a DOCX or DOC file
		if not self._is_docx_or_doc(blob):
			logger.info(f"Skipping non-DOCX/DOC file: {v.metadata.id}")
			return

		# Prepare output directory for this document
		doc_dir = os.path.join(self.files_base_dir, self._safe_id(v.metadata.id))
		os.makedirs(doc_dir, exist_ok=True)

		# Determine file extension based on format
		# DOCX files start with PK (ZIP), DOC files start with OLE signature
		file_ext = ".docx" if blob[:2] == b'PK' else ".doc"
		temp_doc_path = os.path.join(doc_dir, f"source{file_ext}")
		
		# Write blob to a file for Spire.Doc to load
		try:
			with open(temp_doc_path, "wb") as f:
				f.write(blob)
			logger.debug(f"Saved document to file: {temp_doc_path}")
		except Exception as e:
			logger.error(f"Failed to save document to file: {e}")
			return

		# Load DOCX/DOC using Spire.Doc
		try:
			document = Document()
			document.LoadFromFile(temp_doc_path)
		except Exception as e:
			logger.error(f"Failed to load document with Spire.Doc: {e}")
			return

		# Extract and output text content first
		text_segments = self._extract_text_segments(document)
		for ix, text in enumerate(text_segments):
			if text.strip():
				r = TextDocument(
					metadata=v.metadata,
					text=text.encode("utf-8"),
				)
				await flow("output").send(r)

		# Extract embedded images using Spire.Doc
		images = self._extract_images(document)
		logger.info(f"Found {len(images)} embedded images in document")

		for ix, image_bytes in enumerate(images):
			image_filename = f"image_{ix + 1}.png"
			image_path = os.path.join(doc_dir, image_filename)

			try:
				# Save embedded image
				with open(image_path, "wb") as f:
					f.write(image_bytes)
				logger.debug(f"Saved embedded image: {image_path}")
			except Exception as e:
				logger.warning(f"Failed saving embedded image {ix + 1}: {e}")
				continue

			# Describe image via vLLM
			description = self._describe_image_with_vllm(image_path)
			image_text = f"Embedded Image {ix + 1} Description:\n{description}\n"

			r = TextDocument(
				metadata=v.metadata,
				text=image_text.encode("utf-8"),
			)

			await flow("output").send(r)

		# Close the Spire.Doc document
		try:
			document.Close()
		except Exception as e:
			logger.warning(f"Failed to close Spire.Doc document: {e}")

		logger.info("DOCX/DOC text and image descriptions complete")

	@staticmethod
	def add_args(parser):
		FlowProcessor.add_args(parser)

	# // ---> on_message > [_is_docx_or_doc] > check if blob is a DOCX or DOC file
	def _is_docx_or_doc(self, blob: bytes) -> bool:
		"""
		Check if the blob is a DOCX or DOC file by examining magic bytes.
		DOCX files are ZIP archives (PK signature) containing Office Open XML.
		DOC files are OLE compound documents.
		"""
		import zipfile
		from io import BytesIO
		
		# Check for DOCX (ZIP-based Office Open XML)
		if blob[:2] == b'PK':
			try:
				with zipfile.ZipFile(BytesIO(blob), 'r') as zf:
					names = zf.namelist()
					# DOCX files contain word/document.xml
					if any(name.startswith('word/') for name in names):
						return True
			except Exception:
				pass
		
		# Check for DOC (OLE compound document)
		# OLE signature: D0 CF 11 E0 A1 B1 1A E1
		OLE_SIGNATURE = b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1"
		if blob.startswith(OLE_SIGNATURE):
			# Check for Word document marker in the blob
			lowered = blob.lower()
			if b"worddocument" in lowered:
				return True
		
		return False

	# // ---> on_message > [_safe_id] > sanitize directory name
	def _safe_id(self, value: str) -> str:
		if value is None:
			return "unknown"
		return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))

	# // ---> on_message > [_extract_text_segments] > extract text paragraphs from DOCX using Spire.Doc
	def _extract_text_segments(self, document: Document) -> list:
		"""
		Extract all text paragraphs from the DOCX document using Spire.Doc.
		"""
		segments = []
		
		# Iterate over sections
		for s in range(document.Sections.Count):
			section = document.Sections[s]
			
			# Iterate over paragraphs
			for p in range(section.Paragraphs.Count):
				paragraph = section.Paragraphs[p]
				text = paragraph.Text.strip() if paragraph.Text else ""
				if text:
					segments.append(text)
			
			# Also extract from tables
			for t in range(section.Tables.Count):
				table = section.Tables[t]
				for r in range(table.Rows.Count):
					row = table.Rows[r]
					row_texts = []
					for c in range(row.Cells.Count):
						cell = row.Cells[c]
						# Get text from all paragraphs in the cell
						cell_text_parts = []
						for cp in range(cell.Paragraphs.Count):
							cell_para = cell.Paragraphs[cp]
							cell_para_text = cell_para.Text.strip() if cell_para.Text else ""
							if cell_para_text:
								cell_text_parts.append(cell_para_text)
						if cell_text_parts:
							row_texts.append(" ".join(cell_text_parts))
					if row_texts:
						segments.append(" | ".join(row_texts))
		
		return segments

	# // ---> on_message > [_extract_images] > extract embedded images from DOCX using Spire.Doc
	def _extract_images(self, document: Document) -> list:
		"""
		Extract all embedded images from the DOCX document using Spire.Doc.
		Returns a list of image byte data.
		"""
		images = []
		
		# Iterate over sections
		for s in range(document.Sections.Count):
			section = document.Sections[s]
			
			# Iterate over paragraphs
			for p in range(section.Paragraphs.Count):
				paragraph = section.Paragraphs[p]
				
				# Iterate over child objects
				for c in range(paragraph.ChildObjects.Count):
					obj = paragraph.ChildObjects[c]
					# Extract image data
					if isinstance(obj, DocPicture):
						try:
							picture = obj
							# Get image bytes
							dataBytes = picture.ImageBytes
							if dataBytes:
								images.append(dataBytes)
						except Exception as e:
							logger.warning(f"Failed to extract image from paragraph: {e}")
							continue
			
			# Also check tables for images
			for t in range(section.Tables.Count):
				table = section.Tables[t]
				for r in range(table.Rows.Count):
					row = table.Rows[r]
					for cell_idx in range(row.Cells.Count):
						cell = row.Cells[cell_idx]
						for cp in range(cell.Paragraphs.Count):
							cell_para = cell.Paragraphs[cp]
							for co in range(cell_para.ChildObjects.Count):
								obj = cell_para.ChildObjects[co]
								if isinstance(obj, DocPicture):
									try:
										picture = obj
										dataBytes = picture.ImageBytes
										if dataBytes:
											images.append(dataBytes)
									except Exception as e:
										logger.warning(f"Failed to extract image from table cell: {e}")
										continue
		
		return images

	# // ---> on_message > [_image_to_base64_data_url] > reads image file, returns data:image/... URL
	def _image_to_base64_data_url(self, image_path: str) -> str:
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
	def _describe_image_with_vllm(self, image_path: str) -> str:
		try:
			# Convert image to base64 data URL instead of file:// path.
			# CRITICAL: vLLM runs in a separate container and cannot access
			# file:// paths from this container's filesystem. Using file:// causes
			# vLLM to fail with "No valid structured output parameter found" because
			# the malformed request triggers an incorrect code path in vLLM v1.
			logger.debug(f"Converting image to base64: {image_path}")
			try:
				image_url = self._image_to_base64_data_url(image_path)
				logger.debug(f"Base64 image URL created, length: {len(image_url)} chars")
			except Exception as e:
				logger.error(f"Failed to encode image {image_path} to base64: {e}")
				return "Description unavailable (image encoding failed)."

			# NOTE: Do NOT include "response_format" parameter for plain text responses.
			# vLLM v1 has a bug where any response_format (even {"type": "text"}) triggers
			# structured output validation that fails with "No valid structured output parameter found"
			payload = {
				"model": self.vllm_model,
				"messages": [
					{
						"role": "user",
						"content": [
							{"type": "text", "text": "Describe this image."},
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


