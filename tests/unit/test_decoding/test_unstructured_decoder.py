"""
Unit tests for the unstructured decoder.
"""

import base64
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch

from trustgraph.decoding.unstructured.unstructured_decoder import Processor
from trustgraph.schema import Document, Metadata


class TestUnstructuredDecoderProcessor(IsolatedAsyncioTestCase):
    """Validate unstructured decoder behaviour."""

    @patch("trustgraph.decoding.unstructured.unstructured_decoder.partition")
    @patch("trustgraph.base.flow_processor.FlowProcessor.__init__")
    async def test_processor_initialization(self, mock_flow_init, mock_partition):
        mock_flow_init.return_value = None
        mock_partition.return_value = []

        config = {
            "id": "test-unstructured-decoder",
            "taskgroup": AsyncMock(),
        }

        with patch.object(Processor, "register_specification") as mock_register:
            Processor(**config)

        mock_flow_init.assert_called_once()
        assert mock_register.call_count == 2

    @patch("trustgraph.decoding.unstructured.unstructured_decoder.partition")
    @patch("trustgraph.base.flow_processor.FlowProcessor.__init__")
    async def test_on_message_success(self, mock_flow_init, mock_partition):
        mock_flow_init.return_value = None
        mock_partition.return_value = [
            SimpleNamespace(text="Segment 1 "),
            SimpleNamespace(text="Second segment"),
        ]

        config = {
            "id": "test-unstructured-decoder",
            "taskgroup": AsyncMock(),
        }

        with patch.object(Processor, "register_specification"):
            processor = Processor(**config)

        payload = base64.b64encode(b"<html>Hello</html>").decode("utf-8")
        metadata = Metadata(id="doc-1")
        document = Document(
            metadata=metadata,
            data=payload,
            content_type="text/html",
            filename="doc.html",
        )
        msg = MagicMock()
        msg.value.return_value = document

        mock_output_flow = AsyncMock()
        flow = MagicMock(return_value=mock_output_flow)

        await processor.on_message(msg, None, flow)

        assert mock_partition.call_count == 1
        call_kwargs = mock_partition.call_args.kwargs
        assert call_kwargs["metadata_filename"] == "doc.html"

        assert mock_output_flow.send.await_count == 2
        texts = [
            call.args[0].text.decode("utf-8")
            for call in mock_output_flow.send.await_args_list
        ]
        assert texts == ["Segment 1", "Second segment"]

    @patch("trustgraph.decoding.unstructured.unstructured_decoder.partition")
    @patch("trustgraph.base.flow_processor.FlowProcessor.__init__")
    async def test_on_message_fallback(self, mock_flow_init, mock_partition):
        mock_flow_init.return_value = None
        mock_partition.side_effect = RuntimeError("boom")

        config = {
            "id": "test-unstructured-decoder",
            "taskgroup": AsyncMock(),
        }

        with patch.object(Processor, "register_specification"):
            processor = Processor(**config)

        payload = base64.b64encode(b"Plain text content").decode("utf-8")
        metadata = Metadata(id="doc-2")
        document = Document(
            metadata=metadata,
            data=payload,
            content_type=None,
            filename=None,
        )
        msg = MagicMock()
        msg.value.return_value = document

        mock_output_flow = AsyncMock()
        flow = MagicMock(return_value=mock_output_flow)

        await processor.on_message(msg, None, flow)

        mock_output_flow.send.assert_awaited()
        sent_doc = mock_output_flow.send.await_args.args[0]
        assert sent_doc.text.decode("utf-8") == "Plain text content"

