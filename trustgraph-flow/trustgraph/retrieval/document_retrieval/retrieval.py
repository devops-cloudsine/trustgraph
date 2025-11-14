
"""
Document Retrieval service - returns document chunks without LLM generation.
Input is query, output is documents and metadata.
"""

import logging
from ... schema import DocumentRetrievalQuery, DocumentRetrievalResponse, Error
from .. document_rag.document_rag import DocumentRag, Query
from ... base import FlowProcessor, ConsumerSpec, ProducerSpec
from ... base import EmbeddingsClientSpec
from ... base import DocumentEmbeddingsClientSpec

# Module logger
logger = logging.getLogger(__name__)

default_ident = "document-retrieval"

class Processor(FlowProcessor):

    def __init__(self, **params):

        id = params.get("id", default_ident)

        doc_limit = params.get("doc_limit", 20)

        super(Processor, self).__init__(
            **params | {
                "id": id,
                "doc_limit": doc_limit,
            }
        )

        self.default_doc_limit = doc_limit

        self.register_specification(
            ConsumerSpec(
                name = "request",
                schema = DocumentRetrievalQuery,
                handler = self.on_request,
            )
        )

        self.register_specification(
            EmbeddingsClientSpec(
                request_name = "embeddings-request",
                response_name = "embeddings-response",
            )
        )

        self.register_specification(
            DocumentEmbeddingsClientSpec(
                request_name = "document-embeddings-request",
                response_name = "document-embeddings-response",
            )
        )

        self.register_specification(
            ProducerSpec(
                name = "response",
                schema = DocumentRetrievalResponse,
            )
        )

    async def on_request(self, msg, consumer, flow):

        try:

            rag = DocumentRag(
                embeddings_client = flow("embeddings-request"),
                doc_embeddings_client = flow("document-embeddings-request"),
                prompt_client = None,  # Not needed for retrieval only
                verbose=True,
            )

            v = msg.value()

            # Sender-produced ID
            id = msg.properties()["id"]

            logger.info(f"Handling retrieval request {id}...")

            # Get parameters with defaults
            doc_limit = v.doc_limit if v.doc_limit else self.default_doc_limit

            # Create query instance
            q = Query(
                rag = rag,
                user = v.user,
                collection = v.collection,
                verbose = True,
                doc_limit = doc_limit
            )

            # Get documents
            docs = await q.get_docs(v.query)

            # Build metadata
            metadata = {
                "doc_count": str(len(docs)),
                "doc_limit": str(doc_limit),
            }

            await flow("response").send(
                DocumentRetrievalResponse(
                    documents = docs,
                    metadata = metadata,
                    error = None
                ),
                properties = {"id": id}
            )

            logger.info(f"Retrieval complete: {len(docs)} documents")

        except Exception as e:

            logger.error(f"Document retrieval service exception: {e}", exc_info=True)

            logger.debug("Sending error response...")

            await flow("response").send(
                DocumentRetrievalResponse(
                    documents = [],
                    metadata = {},
                    error = Error(
                        type = "document-retrieval-error",
                        message = str(e),
                    ),
                ),
                properties = {"id": id}
            )

    @staticmethod
    def add_args(parser):

        FlowProcessor.add_args(parser)

        parser.add_argument(
            '-d', '--doc-limit',
            type=int,
            default=20,
            help=f'Default document fetch limit (default: 20)'
        )

def run():

    Processor.launch(default_ident, __doc__)


