
"""
Graph Retrieval service - returns knowledge graph triples without LLM generation.
Input is query, output is triples and metadata.
"""

import logging
from ... schema import GraphRetrievalQuery, GraphRetrievalResponse, Error
from .. graph_rag.graph_rag import GraphRag, Query
from ... base import FlowProcessor, ConsumerSpec, ProducerSpec
from ... base import EmbeddingsClientSpec
from ... base import GraphEmbeddingsClientSpec, TriplesClientSpec

# Module logger
logger = logging.getLogger(__name__)

default_ident = "graph-retrieval"
default_concurrency = 1

class Processor(FlowProcessor):

    def __init__(self, **params):

        id = params.get("id", default_ident)
        concurrency = params.get("concurrency", 1)

        entity_limit = params.get("entity_limit", 50)
        triple_limit = params.get("triple_limit", 30)
        max_subgraph_size = params.get("max_subgraph_size", 1000)
        max_path_length = params.get("max_path_length", 2)

        super(Processor, self).__init__(
            **params | {
                "id": id,
                "concurrency": concurrency,
                "entity_limit": entity_limit,
                "triple_limit": triple_limit,
                "max_subgraph_size": max_subgraph_size,
                "max_path_length": max_path_length,
            }
        )

        self.default_entity_limit = entity_limit
        self.default_triple_limit = triple_limit
        self.default_max_subgraph_size = max_subgraph_size
        self.default_max_path_length = max_path_length

        # CRITICAL SECURITY: NEVER share data between users or collections
        # Each user/collection combination MUST have isolated data access

        self.register_specification(
            ConsumerSpec(
                name = "request",
                schema = GraphRetrievalQuery,
                handler = self.on_request,
                concurrency = concurrency,
            )
        )

        self.register_specification(
            EmbeddingsClientSpec(
                request_name = "embeddings-request",
                response_name = "embeddings-response",
            )
        )

        self.register_specification(
            GraphEmbeddingsClientSpec(
                request_name = "graph-embeddings-request",
                response_name = "graph-embeddings-response",
            )
        )

        self.register_specification(
            TriplesClientSpec(
                request_name = "triples-request",
                response_name = "triples-response",
            )
        )

        self.register_specification(
            ProducerSpec(
                name = "response",
                schema = GraphRetrievalResponse,
            )
        )

    async def on_request(self, msg, consumer, flow):

        try:

            # CRITICAL SECURITY: Create new GraphRag instance per request
            # This ensures proper isolation between users and collections
            rag = GraphRag(
                embeddings_client=flow("embeddings-request"),
                graph_embeddings_client=flow("graph-embeddings-request"),
                triples_client=flow("triples-request"),
                prompt_client=None,  # Not needed for retrieval only
                verbose=True,
            )

            v = msg.value()

            # Sender-produced ID
            id = msg.properties()["id"]
         
            logger.info(f"Handling retrieval request {id}...")

            # Get parameters with defaults
            entity_limit = v.entity_limit if v.entity_limit else self.default_entity_limit
            triple_limit = v.triple_limit if v.triple_limit else self.default_triple_limit
            max_subgraph_size = v.max_subgraph_size if v.max_subgraph_size else self.default_max_subgraph_size
            max_path_length = v.max_path_length if v.max_path_length else self.default_max_path_length

            # ---> on_request > [Query] > get_labelgraph_with_images() for triples + image contexts
            # Create query instance
            q = Query(
                rag = rag,
                user = v.user,
                collection = v.collection,
                verbose = True,
                entity_limit = entity_limit,
                triple_limit = triple_limit,
                max_subgraph_size = max_subgraph_size,
                max_path_length = max_path_length,
            )

            # Get entities for metadata
            entities = await q.get_entities(v.query)
            
            # Get labeled graph with image contexts (triples + image data)
            triples, image_contexts, image_sources = await q.get_labelgraph_with_images(v.query)

            # Format triples as list of dicts
            triples_list = [
                {"s": str(t[0]), "p": str(t[1]), "o": str(t[2])}
                for t in triples
            ]

            # Build metadata
            metadata = {
                "entity_count": str(len(entities)),
                "triple_count": str(len(triples)),
                "max_subgraph_size": str(max_subgraph_size),
                "max_path_length": str(max_path_length),
                "image_context_count": str(len(image_contexts)),
                "image_source_count": str(len(image_sources)),
            }

            await flow("response").send(
                GraphRetrievalResponse(
                    triples = triples_list,
                    metadata = metadata,
                    image_contexts = image_contexts,
                    image_sources = image_sources,
                    error = None
                ),
                properties = {"id": id}
            )

            logger.info(f"Retrieval complete: {len(triples)} triples from {len(entities)} entities, "
                       f"{len(image_contexts)} image contexts")

        except Exception as e:

            logger.error(f"Graph retrieval service exception: {e}", exc_info=True)

            logger.debug("Sending error response...")

            await flow("response").send(
                GraphRetrievalResponse(
                    triples = [],
                    metadata = {},
                    image_contexts = {},
                    image_sources = {},
                    error = Error(
                        type = "graph-retrieval-error",
                        message = str(e),
                    ),
                ),
                properties = {"id": id}
            )

    @staticmethod
    def add_args(parser):

        parser.add_argument(
            '-c', '--concurrency',
            type=int,
            default=default_concurrency,
            help=f'Concurrent processing threads (default: {default_concurrency})'
        )

        FlowProcessor.add_args(parser)

        parser.add_argument(
            '-e', '--entity-limit',
            type=int,
            default=50,
            help=f'Default entity vector fetch limit (default: 50)'
        )

        parser.add_argument(
            '-t', '--triple-limit',
            type=int,
            default=30,
            help=f'Default triple query limit, per query (default: 30)'
        )

        parser.add_argument(
            '-u', '--max-subgraph-size',
            type=int,
            default=1000,
            help=f'Default max subgraph size (default: 1000)'
        )

        parser.add_argument(
            '-a', '--max-path-length',
            type=int,
            default=2,
            help=f'Default max path length (default: 2)'
        )

def run():

    Processor.launch(default_ident, __doc__)


