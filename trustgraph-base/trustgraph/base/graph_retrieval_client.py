
from . request_response_spec import RequestResponse, RequestResponseSpec
from .. schema import GraphRetrievalQuery, GraphRetrievalResponse

class GraphRetrievalClient(RequestResponse):
    async def retrieve(self, query, user="trustgraph", collection="default",
                      entity_limit=None, triple_limit=None,
                      max_subgraph_size=None, max_path_length=None,
                      timeout=600):
        resp = await self.request(
            GraphRetrievalQuery(
                query = query,
                user = user,
                collection = collection,
                entity_limit = entity_limit,
                triple_limit = triple_limit,
                max_subgraph_size = max_subgraph_size,
                max_path_length = max_path_length,
            ),
            timeout=timeout
        )

        if resp.error:
            raise RuntimeError(resp.error.message)

        return resp.triples, resp.metadata

class GraphRetrievalClientSpec(RequestResponseSpec):
    def __init__(
            self, request_name, response_name,
    ):
        super(GraphRetrievalClientSpec, self).__init__(
            request_name = request_name,
            request_schema = GraphRetrievalQuery,
            response_name = response_name,
            response_schema = GraphRetrievalResponse,
            impl = GraphRetrievalClient,
        )


