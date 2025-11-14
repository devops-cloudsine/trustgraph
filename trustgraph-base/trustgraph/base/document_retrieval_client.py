
from . request_response_spec import RequestResponse, RequestResponseSpec
from .. schema import DocumentRetrievalQuery, DocumentRetrievalResponse

class DocumentRetrievalClient(RequestResponse):
    async def retrieve(self, query, user="trustgraph", collection="default",
                      doc_limit=None, timeout=600):
        resp = await self.request(
            DocumentRetrievalQuery(
                query = query,
                user = user,
                collection = collection,
                doc_limit = doc_limit,
            ),
            timeout=timeout
        )

        if resp.error:
            raise RuntimeError(resp.error.message)

        return resp.documents, resp.metadata

class DocumentRetrievalClientSpec(RequestResponseSpec):
    def __init__(
            self, request_name, response_name,
    ):
        super(DocumentRetrievalClientSpec, self).__init__(
            request_name = request_name,
            request_schema = DocumentRetrievalQuery,
            response_name = response_name,
            response_schema = DocumentRetrievalResponse,
            impl = DocumentRetrievalClient,
        )


