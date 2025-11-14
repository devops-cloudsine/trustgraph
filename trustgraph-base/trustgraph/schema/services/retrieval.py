
from pulsar.schema import Record, Bytes, String, Boolean, Integer, Array, Double, Map
from ..core.topic import topic
from ..core.primitives import Error, Value

############################################################################

# Graph RAG text retrieval

class GraphRagQuery(Record):
    query = String()
    user = String()
    collection = String()
    entity_limit = Integer()
    triple_limit = Integer()
    max_subgraph_size = Integer()
    max_path_length = Integer()

class GraphRagResponse(Record):
    error = Error()
    response = String()

############################################################################

# Document RAG text retrieval

class DocumentRagQuery(Record):
    query = String()
    user = String()
    collection = String()
    doc_limit = Integer()

class DocumentRagResponse(Record):
    error = Error()
    response = String()

############################################################################

# Graph Retrieval (without LLM) - returns raw triples and metadata

class GraphRetrievalQuery(Record):
    query = String()
    user = String()
    collection = String()
    entity_limit = Integer()
    triple_limit = Integer()
    max_subgraph_size = Integer()
    max_path_length = Integer()

class GraphRetrievalResponse(Record):
    error = Error()
    triples = Array(Map(String()))  # List of {s, p, o} dicts
    metadata = Map(String())  # {entity_count, triple_count, etc.}

############################################################################

# Document Retrieval (without LLM) - returns raw document chunks and metadata

class DocumentRetrievalQuery(Record):
    query = String()
    user = String()
    collection = String()
    doc_limit = Integer()

class DocumentRetrievalResponse(Record):
    error = Error()
    documents = Array(String())  # List of document text chunks
    metadata = Map(String())  # {doc_count, etc.}

############################################################################

# Queue definitions for graph retrieval
graph_retrieval_request_queue = topic(
    'graph-retrieval', kind='non-persistent', namespace='request'
)
graph_retrieval_response_queue = topic(
    'graph-retrieval', kind='non-persistent', namespace='response'
)

# Queue definitions for document retrieval
document_retrieval_request_queue = topic(
    'document-retrieval', kind='non-persistent', namespace='request'
)
document_retrieval_response_queue = topic(
    'document-retrieval', kind='non-persistent', namespace='response'
)

