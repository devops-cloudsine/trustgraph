
from pulsar.schema import Record, Bytes, String, Array, Long, Integer
from ..core.primitives import Triple, Error
from ..core.topic import topic
from ..core.metadata import Metadata
from ..knowledge.document import Document, TextDocument

# add-document
#   -> (document_id, document_metadata, content)
#   <- ()
#   <- (error)

# remove-document
#   -> (document_id)
#   <- ()
#   <- (error)

# update-document
#   -> (document_id, document_metadata)
#   <- ()
#   <- (error)

# get-document-metadata
#   -> (document_id)
#   <- (document_metadata)
#   <- (error)

# get-document-content
#   -> (document_id)
#   <- (content)
#   <- (error)

# add-processing
#   -> (processing_id, processing_metadata)
#   <- ()
#   <- (error)

# remove-processing
#   -> (processing_id)
#   <- ()
#   <- (error)

# list-documents
#   -> (user, collection?)
#   <- (document_metadata[])
#   <- (error)

# list-processing
#   -> (user, collection?)
#   <- (processing_metadata[])
#   <- (error)

# get-document-status
#   -> (document_id, user, collection?)
#   <- (document_status)
#   <- (error)

class DocumentMetadata(Record):
    id = String()
    time = Long()
    kind = String()
    title = String()
    comments = String()
    metadata = Array(Triple())
    user = String()
    tags = Array(String())

class ProcessingMetadata(Record):
    id = String()
    document_id = String()
    time = Long()
    flow = String()
    user = String()
    collection = String()
    tags = Array(String())

class Criteria(Record):
    key = String()
    value = String()
    operator = String()

class DocumentStatus(Record):
    document_id = String()
    user = String()
    collection = String()
    status = String()  # "pending" | "processing" | "completed" | "not_found"
    embedding_count = Long()
    chunk_count = Long()
    last_updated = Long()  # Unix timestamp in milliseconds

class GraphStatus(Record):
    document_id = String()
    user = String()
    collection = String()
    status = String()  # "not_found" | "pending" | "processing" | "completed" | "failed"
    triples_count = Long()
    graph_embeddings_count = Long()
    chunks_total = Integer()           # Total number of chunks created from document
    chunks_processed = Integer()       # Number of chunks fully processed (min of triples_stored, embeddings_stored)
    triples_stored = Integer()         # Number of chunks with triples stored in user keyspace
    embeddings_stored = Integer()      # Number of chunks with graph embeddings stored in vector DB
    failed_chunks = Integer()          # Number of chunks that failed processing
    last_updated = Long()  # Unix timestamp in milliseconds

class LibrarianRequest(Record):

    # add-document, remove-document, update-document, get-document-metadata,
    # get-document-content, add-processing, remove-processing, list-documents,
    # list-processing, get-document-status
    operation = String()

    # add-document, remove-document, update-document, get-document-metadata,
    # get-document-content
    document_id = String()

    # add-processing, remove-processing
    processing_id = String()

    # add-document, update-document
    document_metadata = DocumentMetadata()

    # add-processing
    processing_metadata = ProcessingMetadata()

    # add-document
    content = Bytes()

    # list-documents, list-processing
    user = String()

    # list-documents?, list-processing?
    collection = String()

    # 
    criteria = Array(Criteria())

class LibrarianResponse(Record):
    error = Error()
    document_metadata = DocumentMetadata()
    content = Bytes()
    document_metadatas = Array(DocumentMetadata())
    processing_metadatas = Array(ProcessingMetadata())
    document_status = DocumentStatus()
    graph_status = GraphStatus()

# FIXME: Is this right?  Using persistence on librarian so that
# message chunking works

librarian_request_queue = topic(
    'librarian', kind='persistent', namespace='request'
)
librarian_response_queue = topic(
    'librarian', kind='persistent', namespace='response',
)

