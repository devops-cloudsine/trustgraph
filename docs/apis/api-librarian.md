# TrustGraph Librarian API

This API provides document library management for TrustGraph. It handles document storage, 
metadata management, and processing orchestration using hybrid storage (MinIO for content, 
Cassandra for metadata) with multi-user support.

## Request/response

### Request

The request contains the following fields:
- `operation`: The operation to perform (see operations below)
- `document_id`: Document identifier (for document operations)
- `document_metadata`: Document metadata object (for add/update operations)
  - `id`: Document identifier (required)
  - `time`: Unix timestamp in seconds as a float (required for add operations)
  - `kind`: MIME type of document (required, e.g., "text/plain", "application/pdf")
  - `title`: Document title (optional)
  - `comments`: Document comments (optional)
  - `user`: Document owner (required)
  - `tags`: Array of tags (optional)
  - `metadata`: Array of RDF triples (optional) - each triple has:
    - `s`: Subject with `v` (value) and `e` (is_uri boolean)
    - `p`: Predicate with `v` (value) and `e` (is_uri boolean)
    - `o`: Object with `v` (value) and `e` (is_uri boolean)
- `content`: Document content as base64-encoded bytes (for add operations)
- `processing_id`: Processing job identifier (for processing operations)
- `processing_metadata`: Processing metadata object (for add-processing)
- `user`: User identifier (required for most operations)
- `collection`: Collection filter (optional for list operations)
- `criteria`: Query criteria array (for filtering operations)

### Response

The response contains the following fields:
- `error`: Error information if operation fails
- `document_metadata`: Single document metadata (for get operations)
- `content`: Document content as base64-encoded bytes (for get-content)
- `document_metadatas`: Array of document metadata (for list operations)
- `processing_metadatas`: Array of processing metadata (for list-processing)

## Document Operations

### ADD-DOCUMENT - Add Document to Library

Request:
```json
{
    "operation": "add-document",
    "document_metadata": {
        "id": "doc-123",
        "time": 1640995200.0,
        "kind": "application/pdf",
        "title": "Research Paper",
        "comments": "Important research findings",
        "user": "alice",
        "tags": ["research", "ai", "machine-learning"],
        "metadata": [
            {
                "s": {
                    "v": "http://example.com/doc-123",
                    "e": true
                },
                "p": {
                    "v": "http://purl.org/dc/elements/1.1/creator",
                    "e": true
                },
                "o": {
                    "v": "Dr. Smith",
                    "e": false
                }
            }
        ]
    },
    "content": "JVBERi0xLjQKMSAwIG9iago8PAovVHlwZSAvQ2F0YWxvZwovUGFnZXMgMiAwIFIKPj4KZW5kb2JqCg=="
}
```

Response:
```json
{}
```

### GET-DOCUMENT-METADATA - Get Document Metadata

Request:
```json
{
    "operation": "get-document-metadata",
    "document_id": "doc-123",
    "user": "alice"
}
```

Response:
```json
{
    "document_metadata": {
        "id": "doc-123",
        "time": 1640995200.0,
        "kind": "application/pdf",
        "title": "Research Paper",
        "comments": "Important research findings",
        "user": "alice",
        "tags": ["research", "ai", "machine-learning"],
        "metadata": [
            {
                "s": {
                    "v": "http://example.com/doc-123",
                    "e": true
                },
                "p": {
                    "v": "http://purl.org/dc/elements/1.1/creator",
                    "e": true
                },
                "o": {
                    "v": "Dr. Smith",
                    "e": false
                }
            }
        ]
    }
}
```

### GET-DOCUMENT-CONTENT - Get Document Content

Request:
```json
{
    "operation": "get-document-content",
    "document_id": "doc-123",
    "user": "alice"
}
```

Response:
```json
{
    "content": "JVBERi0xLjQKMSAwIG9iago8PAovVHlwZSAvQ2F0YWxvZwovUGFnZXMgMiAwIFIKPj4KZW5kb2JqCg=="
}
```

### LIST-DOCUMENTS - List Documents

List documents filtered by user, collection, or both. At least one filter parameter must be provided.

**Filtering Options:**
- **user only**: Returns all documents for the specified user
- **collection only**: Returns all documents in the specified collection (across all users)
- **user + collection**: Returns documents for the specified user in the specified collection

Request (with both filters):
```json
{
    "operation": "list-documents",
    "user": "alice",
    "collection": "research"
}
```

Request (user only):
```json
{
    "operation": "list-documents",
    "user": "alice"
}
```

Request (collection only):
```json
{
    "operation": "list-documents",
    "collection": "research"
}
```

Response:
```json
{
    "document_metadatas": [
        {
            "id": "doc-123",
            "time": 1640995200.0,
            "kind": "application/pdf",
            "title": "Research Paper",
            "comments": "Important research findings",
            "user": "alice",
            "tags": ["research", "ai"]
        },
        {
            "id": "doc-124",
            "time": 1640995300.0,
            "kind": "text/plain",
            "title": "Meeting Notes",
            "comments": "Team meeting discussion",
            "user": "alice",
            "tags": ["meeting", "notes"]
        }
    ]
}
```

### UPDATE-DOCUMENT - Update Document Metadata

Request:
```json
{
    "operation": "update-document",
    "document_metadata": {
        "id": "doc-123",
        "time": 1640995500.0,
        "title": "Updated Research Paper",
        "comments": "Updated findings and conclusions",
        "user": "alice",
        "tags": ["research", "ai", "machine-learning", "updated"],
        "metadata": []
    }
}
```

Response:
```json
{}
```

### REMOVE-DOCUMENT - Remove Document

Permanently removes a document and all associated data from the TrustGraph system. This operation performs a **cascade delete** that removes:

- **Document embeddings**: All chunked document text and vector embeddings used for retrieval
- **Knowledge graph data**: RDF triples and graph embeddings extracted from the document
- **Processing records**: All processing job history associated with the document
- **Document blob**: Raw document content stored in MinIO
- **Document metadata**: Document entry in the library catalog

**Warning**: This operation is **irreversible** and permanently deletes all data. Ensure you have backups if the document might be needed in the future.

**Note**: After removal, the document will no longer appear in:
- Document library listings
- Document retrieval queries
- Knowledge graph queries
- Processing history

Request:
```json
{
    "operation": "remove-document",
    "document_id": "doc-123",
    "user": "alice"
}
```

Response:
```json
{}
```

## Processing Operations

### ADD-PROCESSING - Start Document Processing

Request:
```json
{
    "operation": "add-processing",
    "processing_metadata": {
        "id": "proc-456",
        "document_id": "doc-123",
        "time": 1640995400.0,
        "flow": "pdf-extraction",
        "user": "alice",
        "collection": "research",
        "tags": ["extraction", "nlp"]
    }
}
```

Response:
```json
{}
```

### LIST-PROCESSING - List Processing Jobs

Request:
```json
{
    "operation": "list-processing",
    "user": "alice",
    "collection": "research"
}
```

Response:
```json
{
    "processing_metadatas": [
        {
            "id": "proc-456",
            "document_id": "doc-123",
            "time": 1640995400.0,
            "flow": "pdf-extraction",
            "user": "alice",
            "collection": "research",
            "tags": ["extraction", "nlp"]
        }
    ]
}
```

### REMOVE-PROCESSING - Stop Processing Job

Request:
```json
{
    "operation": "remove-processing",
    "processing_id": "proc-456",
    "user": "alice"
}
```

Response:
```json
{}
```

### GET-DOCUMENT-STATUS - Check Document Processing Status

Check whether a document has completed processing and has obtained all embeddings.

Request:
```json
{
    "operation": "get-document-status",
    "document-id": "doc-123",
    "user": "alice",
    "collection": "research"
}
```

Response:
```json
{
    "document-status": {
        "document-id": "doc-123",
        "user": "alice",
        "collection": "research",
        "status": "completed",
        "embedding-count": 42,
        "chunk-count": 15,
        "last-updated": 1640995500000
    }
}
```

**Status Values:**
- `not_found` - Document does not exist in the library
- `pending` - Document exists but no processing has been started
- `processing` - Processing has been started but no embeddings found yet
- `completed` - Embeddings have been generated and stored

### GET-GRAPH-STATUS - Check Graph Extraction Status

Check whether a document's graph entities and triples have been extracted and stored. This endpoint provides chunk-level progress tracking to reliably determine when graph processing is complete.

Request:
```json
{
    "operation": "get-graph-status",
    "document-id": "doc-123",
    "user": "alice",
    "collection": "research"
}
```

Response:
```json
{
    "graph-status": {
        "document-id": "doc-123",
        "user": "alice",
        "collection": "research",
        "status": "processing",
        "triples-count": 387,
        "graph-embeddings-count": 156,
        "chunks-total": 10,
        "chunks-processed": 7,
        "triples-stored": 8,
        "embeddings-stored": 7,
        "failed-chunks": 0,
        "last-updated": 1640995650000
    }
}
```

**Status Values:**
- `not_found` - Document does not exist in the library
- `pending` - Document exists but has not been chunked yet (`chunks-total` = 0)
- `processing` - Graph extraction is in progress (`chunks-processed` < `chunks-total`)
- `completed` - All chunks have been processed (`chunks-processed` >= `chunks-total`)
- `failed` - Some chunks failed processing (`failed-chunks` > 0)

**Field Descriptions:**
- `chunks-total`: Total number of chunks created from the document
- `chunks-processed`: Number of chunks fully processed (minimum of `triples-stored` and `embeddings-stored`)
- `triples-stored`: Number of chunks with triples stored in user keyspace (for graph queries)
- `embeddings-stored`: Number of chunks with graph embeddings stored in vector database
- `failed-chunks`: Number of chunks that failed processing
- `triples-count`: Total number of triples extracted (informational, from knowledge keyspace)
- `graph-embeddings-count`: Total number of graph embeddings created (informational, from knowledge keyspace)
- `last-updated`: Unix timestamp (milliseconds) of the most recent progress update

**Completion Logic:**
A chunk is considered "processed" only when **both** triples and graph embeddings have been stored. The `chunks-processed` value is the minimum of `triples-stored` and `embeddings-stored`, ensuring that status is "completed" only when all chunks have been fully processed and are ready for graph-retrieval queries.

## REST service

The REST service is available at `/api/v1/librarian` and accepts the above request formats.

## Websocket

Requests have a `request` object containing the operation fields.
Responses have a `response` object containing the response fields.

Request:
```json
{
    "id": "unique-request-id",
    "service": "librarian",
    "request": {
        "operation": "list-documents",
        "user": "alice"
    }
}
```

Response:
```json
{
    "id": "unique-request-id",
    "response": {
        "document_metadatas": [...]
    },
    "complete": true
}
```

## Pulsar

The Pulsar schema for the Librarian API is defined in Python code here:

https://github.com/trustgraph-ai/trustgraph/blob/master/trustgraph-base/trustgraph/schema/library.py

Default request queue:
`non-persistent://tg/request/librarian`

Default response queue:
`non-persistent://tg/response/librarian`

Request schema:
`trustgraph.schema.LibrarianRequest`

Response schema:
`trustgraph.schema.LibrarianResponse`

## Python SDK

The Python SDK provides convenient access to the Librarian API:

```python
from trustgraph.api.library import LibrarianClient

client = LibrarianClient()

# Add a document
with open("document.pdf", "rb") as f:
    content = f.read()
    
await client.add_document(
    doc_id="doc-123",
    title="Research Paper",
    content=content,
    user="alice",
    tags=["research", "ai"]
)

# Get document metadata
metadata = await client.get_document_metadata("doc-123", "alice")

# List documents
documents = await client.list_documents("alice", collection="research")

# Start processing
await client.add_processing(
    processing_id="proc-456",
    document_id="doc-123",
    flow="pdf-extraction",
    user="alice"
)

# Check document processing status
status = await client.get_document_status("doc-123", "alice", "research")
if status["status"] == "completed":
    print(f"Processing complete! {status['embedding-count']} embeddings generated.")
```

## Features

- **Hybrid Storage**: MinIO for content, Cassandra for metadata
- **Multi-user Support**: User-based document ownership and access control
- **Rich Metadata**: RDF-style metadata triples and tagging system
- **Processing Integration**: Automatic triggering of document processing workflows
- **Content Types**: Support for multiple document formats (PDF, text, etc.)
- **Collection Management**: Optional document grouping by collection
- **Metadata Search**: Query documents by metadata criteria

## Use Cases

- **Document Management**: Store and organize documents with rich metadata
- **Knowledge Extraction**: Process documents to extract structured knowledge
- **Research Libraries**: Manage collections of research papers and documents
- **Content Processing**: Orchestrate document processing workflows
- **Multi-tenant Systems**: Support multiple users with isolated document libraries