
from .. schema import LibrarianRequest, LibrarianResponse, Error, Triple
from .. knowledge import hash
from .. exceptions import RequestError
from .. tables.library import LibraryTableStore
from . blob_store import BlobStore
import base64
import logging

import uuid

# Module logger
logger = logging.getLogger(__name__)

# Textual kinds are routed to text-load
SUPPORTED_TEXT_KINDS = {
    "text/plain",
    "text/csv",
    "text/html",
    "text/calendar",
    "text/markdown",
    "application/json",
}

# Binary/non-text kinds are routed to document-load
SUPPORTED_BINARY_KINDS = {
    "application/pdf",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "image/gif",
    "image/jpeg",
    "image/png",
    "application/vnd.ms-powerpoint",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "image/webp",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}

# Backward-compatible union of supported kinds
SUPPORTED_DOCUMENT_KINDS = SUPPORTED_TEXT_KINDS | SUPPORTED_BINARY_KINDS

class Librarian:

    def __init__(
            self,
            cassandra_host, cassandra_username, cassandra_password,
            minio_host, minio_access_key, minio_secret_key,
            bucket_name, keyspace, load_document,
    ):

        self.blob_store = BlobStore(
            minio_host, minio_access_key, minio_secret_key, bucket_name
        )

        self.table_store = LibraryTableStore(
            cassandra_host, cassandra_username, cassandra_password, keyspace
        )

        self.load_document = load_document

    # ---> HTTP API Gateway > Processor.process_request('add-document') > [Librarian.add_document] > BlobStore.add + LibraryTableStore.add_document
    async def add_document(self, request):

        kind = (request.document_metadata.kind or "").lower()

        if kind not in SUPPORTED_DOCUMENT_KINDS:
            raise RequestError(
                "Invalid document kind: "
                + str(request.document_metadata.kind)
                + ". Supported kinds: "
                + ", ".join(sorted(SUPPORTED_DOCUMENT_KINDS))
            )

        if await self.table_store.document_exists(
                request.document_metadata.user,
                request.document_metadata.id
        ):
            raise RuntimeError("Document already exists")

        # Create object ID for blob
        object_id = uuid.uuid4()

        logger.debug("Adding blob...")

        await self.blob_store.add(
            object_id, base64.b64decode(request.content),
            request.document_metadata.kind
        )

        logger.debug("Adding to table...")

        await self.table_store.add_document(
            request.document_metadata, object_id
        )

        logger.debug("Add complete")

        return LibrarianResponse(
            error = None,
            document_metadata = None,
            content = None,
            document_metadatas = None,
            processing_metadatas = None,
        )

    # ---> HTTP API Gateway > Processor.process_request('remove-document') > [Librarian.remove_document] > BlobStore.remove + LibraryTableStore.remove_document
    async def remove_document(self, request):

        logger.debug("Removing document...")

        if not await self.table_store.document_exists(
                request.user,
                request.document_id,
        ):
            raise RuntimeError("Document does not exist")

        object_id = await self.table_store.get_document_object_id(
            request.user,
            request.document_id
        )

        # Remove blob...
        await self.blob_store.remove(object_id)

        # Remove doc table row
        await self.table_store.remove_document(
            request.user,
            request.document_id
        )

        logger.debug("Remove complete")

        return LibrarianResponse(
            error = None,
            document_metadata = None,
            content = None,
            document_metadatas = None,
            processing_metadatas = None,
        )

    # ---> HTTP API Gateway > Processor.process_request('update-document') > [Librarian.update_document] > LibraryTableStore.update_document
    async def update_document(self, request):

        logger.debug("Updating document...")

        # You can't update the document ID, user or kind.

        if not await self.table_store.document_exists(
                request.document_metadata.user,
                request.document_metadata.id
        ):
            raise RuntimeError("Document does not exist")

        await self.table_store.update_document(request.document_metadata)

        logger.debug("Update complete")

        return LibrarianResponse(
            error = None,
            document_metadata = None,
            content = None,
            document_metadatas = None,
            processing_metadatas = None,
        )

    # ---> HTTP API Gateway > Processor.process_request('get-document-metadata') > [Librarian.get_document_metadata] > returns LibrarianResponse
    async def get_document_metadata(self, request):

        logger.debug("Getting document metadata...")

        doc = await self.table_store.get_document(
            request.user,
            request.document_id
        )

        logger.debug("Get complete")

        return LibrarianResponse(
            error = None,
            document_metadata = doc,
            content = None,
            document_metadatas = None,
            processing_metadatas = None,
        )

    # ---> HTTP API Gateway > Processor.process_request('get-document-content') > [Librarian.get_document_content] > BlobStore.get -> returns LibrarianResponse
    async def get_document_content(self, request):

        logger.debug("Getting document content...")

        object_id = await self.table_store.get_document_object_id(
            request.user,
            request.document_id
        )

        content = await self.blob_store.get(
            object_id
        )

        logger.debug("Get complete")

        return LibrarianResponse(
            error = None,
            document_metadata = None,
            content = base64.b64encode(content).decode("utf-8"),
            document_metadatas = None,
            processing_metadatas = None,
        )

    # ---> Processor.add_processing_with_collection > [Librarian.add_processing] > Processor.load_document(document) -> flow('text-load'|'document-load')
    async def add_processing(self, request):

        logger.debug("Adding processing metadata...")

        if not request.processing_metadata.collection:
            raise RuntimeError("Collection parameter is required")

        if await self.table_store.processing_exists(
                request.processing_metadata.user,
                request.processing_metadata.id
        ):
            raise RuntimeError("Processing already exists")

        doc = await self.table_store.get_document(
            request.processing_metadata.user,
            request.processing_metadata.document_id
        )

        object_id = await self.table_store.get_document_object_id(
            request.processing_metadata.user,
            request.processing_metadata.document_id
        )

        content = await self.blob_store.get(
            object_id
        )

        logger.debug("Retrieved content")

        logger.debug("Adding processing to table...")

        await self.table_store.add_processing(request.processing_metadata)

        logger.debug("Invoking document processing...")

        await self.load_document(
            document = doc,
            processing = request.processing_metadata,
            content = content,
        )

        logger.debug("Add complete")

        return LibrarianResponse(
            error = None,
            document_metadata = None,
            content = None,
            document_metadatas = None,
            processing_metadatas = None,
        )

    # ---> HTTP API Gateway > Processor.process_request('remove-processing') > [Librarian.remove_processing] > LibraryTableStore.remove_processing
    async def remove_processing(self, request):

        logger.debug("Removing processing metadata...")

        if not await self.table_store.processing_exists(
                request.user,
                request.processing_id,
        ):
            raise RuntimeError("Processing object does not exist")

        # Remove doc table row
        await self.table_store.remove_processing(
            request.user,
            request.processing_id
        )

        logger.debug("Remove complete")

        return LibrarianResponse(
            error = None,
            document_metadata = None,
            content = None,
            document_metadatas = None,
            processing_metadatas = None,
        )

    # ---> HTTP API Gateway > Processor.process_request('list-documents') > [Librarian.list_documents] > returns LibrarianResponse
    async def list_documents(self, request):

        docs = await self.table_store.list_documents(request.user)

        return LibrarianResponse(
            error = None,
            document_metadata = None,
            content = None,
            document_metadatas = docs,
            processing_metadatas = None,
        )

    # ---> HTTP API Gateway > Processor.process_request('list-processing') > [Librarian.list_processing] > returns LibrarianResponse
    async def list_processing(self, request):

        procs = await self.table_store.list_processing(request.user)

        return LibrarianResponse(
            error = None,
            document_metadata = None,
            content = None,
            document_metadatas = None,
            processing_metadatas = procs,
        )

