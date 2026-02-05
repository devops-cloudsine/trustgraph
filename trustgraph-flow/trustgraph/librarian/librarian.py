
from .. schema import LibrarianRequest, LibrarianResponse, Error, Triple, DocumentStatus, GraphStatus
from .. knowledge import hash
from .. exceptions import RequestError
from .. tables.library import LibraryTableStore
from .. tables.knowledge import KnowledgeTableStore
from . blob_store import BlobStore
import base64
import logging

from qdrant_client import QdrantClient
from .. direct.qdrant_helpers import QdrantVectorManager

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
            milvus_uri=None,
    ):

        self.blob_store = BlobStore(
            minio_host, minio_access_key, minio_secret_key, bucket_name
        )

        self.table_store = LibraryTableStore(
            cassandra_host, cassandra_username, cassandra_password, keyspace
        )

        # Knowledge table store for graph status queries
        self.knowledge_store = KnowledgeTableStore(
            cassandra_host, cassandra_username, cassandra_password, "knowledge"
        )

        self.load_document = load_document

        # Qdrant connection for document status check
        self.qdrant_uri = milvus_uri or "http://qdrant:6333"  # Use qdrant default
        try:
            self.qdrant = QdrantClient(url=self.qdrant_uri)
            logger.info(f"Librarian connected to Qdrant at {self.qdrant_uri}")
        except Exception as e:
            logger.warning(f"Could not connect to Qdrant: {e}. Status checks may be limited.")
            self.qdrant = None
        
        # Vector store manager for cascade deletion
        try:
            self.vector_manager = QdrantVectorManager(
                qdrant_uri=self.qdrant_uri,
                api_key=None
            )
            logger.info("Librarian connected to vector store manager")
        except Exception as e:
            logger.warning(f"Could not create vector manager: {e}")
            self.vector_manager = None

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
            document_status = None,
        )

    # ---> HTTP API Gateway > Processor.process_request('remove-document') > [Librarian.remove_document] > BlobStore.remove + LibraryTableStore.remove_document
    async def remove_document(self, request):
        """
        Remove document and all associated data (cascade delete).
        
        This operation permanently deletes:
        - Document embeddings (chunks used for retrieval)
        - Knowledge graph data (triples and graph embeddings)
        - Processing records
        - MinIO blob (raw document content)
        - Document metadata
        
        Order of operations ensures recoverable data is deleted first,
        with document metadata deletion serving as the final commit point.
        """

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

        # === CASCADE DELETE ALL RELATED DATA ===
        
        # 1. Delete document embeddings (chunks used for retrieval)
        try:
            await self.knowledge_store.delete_document_embeddings(
                request.user,
                request.document_id
            )
            logger.info(f"Deleted document embeddings for {request.document_id}")
        except Exception as e:
            logger.warning(f"Failed to delete document embeddings: {e}")

        # 2. Delete knowledge graph data (triples + graph embeddings)
        try:
            await self.knowledge_store.delete_kg_core(
                request.user,
                request.document_id
            )
            logger.info(f"Deleted knowledge graph data for {request.document_id}")
        except Exception as e:
            logger.warning(f"Failed to delete knowledge graph data: {e}")

        # 2.5. Get processing records to identify collections (before deleting them)
        collections_to_clean = set()
        processing_list = []
        try:
            processing_list = await self.table_store.list_processing(request.user)
            for proc in processing_list:
                if proc.document_id == request.document_id:
                    if proc.collection:
                        collections_to_clean.add(proc.collection)
        except Exception as e:
            logger.warning(f"Failed to get processing records for collection extraction: {e}")

        # 2.6. Delete document vectors from Qdrant vector stores
        if self.vector_manager and collections_to_clean:
            try:
                total_deleted = 0
                for coll in collections_to_clean:
                    # Delete document chunk vectors
                    doc_count = self.vector_manager.delete_document_vectors(
                        request.user, coll, request.document_id
                    )
                    total_deleted += doc_count
                    
                    # Delete graph entity vectors
                    graph_count = self.vector_manager.delete_graph_vectors(
                        request.user, coll, request.document_id
                    )
                    total_deleted += graph_count
                    
                    logger.debug(f"Deleted {doc_count} doc vectors and {graph_count} graph vectors from collection {coll}")
                
                if total_deleted > 0:
                    logger.info(f"Deleted {total_deleted} total vectors from vector stores for {request.document_id}")
            except Exception as e:
                logger.warning(f"Failed to delete vectors from vector store: {e}")
        elif not self.vector_manager:
            logger.debug("Vector manager not available, skipping vector deletion")
        elif not collections_to_clean:
            logger.debug(f"No collections found for document {request.document_id}, skipping vector deletion")

        # 2.7. Delete triples from graph database (per-user keyspace)
        if collections_to_clean:
            try:
                from ..direct.cassandra_kg import KnowledgeGraph
                
                # Connect to user's graph keyspace
                kg = KnowledgeGraph(
                    hosts=self.table_store.cassandra_host,
                    keyspace=request.user,
                    username=self.table_store.cassandra_username,
                    password=self.table_store.cassandra_password
                )
                
                total_triples_deleted = 0
                for coll in collections_to_clean:
                    count = kg.delete_document_triples(coll, request.document_id)
                    total_triples_deleted += count
                    logger.debug(f"Deleted {count} triples from collection {coll}")
                
                if total_triples_deleted > 0:
                    logger.info(f"Deleted {total_triples_deleted} total triples from graph database for {request.document_id}")
                
                # Clean up connection
                kg.close()
                
            except Exception as e:
                logger.warning(f"Failed to delete triples from graph database: {e}")

        # 3. Delete processing records
        try:
            deleted_count = 0
            for proc in processing_list:
                if proc.document_id == request.document_id:
                    await self.table_store.remove_processing(
                        request.user,
                        proc.id
                    )
                    deleted_count += 1
                    logger.debug(f"Deleted processing record {proc.id}")
            if deleted_count > 0:
                logger.info(f"Deleted {deleted_count} processing record(s) for {request.document_id}")
        except Exception as e:
            logger.warning(f"Failed to delete processing records: {e}")

        # 4. Remove blob from MinIO
        try:
            await self.blob_store.remove(object_id)
            logger.info(f"Deleted blob {object_id} from MinIO")
        except Exception as e:
            logger.warning(f"Failed to delete blob: {e}")

        # 5. Remove document metadata (do this last as a commit point)
        await self.table_store.remove_document(
            request.user,
            request.document_id
        )

        logger.info(f"Document {request.document_id} completely removed (cascade delete)")

        return LibrarianResponse(
            error = None,
            document_metadata = None,
            content = None,
            document_metadatas = None,
            processing_metadatas = None,
            document_status = None,
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
            document_status = None,
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
            document_status = None,
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
            document_status = None,
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
            document_status = None,
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
            document_status = None,
        )

    # ---> HTTP API Gateway > Processor.process_request('list-documents') > [Librarian.list_documents] > returns LibrarianResponse
    async def list_documents(self, request):
        """
        List documents filtered by user, collection, or both.
        
        Request can contain:
        - user: Filter by user ID
        - collection: Filter by collection ID
        - Both: Filter by both user and collection
        """
        user = getattr(request, 'user', None)
        collection = getattr(request, 'collection', None)

        docs = await self.table_store.list_documents(user=user, collection=collection)

        return LibrarianResponse(
            error = None,
            document_metadata = None,
            content = None,
            document_metadatas = docs,
            processing_metadatas = None,
            document_status = None,
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
            document_status = None,
        )

    # ---> HTTP API Gateway > Processor.process_request('get-document-status') > [Librarian.get_document_status] > returns LibrarianResponse with DocumentStatus
    async def get_document_status(self, request):
        """
        Get the processing status and embedding count for a document.
        Checks if document processing is complete by querying Milvus embeddings.
        """
        logger.debug(f"Getting document status for {request.user}/{request.document_id}")

        user = request.user
        document_id = request.document_id
        collection = request.collection

        # Check if document exists in library
        doc_exists = await self.table_store.document_exists(user, document_id)
        
        if not doc_exists:
            doc_status = DocumentStatus(
                document_id=document_id,
                user=user,
                collection=collection or "",
                status="not_found",
                embedding_count=0,
                chunk_count=0,
                last_updated=0,
            )
            return LibrarianResponse(
                error=None,
                document_metadata=None,
                content=None,
                document_metadatas=None,
                processing_metadatas=None,
                document_status=doc_status,
            )

        # Check if there's a processing record for this document
        processing_exists = await self._has_processing_for_document(user, document_id)
        
        # If no collection specified, try to get it from processing record
        if not collection and processing_exists:
            collection = await self._get_collection_for_document(user, document_id)

        # Query Qdrant for embedding count
        embedding_count = 0
        status = "pending"
        
        if collection and self.qdrant:
            try:
                # Qdrant collection name format: d_{user}_{collection}
                qdrant_collection = f"d_{user}_{collection}"
                
                if self.qdrant.collection_exists(qdrant_collection):
                    # Get collection info to get point count
                    collection_info = self.qdrant.get_collection(qdrant_collection)
                    embedding_count = collection_info.points_count or 0
                    
                    if embedding_count > 0:
                        status = "completed"
                    elif processing_exists:
                        status = "processing"
                    else:
                        status = "pending"
                elif processing_exists:
                    status = "processing"
            except Exception as e:
                logger.warning(f"Error querying Qdrant for status: {e}")
                if processing_exists:
                    status = "processing"
        elif processing_exists:
            status = "processing"

        doc_status = DocumentStatus(
            document_id=document_id,
            user=user,
            collection=collection or "",
            status=status,
            embedding_count=embedding_count,
            chunk_count=embedding_count,  # In Milvus, each row is a chunk
            last_updated=0,  # Milvus doesn't track timestamps per-document
        )

        return LibrarianResponse(
            error=None,
            document_metadata=None,
            content=None,
            document_metadatas=None,
            processing_metadatas=None,
            document_status=doc_status,
        )

    async def _has_processing_for_document(self, user, document_id):
        """Check if any processing record exists for this document"""
        try:
            processing_list = await self.table_store.list_processing(user)
            for proc in processing_list:
                if proc.document_id == document_id:
                    return True
            return False
        except Exception:
            return False

    async def _get_collection_for_document(self, user, document_id):
        """Get the collection name from processing record for a document"""
        try:
            processing_list = await self.table_store.list_processing(user)
            for proc in processing_list:
                if proc.document_id == document_id:
                    return proc.collection
            return None
        except Exception:
            return None

    async def _count_user_triples(self, user, collection):
        """Count triples in the user's graph store (user keyspace)"""
        try:
            from cassandra.cluster import Cluster
            from cassandra.auth import PlainTextAuthProvider
            
            # Connect to Cassandra
            if self.table_store.cassandra_username and self.table_store.cassandra_password:
                auth_provider = PlainTextAuthProvider(
                    username=self.table_store.cassandra_username,
                    password=self.table_store.cassandra_password
                )
                cluster = Cluster(self.table_store.cassandra_host, auth_provider=auth_provider)
            else:
                cluster = Cluster(self.table_store.cassandra_host)
            
            session = cluster.connect()
            
            # Query the user's triples store
            query = f"""
                SELECT COUNT(*) FROM {user}.triples_s 
                WHERE collection = %s ALLOW FILTERING
            """
            result = session.execute(query, (collection,))
            count = result.one()[0]
            
            session.shutdown()
            cluster.shutdown()
            
            return count
        except Exception as e:
            logger.warning(f"Could not count user triples: {e}")
            return 0

    async def get_graph_status(self, request):
        """
        Get graph processing status for a document.
        Returns chunk-level progress and completion status.
        """
        user = request.user
        document_id = request.document_id
        collection = request.collection

        # 1. Check if document exists
        doc_exists = await self.table_store.document_exists(user, document_id)
        if not doc_exists:
            return LibrarianResponse(
                error=None,
                document_metadata=None,
                content=None,
                document_metadatas=None,
                processing_metadatas=None,
                document_status=None,
                graph_status=GraphStatus(
                    document_id=document_id,
                    user=user,
                    collection=collection or "",
                    status="not_found",
                    triples_count=0,
                    graph_embeddings_count=0,
                    chunks_total=0,
                    chunks_processed=0,
                    triples_stored=0,
                    embeddings_stored=0,
                    failed_chunks=0,
                    last_updated=0,
                )
            )

        # 2. Get chunk progress
        progress = await self.table_store.get_chunk_progress(user, document_id)
        
        total_chunks = progress["total_chunks"]
        triples_stored = progress["triples_stored"]
        embeddings_stored = progress["embeddings_stored"]
        failed_chunks = progress["failed_chunks"]
        last_updated = progress["last_updated"]

        # 3. Get actual data counts from knowledge keyspace (for info only)
        knowledge_status = await self.knowledge_store.get_graph_status(user, document_id)
        triples_count = knowledge_status["triples_count"]
        graph_embeddings_count = knowledge_status["graph_embeddings_count"]

        # 4. Determine status based on chunk progress
        # A chunk is complete when BOTH triples AND embeddings are stored
        # Counts are now based on unique chunk_ids, so no need to cap
        chunks_processed = min(triples_stored, embeddings_stored)
        
        if total_chunks == 0:
            status = "pending"  # Not yet chunked
        elif failed_chunks > 0:
            status = "failed"   # Some chunks failed
        elif chunks_processed >= total_chunks:
            status = "completed"  # All chunks processed
        else:
            status = "processing"  # In progress

        return LibrarianResponse(
            error=None,
            document_metadata=None,
            content=None,
            document_metadatas=None,
            processing_metadatas=None,
            document_status=None,
            graph_status=GraphStatus(
                document_id=document_id,
                user=user,
                collection=collection or "",
                status=status,
                triples_count=triples_count,
                graph_embeddings_count=graph_embeddings_count,
            chunks_total=total_chunks,
            chunks_processed=chunks_processed,
            triples_stored=triples_stored,  # Unique chunk count
            embeddings_stored=embeddings_stored,  # Unique chunk count
            failed_chunks=failed_chunks,
            last_updated=last_updated,
            )
        )

