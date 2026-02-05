
from .. schema import LibrarianRequest, LibrarianResponse
from .. schema import DocumentMetadata, ProcessingMetadata
from .. schema import Error, Triple, Value
from .. knowledge import hash
from .. exceptions import RequestError

from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.query import BatchStatement
from ssl import SSLContext, PROTOCOL_TLSv1_2

import uuid
import time
import asyncio
import logging

logger = logging.getLogger(__name__)

class LibraryTableStore:

    def __init__(
            self,
            cassandra_host, cassandra_username, cassandra_password, keyspace,
    ):

        self.keyspace = keyspace

        logger.info("Connecting to Cassandra...")

        # Ensure cassandra_host is a list
        if isinstance(cassandra_host, str):
            cassandra_host = [h.strip() for h in cassandra_host.split(',')]

        # Store connection details for graph database access
        self.cassandra_host = cassandra_host
        self.cassandra_username = cassandra_username
        self.cassandra_password = cassandra_password

        if cassandra_username and cassandra_password:
            ssl_context = SSLContext(PROTOCOL_TLSv1_2)
            auth_provider = PlainTextAuthProvider(
                username=cassandra_username, password=cassandra_password
            )
            self.cluster = Cluster(
                cassandra_host,
                auth_provider=auth_provider,
                ssl_context=ssl_context
            )
        else:
            self.cluster = Cluster(cassandra_host)

        self.cassandra = self.cluster.connect()
        
        logger.info("Connected.")

        self.ensure_cassandra_schema()

        self.prepare_statements()

    def ensure_cassandra_schema(self):

        logger.debug("Ensure Cassandra schema...")

        logger.debug("Keyspace...")
        
        # FIXME: Replication factor should be configurable
        self.cassandra.execute(f"""
            create keyspace if not exists {self.keyspace}
                with replication = {{ 
                   'class' : 'SimpleStrategy', 
                   'replication_factor' : 1 
                }};
        """);

        self.cassandra.set_keyspace(self.keyspace)

        logger.debug("document table...")

        self.cassandra.execute("""
            CREATE TABLE IF NOT EXISTS document (
                id text,
                user text,
                time timestamp,
                kind text,
                title text,
                comments text,
                metadata list<tuple<
                    text, boolean, text, boolean, text, boolean
                >>,
                tags list<text>,
                object_id uuid,
                PRIMARY KEY (user, id)
            );
        """);

        logger.debug("object index...")

        self.cassandra.execute("""
            CREATE INDEX IF NOT EXISTS document_object
            ON document (object_id)
        """);

        logger.debug("processing table...")

        self.cassandra.execute("""
            CREATE TABLE IF NOT EXISTS processing (
                id text,
                document_id text,
                time timestamp,
                flow text,
                user text,
                collection text,
                tags list<text>,
                PRIMARY KEY (user, id)
            );
        """);

        logger.debug("processing collection index...")

        self.cassandra.execute("""
            CREATE INDEX IF NOT EXISTS processing_collection
            ON processing (collection)
        """);

        logger.debug("processing document_id index...")

        self.cassandra.execute("""
            CREATE INDEX IF NOT EXISTS processing_document_id
            ON processing (document_id)
        """);

        logger.debug("collections table...")

        self.cassandra.execute("""
            CREATE TABLE IF NOT EXISTS collections (
                user text,
                collection text,
                name text,
                description text,
                tags set<text>,
                created_at timestamp,
                updated_at timestamp,
                PRIMARY KEY (user, collection)
            );
        """);

        logger.debug("chunk_progress table...")

        self.cassandra.execute("""
            CREATE TABLE IF NOT EXISTS chunk_progress (
                user text,
                document_id text,
                total_chunks int,
                triples_stored int,
                embeddings_stored int,
                failed_chunks int,
                last_updated timestamp,
                PRIMARY KEY (user, document_id)
            );
        """);

        logger.debug("chunk_processing_status table...")

        self.cassandra.execute("""
            CREATE TABLE IF NOT EXISTS chunk_processing_status (
                user text,
                document_id text,
                chunk_id text,
                triples_stored boolean,
                embeddings_stored boolean,
                last_updated timestamp,
                PRIMARY KEY ((user, document_id), chunk_id)
            );
        """);

        logger.info("Cassandra schema OK.")

    def prepare_statements(self):

        self.insert_document_stmt = self.cassandra.prepare("""
            INSERT INTO document
            (
                id, user, time,
                kind, title, comments,
                metadata, tags, object_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """)

        # Cross-keyspace query for document embeddings status
        # Uses knowledge keyspace to check embedding status
        self.get_document_embeddings_status_stmt = self.cassandra.prepare("""
            SELECT id, time, chunks
            FROM knowledge.document_embeddings
            WHERE user = ? AND document_id = ?
        """)

        self.update_document_stmt = self.cassandra.prepare("""
            UPDATE document
            SET time = ?, title = ?, comments = ?,
                metadata = ?, tags = ?
            WHERE user = ? AND id = ?
        """)

        self.get_document_stmt = self.cassandra.prepare("""
            SELECT time, kind, title, comments, metadata, tags, object_id
            FROM document
            WHERE user = ? AND id = ?
        """)

        self.delete_document_stmt = self.cassandra.prepare("""
            DELETE FROM document
            WHERE user = ? AND id = ?
        """)

        self.test_document_exists_stmt = self.cassandra.prepare("""
            SELECT id
            FROM document
            WHERE user = ? AND id = ?
            LIMIT 1
        """)

        self.list_document_stmt = self.cassandra.prepare("""
            SELECT
                id, time, kind, title, comments, metadata, tags, object_id
            FROM document
            WHERE user = ?
        """)

        self.list_document_by_tag_stmt = self.cassandra.prepare("""
            SELECT
                id, time, kind, title, comments, metadata, tags, object_id
            FROM document
            WHERE user = ? AND tags CONTAINS ?
            ALLOW FILTERING
        """)

        # Get document IDs by collection (via processing table)
        # Note: May return duplicate document_ids, dedup in application code
        self.list_document_ids_by_collection_stmt = self.cassandra.prepare("""
            SELECT document_id, user
            FROM processing
            WHERE collection = ?
            ALLOW FILTERING
        """)

        # Get document IDs by user and collection (via processing table)
        # Note: May return duplicate document_ids, dedup in application code
        self.list_document_ids_by_user_collection_stmt = self.cassandra.prepare("""
            SELECT document_id
            FROM processing
            WHERE user = ? AND collection = ?
            ALLOW FILTERING
        """)

        self.insert_processing_stmt = self.cassandra.prepare("""
            INSERT INTO processing
            (
                id, document_id, time,
                flow, user, collection,
                tags
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """)

        self.delete_processing_stmt = self.cassandra.prepare("""
            DELETE FROM processing
            WHERE user = ? AND id = ?
        """)

        self.test_processing_exists_stmt = self.cassandra.prepare("""
            SELECT id
            FROM processing
            WHERE user = ? AND id = ?
            LIMIT 1
        """)

        # Collection management statements
        self.insert_collection_stmt = self.cassandra.prepare("""
            INSERT INTO collections
            (user, collection, name, description, tags, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """)

        self.update_collection_stmt = self.cassandra.prepare("""
            UPDATE collections
            SET name = ?, description = ?, tags = ?, updated_at = ?
            WHERE user = ? AND collection = ?
        """)

        self.get_collection_stmt = self.cassandra.prepare("""
            SELECT collection, name, description, tags, created_at, updated_at
            FROM collections
            WHERE user = ? AND collection = ?
        """)

        self.list_collections_stmt = self.cassandra.prepare("""
            SELECT collection, name, description, tags, created_at, updated_at
            FROM collections
            WHERE user = ?
        """)

        self.delete_collection_stmt = self.cassandra.prepare("""
            DELETE FROM collections
            WHERE user = ? AND collection = ?
        """)

        self.collection_exists_stmt = self.cassandra.prepare("""
            SELECT collection
            FROM collections
            WHERE user = ? AND collection = ?
            LIMIT 1
        """)

        # Chunk progress tracking statements
        self.init_chunk_progress_stmt = self.cassandra.prepare("""
            INSERT INTO chunk_progress 
            (user, document_id, total_chunks, triples_stored, embeddings_stored, failed_chunks, last_updated)
            VALUES (?, ?, ?, 0, 0, 0, ?)
            IF NOT EXISTS
        """)

        self.increment_triples_stored_stmt = self.cassandra.prepare("""
            UPDATE chunk_progress 
            SET triples_stored = ?, last_updated = ?
            WHERE user = ? AND document_id = ?
        """)

        self.increment_embeddings_stored_stmt = self.cassandra.prepare("""
            UPDATE chunk_progress 
            SET embeddings_stored = ?, last_updated = ?
            WHERE user = ? AND document_id = ?
        """)

        self.increment_failed_chunks_stmt = self.cassandra.prepare("""
            UPDATE chunk_progress 
            SET failed_chunks = ?, last_updated = ?
            WHERE user = ? AND document_id = ?
        """)

        self.get_chunk_progress_stmt = self.cassandra.prepare("""
            SELECT total_chunks, triples_stored, embeddings_stored, failed_chunks, last_updated
            FROM chunk_progress
            WHERE user = ? AND document_id = ?
        """)

        # Chunk processing status tracking (unique chunk_ids)
        self.mark_chunk_triples_stored_stmt = self.cassandra.prepare("""
            INSERT INTO chunk_processing_status 
            (user, document_id, chunk_id, triples_stored, embeddings_stored, last_updated)
            VALUES (?, ?, ?, true, false, ?)
        """)

        self.mark_chunk_embeddings_stored_stmt = self.cassandra.prepare("""
            UPDATE chunk_processing_status 
            SET embeddings_stored = true, last_updated = ?
            WHERE user = ? AND document_id = ? AND chunk_id = ?
        """)

        self.get_chunk_processing_status_stmt = self.cassandra.prepare("""
            SELECT chunk_id, triples_stored, embeddings_stored
            FROM chunk_processing_status
            WHERE user = ? AND document_id = ?
        """)

        self.list_processing_stmt = self.cassandra.prepare("""
            SELECT
                id, document_id, time, flow, collection, tags
            FROM processing
            WHERE user = ?
        """)

    async def document_exists(self, user, id):

        resp = self.cassandra.execute(
            self.test_document_exists_stmt,
            ( user, id )
        )

        # If a row exists, document exists.  It's a cursor, can't just
        # count the length

        for row in resp:
            return True

        return False

    async def add_document(self, document, object_id):

        logger.info(f"Adding document {document.id} {object_id}")

        metadata = [
            (
                v.s.value, v.s.is_uri, v.p.value, v.p.is_uri,
                v.o.value, v.o.is_uri
            )
            for v in document.metadata
        ]

        while True:

            try:

                resp = self.cassandra.execute(
                    self.insert_document_stmt,
                    (
                        document.id, document.user, int(document.time * 1000),
                        document.kind, document.title, document.comments,
                        metadata, document.tags, object_id
                    )
                )

                break

            except Exception as e:

                logger.error("Exception occurred", exc_info=True)
                raise e

        logger.debug("Add complete")

    async def update_document(self, document):

        logger.info(f"Updating document {document.id}")

        metadata = [
            (
                v.s.value, v.s.is_uri, v.p.value, v.p.is_uri,
                v.o.value, v.o.is_uri
            )
            for v in document.metadata
        ]

        while True:

            try:

                resp = self.cassandra.execute(
                    self.update_document_stmt,
                    (
                        int(document.time * 1000), document.title,
                        document.comments, metadata, document.tags,
                        document.user, document.id
                    )
                )

                break

            except Exception as e:

                logger.error("Exception occurred", exc_info=True)
                raise e

        logger.debug("Update complete")

    async def remove_document(self, user, document_id):

        logger.info(f"Removing document {document_id}")

        while True:

            try:

                resp = self.cassandra.execute(
                    self.delete_document_stmt,
                    (
                        user, document_id
                    )
                )

                break

            except Exception as e:

                logger.error("Exception occurred", exc_info=True)
                raise e

        logger.debug("Delete complete")

    async def list_documents(self, user=None, collection=None):
        """
        List documents filtered by user, collection, or both.
        
        Args:
            user: Optional user ID to filter by
            collection: Optional collection ID to filter by
            
        Returns:
            List of DocumentMetadata objects
            
        Raises:
            ValueError: If neither user nor collection is provided
        """
        if not user and not collection:
            raise ValueError("Either user or collection must be provided")

        logger.debug(f"List documents... user={user}, collection={collection}")

        # Case 1: Only user provided (existing behavior)
        if user and not collection:
            while True:
                try:
                    resp = self.cassandra.execute(
                        self.list_document_stmt,
                        (user,)
                    )
                    break
                except Exception as e:
                    logger.error("Exception occurred", exc_info=True)
                    raise e

            lst = [
                DocumentMetadata(
                    id = row[0],
                    user = user,
                    time = int(time.mktime(row[1].timetuple())),
                    kind = row[2],
                    title = row[3],
                    comments = row[4],
                    metadata = [
                        Triple(
                            s=Value(value=m[0], is_uri=m[1]),
                            p=Value(value=m[2], is_uri=m[3]),
                            o=Value(value=m[4], is_uri=m[5])
                        )
                        for m in row[5]
                    ],
                    tags = row[6] if row[6] else [],
                    object_id = row[7],
                )
                for row in resp
            ]

        # Case 2: Both user and collection provided
        elif user and collection:
            # First get document IDs from processing table
            while True:
                try:
                    resp = self.cassandra.execute(
                        self.list_document_ids_by_user_collection_stmt,
                        (user, collection)
                    )
                    break
                except Exception as e:
                    logger.error("Exception occurred", exc_info=True)
                    raise e

            # Deduplicate document IDs (processing table may have multiple entries per document)
            document_ids = set(row[0] for row in resp)
            
            if not document_ids:
                logger.debug("No documents found for user and collection")
                return []

            # Then get document metadata for those IDs
            lst = []
            for doc_id in document_ids:
                try:
                    doc_resp = self.cassandra.execute(
                        self.get_document_stmt,
                        (user, doc_id)
                    )
                    for row in doc_resp:
                        lst.append(
                            DocumentMetadata(
                                id = doc_id,
                                user = user,
                                time = int(time.mktime(row[0].timetuple())),
                                kind = row[1],
                                title = row[2],
                                comments = row[3],
                                metadata = [
                                    Triple(
                                        s=Value(value=m[0], is_uri=m[1]),
                                        p=Value(value=m[2], is_uri=m[3]),
                                        o=Value(value=m[4], is_uri=m[5])
                                    )
                                    for m in row[4]
                                ],
                                tags = row[5] if row[5] else [],
                                object_id = row[6],
                            )
                        )
                except Exception as e:
                    logger.warning(f"Could not get document {doc_id}: {e}")
                    continue

        # Case 3: Only collection provided
        else:  # collection and not user
            # First get document IDs from processing table
            while True:
                try:
                    resp = self.cassandra.execute(
                        self.list_document_ids_by_collection_stmt,
                        (collection,)
                    )
                    break
                except Exception as e:
                    logger.error("Exception occurred", exc_info=True)
                    raise e

            # Build map of doc_id -> user (deduplicate using dict)
            # If same doc_id appears multiple times, last user wins
            doc_id_user_map = {}
            for row in resp:
                doc_id = row[0]
                doc_user = row[1]
                doc_id_user_map[doc_id] = doc_user
            
            if not doc_id_user_map:
                logger.debug("No documents found for collection")
                return []

            # Then get document metadata for those IDs
            lst = []
            for doc_id, doc_user in doc_id_user_map.items():
                try:
                    doc_resp = self.cassandra.execute(
                        self.get_document_stmt,
                        (doc_user, doc_id)
                    )
                    for row in doc_resp:
                        lst.append(
                            DocumentMetadata(
                                id = doc_id,
                                user = doc_user,
                                time = int(time.mktime(row[0].timetuple())),
                                kind = row[1],
                                title = row[2],
                                comments = row[3],
                                metadata = [
                                    Triple(
                                        s=Value(value=m[0], is_uri=m[1]),
                                        p=Value(value=m[2], is_uri=m[3]),
                                        o=Value(value=m[4], is_uri=m[5])
                                    )
                                    for m in row[4]
                                ],
                                tags = row[5] if row[5] else [],
                                object_id = row[6],
                            )
                        )
                except Exception as e:
                    logger.warning(f"Could not get document {doc_id}: {e}")
                    continue

        logger.debug("Done")

        return lst

    async def get_document(self, user, id):

        logger.debug("Get document")

        while True:

            try:

                resp = self.cassandra.execute(
                    self.get_document_stmt,
                    (user, id)
                )

                break

            except Exception as e:
                logger.error("Exception occurred", exc_info=True)
                raise e


        for row in resp:
            doc = DocumentMetadata(
                id = id,
                user = user,
                time = int(time.mktime(row[0].timetuple())),
                kind = row[1],
                title = row[2],
                comments = row[3],
                metadata = [
                    Triple(
                        s=Value(value=m[0], is_uri=m[1]),
                        p=Value(value=m[2], is_uri=m[3]),
                        o=Value(value=m[4], is_uri=m[5])
                    )
                    for m in row[4]
                ] if row[4] else [],
                tags = row[5] if row[5] else [],
                object_id = row[6],
            )

            logger.debug("Done")
            return doc

        raise RuntimeError("No such document row?")

    async def get_document_object_id(self, user, id):

        logger.debug("Get document obj ID")

        while True:

            try:

                resp = self.cassandra.execute(
                    self.get_document_stmt,
                    (user, id)
                )

                break

            except Exception as e:
                logger.error("Exception occurred", exc_info=True)
                raise e


        for row in resp:
            logger.debug("Done")
            return row[6]

        raise RuntimeError("No such document row?")

    async def processing_exists(self, user, id):

        resp = self.cassandra.execute(
            self.test_processing_exists_stmt,
            ( user, id )
        )

        # If a row exists, document exists.  It's a cursor, can't just
        # count the length

        for row in resp:
            return True

        return False

    async def add_processing(self, processing):

        logger.info(f"Adding processing {processing.id}")

        while True:

            try:

                resp = self.cassandra.execute(
                    self.insert_processing_stmt,
                    (
                        processing.id, processing.document_id,
                        int(processing.time * 1000), processing.flow,
                        processing.user, processing.collection,
                        processing.tags
                    )
                )

                break

            except Exception as e:

                logger.error("Exception occurred", exc_info=True)
                raise e

        logger.debug("Add complete")

    async def remove_processing(self, user, processing_id):

        logger.info(f"Removing processing {processing_id}")

        while True:

            try:

                resp = self.cassandra.execute(
                    self.delete_processing_stmt,
                    (
                        user, processing_id
                    )
                )

                break

            except Exception as e:

                logger.error("Exception occurred", exc_info=True)
                raise e

        logger.debug("Delete complete")

    async def list_processing(self, user):

        logger.debug("List processing objects")

        while True:

            try:

                resp = self.cassandra.execute(
                    self.list_processing_stmt,
                    (user,)
                )

                break

            except Exception as e:
                logger.error("Exception occurred", exc_info=True)
                raise e


        lst = [
            ProcessingMetadata(
                id = row[0],
                document_id = row[1],
                time = int(time.mktime(row[2].timetuple())),
                flow = row[3],
                user = user,
                collection = row[4],
                tags = row[5] if row[5] else [],
            )
            for row in resp
        ]

        logger.debug("Done")

        return lst

    # ---> Librarian.get_document_status > [LibraryTableStore.get_document_status] > returns (status, embedding_count, chunk_count, last_updated)
    async def get_document_status(self, user, document_id, collection=None):
        """
        Query document embeddings status from the knowledge keyspace.
        Returns status information about document processing completion.
        
        Args:
            user: User identifier
            document_id: Document identifier
            collection: Optional collection name (for future filtering)
            
        Returns:
            dict with keys: status, embedding_count, chunk_count, last_updated
        """
        logger.debug(f"Getting document status for {user}/{document_id}")

        # Check if document exists in library
        doc_exists = await self.document_exists(user, document_id)
        
        if not doc_exists:
            return {
                "status": "not_found",
                "embedding_count": 0,
                "chunk_count": 0,
                "last_updated": 0,
            }

        # Query the knowledge keyspace for document embeddings
        try:
            resp = self.cassandra.execute(
                self.get_document_embeddings_status_stmt,
                (user, document_id)
            )

            total_chunks = 0
            total_embeddings = 0
            latest_time = 0

            for row in resp:
                # row[0] = id (uuid)
                # row[1] = time (timestamp)
                # row[2] = chunks (list of tuples)
                
                if row[1]:
                    row_time = int(row[1].timestamp() * 1000)
                    if row_time > latest_time:
                        latest_time = row_time
                
                if row[2]:
                    for chunk in row[2]:
                        total_chunks += 1
                        # chunk is (blob, list of vectors)
                        if len(chunk) > 1 and chunk[1]:
                            total_embeddings += len(chunk[1])

            if total_embeddings > 0:
                status = "completed"
            else:
                # Check if there's a processing record
                processing_exists = await self._has_processing_for_document(
                    user, document_id
                )
                if processing_exists:
                    status = "processing"
                else:
                    status = "pending"

            return {
                "status": status,
                "embedding_count": total_embeddings,
                "chunk_count": total_chunks,
                "last_updated": latest_time,
            }

        except Exception as e:
            logger.error(f"Error querying document status: {e}", exc_info=True)
            # If knowledge keyspace doesn't exist yet, return pending
            return {
                "status": "pending",
                "embedding_count": 0,
                "chunk_count": 0,
                "last_updated": 0,
            }

    async def _has_processing_for_document(self, user, document_id):
        """Check if any processing record exists for this document"""
        try:
            # List all processing and filter by document_id
            processing_list = await self.list_processing(user)
            for proc in processing_list:
                if proc.document_id == document_id:
                    return True
            return False
        except Exception:
            return False



    # Collection management methods

    async def ensure_collection_exists(self, user, collection):
        """Ensure collection metadata record exists, create if not"""
        try:
            resp = await asyncio.get_event_loop().run_in_executor(
                None, self.cassandra.execute, self.collection_exists_stmt, [user, collection]
            )
            if resp:
                return
            import datetime
            now = datetime.datetime.now()
            await asyncio.get_event_loop().run_in_executor(
                None, self.cassandra.execute, self.insert_collection_stmt,
                [user, collection, collection, "", set(), now, now]
            )
            logger.debug(f"Created collection metadata for {user}/{collection}")
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
            raise

    async def list_collections(self, user, tag_filter=None):
        """List collections for a user, optionally filtered by tags"""
        try:
            resp = await asyncio.get_event_loop().run_in_executor(
                None, self.cassandra.execute, self.list_collections_stmt, [user]
            )
            collections = []
            for row in resp:
                collection_data = {
                    "user": user,
                    "collection": row[0],
                    "name": row[1] or row[0],
                    "description": row[2] or "",
                    "tags": list(row[3]) if row[3] else [],
                    "created_at": row[4].isoformat() if row[4] else "",
                    "updated_at": row[5].isoformat() if row[5] else ""
                }
                if tag_filter:
                    collection_tags = set(collection_data["tags"])
                    filter_tags = set(tag_filter)
                    if not filter_tags.intersection(collection_tags):
                        continue
                collections.append(collection_data)
            return collections
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            raise

    async def update_collection(self, user, collection, name=None, description=None, tags=None):
        """Update collection metadata"""
        try:
            resp = await asyncio.get_event_loop().run_in_executor(
                None, self.cassandra.execute, self.get_collection_stmt, [user, collection]
            )
            if not resp:
                raise RequestError(f"Collection {collection} not found")
            row = resp.one()
            current_name = row[1] or collection
            current_description = row[2] or ""
            current_tags = set(row[3]) if row[3] else set()
            new_name = name if name is not None else current_name
            new_description = description if description is not None else current_description
            new_tags = set(tags) if tags is not None else current_tags
            import datetime
            now = datetime.datetime.now()
            await asyncio.get_event_loop().run_in_executor(
                None, self.cassandra.execute, self.update_collection_stmt,
                [new_name, new_description, new_tags, now, user, collection]
            )
            return {
                "user": user, "collection": collection, "name": new_name,
                "description": new_description, "tags": list(new_tags),
                "updated_at": now.isoformat()
            }
        except Exception as e:
            logger.error(f"Error updating collection: {e}")
            raise

    async def delete_collection(self, user, collection):
        """Delete collection metadata record"""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, self.cassandra.execute, self.delete_collection_stmt, [user, collection]
            )
            logger.debug(f"Deleted collection metadata for {user}/{collection}")
        except Exception as e:
            logger.error(f"Error deleting collection metadata: {e}")
            raise

    async def get_collection(self, user, collection):
        """Get collection metadata"""
        try:
            resp = await asyncio.get_event_loop().run_in_executor(
                None, self.cassandra.execute, self.get_collection_stmt, [user, collection]
            )
            if not resp:
                return None
            row = resp.one()
            return {
                "user": user, "collection": row[0], "name": row[1] or row[0],
                "description": row[2] or "", "tags": list(row[3]) if row[3] else [],
                "created_at": row[4].isoformat() if row[4] else "",
                "updated_at": row[5].isoformat() if row[5] else ""
            }
        except Exception as e:
            logger.error(f"Error getting collection: {e}")
            raise

    async def create_collection(self, user, collection, name=None, description=None, tags=None):
        """Create a new collection metadata record"""
        try:
            import datetime
            now = datetime.datetime.now()

            # Set defaults for optional parameters
            name = name if name is not None else collection
            description = description if description is not None else ""
            tags = tags if tags is not None else set()

            await asyncio.get_event_loop().run_in_executor(
                None, self.cassandra.execute, self.insert_collection_stmt,
                [user, collection, name, description, tags, now, now]
            )

            logger.info(f"Created collection {user}/{collection}")

            # Return the created collection data
            return {
                "user": user,
                "collection": collection,
                "name": name,
                "description": description,
                "tags": list(tags) if isinstance(tags, set) else tags,
                "created_at": now.isoformat(),
                "updated_at": now.isoformat()
            }
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise

    async def init_chunk_progress(self, user, document_id, total_chunks):
        """Initialize chunk progress tracking for a document"""
        when = int(time.time() * 1000)
        await asyncio.get_event_loop().run_in_executor(
            None, self.cassandra.execute,
            self.init_chunk_progress_stmt,
            (user, document_id, total_chunks, when)
        )

    async def mark_chunk_triples_stored(self, user, document_id, chunk_id):
        """Mark a specific chunk_id as having triples stored (idempotent)"""
        when = int(time.time() * 1000)
        await asyncio.get_event_loop().run_in_executor(
            None, self.cassandra.execute,
            self.mark_chunk_triples_stored_stmt,
            (user, document_id, chunk_id, when)
        )

    async def mark_chunk_embeddings_stored(self, user, document_id, chunk_id):
        """Mark a specific chunk_id as having embeddings stored (idempotent)"""
        when = int(time.time() * 1000)
        await asyncio.get_event_loop().run_in_executor(
            None, self.cassandra.execute,
            self.mark_chunk_embeddings_stored_stmt,
            (when, user, document_id, chunk_id)
        )

    async def increment_triples_stored(self, user, document_id):
        """Increment triples_stored counter (DEPRECATED: use mark_chunk_triples_stored)"""
        # Read current value, increment, write back
        progress = await self.get_chunk_progress(user, document_id)
        
        # If row doesn't exist, initialize it first (race condition protection)
        if progress["total_chunks"] == 0:
            logger.warning(f"Chunk progress row doesn't exist for {user}/{document_id}, initializing with total_chunks=0")
            await self.init_chunk_progress(user, document_id, 0)
            progress = await self.get_chunk_progress(user, document_id)
        
        new_value = progress["triples_stored"] + 1
        when = int(time.time() * 1000)
        await asyncio.get_event_loop().run_in_executor(
            None, self.cassandra.execute,
            self.increment_triples_stored_stmt,
            (new_value, when, user, document_id)
        )

    async def increment_embeddings_stored(self, user, document_id):
        """Increment embeddings_stored counter (DEPRECATED: use mark_chunk_embeddings_stored)"""
        # Read current value, increment, write back
        progress = await self.get_chunk_progress(user, document_id)
        
        # If row doesn't exist, initialize it first (race condition protection)
        if progress["total_chunks"] == 0:
            logger.warning(f"Chunk progress row doesn't exist for {user}/{document_id}, initializing with total_chunks=0")
            await self.init_chunk_progress(user, document_id, 0)
            progress = await self.get_chunk_progress(user, document_id)
        
        new_value = progress["embeddings_stored"] + 1
        when = int(time.time() * 1000)
        await asyncio.get_event_loop().run_in_executor(
            None, self.cassandra.execute,
            self.increment_embeddings_stored_stmt,
            (new_value, when, user, document_id)
        )

    async def increment_failed_chunks(self, user, document_id):
        """Increment failed_chunks counter"""
        # Read current value, increment, write back
        progress = await self.get_chunk_progress(user, document_id)
        new_value = progress["failed_chunks"] + 1
        when = int(time.time() * 1000)
        await asyncio.get_event_loop().run_in_executor(
            None, self.cassandra.execute,
            self.increment_failed_chunks_stmt,
            (new_value, when, user, document_id)
        )

    async def get_chunk_progress(self, user, document_id):
        """Get chunk progress for a document"""
        # Get aggregate counts
        resp = await asyncio.get_event_loop().run_in_executor(
            None, self.cassandra.execute,
            self.get_chunk_progress_stmt,
            (user, document_id)
        )
        
        rows = list(resp)
        if not rows:
            return {
                "total_chunks": 0,
                "triples_stored": 0,
                "embeddings_stored": 0,
                "failed_chunks": 0,
                "last_updated": 0,
            }
        
        row = rows[0]
        total_chunks = row[0] or 0
        
        # Get unique chunk counts from chunk_processing_status
        status_resp = await asyncio.get_event_loop().run_in_executor(
            None, self.cassandra.execute,
            self.get_chunk_processing_status_stmt,
            (user, document_id)
        )
        
        unique_triples_stored = sum(1 for status_row in status_resp if status_row[1])  # triples_stored column
        unique_embeddings_stored = sum(1 for status_row in status_resp if status_row[2])  # embeddings_stored column
        
        # Use unique counts if available, otherwise fall back to aggregate counts
        triples_stored = unique_triples_stored if unique_triples_stored > 0 else (row[1] or 0)
        embeddings_stored = unique_embeddings_stored if unique_embeddings_stored > 0 else (row[2] or 0)
        
        return {
            "total_chunks": total_chunks,
            "triples_stored": min(triples_stored, total_chunks) if total_chunks > 0 else 0,
            "embeddings_stored": min(embeddings_stored, total_chunks) if total_chunks > 0 else 0,
            "failed_chunks": row[3] or 0,
            "last_updated": int(row[4].timestamp() * 1000) if row[4] else 0,
        }
