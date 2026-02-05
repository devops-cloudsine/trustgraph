
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.query import BatchStatement, SimpleStatement
from ssl import SSLContext, PROTOCOL_TLSv1_2
import os
import logging

# Global list to track clusters for cleanup
_active_clusters = []

logger = logging.getLogger(__name__)

class KnowledgeGraph:

    def __init__(
            self, hosts=None,
            keyspace="trustgraph", username=None, password=None
    ):

        if hosts is None:
            hosts = ["localhost"]

        self.keyspace = keyspace
        self.username = username

        # Optimized multi-table schema with collection deletion support
        self.subject_table = "triples_s"
        self.po_table = "triples_p"
        self.object_table = "triples_o"
        self.collection_table = "triples_collection"  # For SPO queries and deletion
        self.collection_metadata_table = "collection_metadata"  # For tracking which collections exist

        if username and password:
            ssl_context = SSLContext(PROTOCOL_TLSv1_2)
            auth_provider = PlainTextAuthProvider(username=username, password=password)
            self.cluster = Cluster(hosts, auth_provider=auth_provider, ssl_context=ssl_context)
        else:
            self.cluster = Cluster(hosts)
        self.session = self.cluster.connect()

        # Track this cluster globally
        _active_clusters.append(self.cluster)

        self.init()
        self.prepare_statements()

    def clear(self):

        self.session.execute(f"""
            drop keyspace if exists {self.keyspace};
        """);

        self.init()

    def init(self):

        self.session.execute(f"""
            create keyspace if not exists {self.keyspace}
                with replication = {{
                   'class' : 'SimpleStrategy',
                   'replication_factor' : 1
                }};
        """);

        self.session.set_keyspace(self.keyspace)
        self.init_optimized_schema()


    def init_optimized_schema(self):
        """Initialize optimized multi-table schema for performance"""
        # Table 1: Subject-centric queries (get_s, get_sp, get_os)
        # Compound partition key for optimal data distribution
        self.session.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.subject_table} (
                collection text,
                document_id text,
                s text,
                p text,
                o text,
                PRIMARY KEY ((collection, s), document_id, p, o)
            );
        """);

        # Create index for document_id-based queries
        self.session.execute(f"""
            CREATE INDEX IF NOT EXISTS triples_s_document_id ON {self.subject_table} (document_id);
        """);

        # Table 2: Predicate-Object queries (get_p, get_po) - eliminates ALLOW FILTERING!
        # Compound partition key for optimal data distribution
        self.session.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.po_table} (
                collection text,
                document_id text,
                p text,
                o text,
                s text,
                PRIMARY KEY ((collection, p), document_id, o, s)
            );
        """);

        # Create index for document_id-based queries
        self.session.execute(f"""
            CREATE INDEX IF NOT EXISTS triples_p_document_id ON {self.po_table} (document_id);
        """);

        # Table 3: Object-centric queries (get_o)
        # Compound partition key for optimal data distribution
        self.session.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.object_table} (
                collection text,
                document_id text,
                o text,
                s text,
                p text,
                PRIMARY KEY ((collection, o), document_id, s, p)
            );
        """);

        # Create index for document_id-based queries
        self.session.execute(f"""
            CREATE INDEX IF NOT EXISTS triples_o_document_id ON {self.object_table} (document_id);
        """);

        # Table 4: Collection management and SPO queries (get_spo)
        # Simple partition key enables efficient collection deletion
        self.session.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.collection_table} (
                collection text,
                document_id text,
                s text,
                p text,
                o text,
                PRIMARY KEY (collection, document_id, s, p, o)
            );
        """);

        # Table 5: Collection metadata tracking
        # Tracks which collections exist without polluting triple data
        self.session.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.collection_metadata_table} (
                collection text,
                created_at timestamp,
                PRIMARY KEY (collection)
            );
        """);

        logger.info("Optimized multi-table schema initialized (5 tables)")

    def prepare_statements(self):
        """Prepare statements for optimal performance"""
        # Insert statements for batch operations
        self.insert_subject_stmt = self.session.prepare(
            f"INSERT INTO {self.subject_table} (collection, document_id, s, p, o) VALUES (?, ?, ?, ?, ?)"
        )

        self.insert_po_stmt = self.session.prepare(
            f"INSERT INTO {self.po_table} (collection, document_id, p, o, s) VALUES (?, ?, ?, ?, ?)"
        )

        self.insert_object_stmt = self.session.prepare(
            f"INSERT INTO {self.object_table} (collection, document_id, o, s, p) VALUES (?, ?, ?, ?, ?)"
        )

        self.insert_collection_stmt = self.session.prepare(
            f"INSERT INTO {self.collection_table} (collection, document_id, s, p, o) VALUES (?, ?, ?, ?, ?)"
        )

        # Query statements for optimized access
        # Use triples_collection for get_all since collection is the partition key (fast!)
        self.get_all_stmt = self.session.prepare(
            f"SELECT document_id, s, p, o FROM {self.collection_table} WHERE collection = ? LIMIT ?"
        )

        self.get_s_stmt = self.session.prepare(
            f"SELECT p, o FROM {self.subject_table} WHERE collection = ? AND s = ? LIMIT ?"
        )

        self.get_p_stmt = self.session.prepare(
            f"SELECT document_id, s, o FROM {self.po_table} WHERE collection = ? AND p = ? LIMIT ?"
        )

        self.get_o_stmt = self.session.prepare(
            f"SELECT document_id, s, p FROM {self.object_table} WHERE collection = ? AND o = ? LIMIT ?"
        )

        self.get_sp_stmt = self.session.prepare(
            f"SELECT document_id, o FROM {self.subject_table} WHERE collection = ? AND s = ? AND p = ? LIMIT ? ALLOW FILTERING"
        )

        # Note: get_po and get_os now require ALLOW FILTERING because document_id is between partition key and o/s
        # This is a trade-off for document-specific deletion capability
        self.get_po_stmt = self.session.prepare(
            f"SELECT document_id, s FROM {self.po_table} WHERE collection = ? AND p = ? AND o = ? LIMIT ? ALLOW FILTERING"
        )

        self.get_os_stmt = self.session.prepare(
            f"SELECT document_id, p FROM {self.object_table} WHERE collection = ? AND o = ? AND s = ? LIMIT ? ALLOW FILTERING"
        )

        self.get_spo_stmt = self.session.prepare(
            f"SELECT document_id, s as x FROM {self.collection_table} WHERE collection = ? AND s = ? AND p = ? AND o = ? LIMIT ? ALLOW FILTERING"
        )

        # Delete statements for collection deletion
        self.delete_subject_stmt = self.session.prepare(
            f"DELETE FROM {self.subject_table} WHERE collection = ? AND s = ? AND document_id = ? AND p = ? AND o = ?"
        )

        self.delete_po_stmt = self.session.prepare(
            f"DELETE FROM {self.po_table} WHERE collection = ? AND p = ? AND document_id = ? AND o = ? AND s = ?"
        )

        self.delete_object_stmt = self.session.prepare(
            f"DELETE FROM {self.object_table} WHERE collection = ? AND o = ? AND document_id = ? AND s = ? AND p = ?"
        )

        self.delete_collection_stmt = self.session.prepare(
            f"DELETE FROM {self.collection_table} WHERE collection = ? AND document_id = ? AND s = ? AND p = ? AND o = ?"
        )

        # NEW: Prepare statement for deleting all triples by document_id
        self.delete_by_document_stmt = self.session.prepare(
            f"DELETE FROM {self.collection_table} WHERE collection = ? AND document_id = ?"
        )

        logger.info("Prepared statements initialized for optimal performance (4 tables)")

    def insert(self, collection, s, p, o, document_id):
        """Insert triple with document tracking"""
        # Batch write to all four tables for consistency
        batch = BatchStatement()

        # Insert into subject table
        batch.add(self.insert_subject_stmt, (collection, document_id, s, p, o))

        # Insert into predicate-object table (column order: collection, document_id, p, o, s)
        batch.add(self.insert_po_stmt, (collection, document_id, p, o, s))

        # Insert into object table (column order: collection, document_id, o, s, p)
        batch.add(self.insert_object_stmt, (collection, document_id, o, s, p))

        # Insert into collection table for SPO queries and deletion tracking
        batch.add(self.insert_collection_stmt, (collection, document_id, s, p, o))

        self.session.execute(batch)

    def get_all(self, collection, limit=50):
        # Use collection_table for get_all queries - collection is partition key (fast!)
        return self.session.execute(
            self.get_all_stmt,
            (collection, limit)
        )

    def get_s(self, collection, s, limit=10):
        # Optimized: Direct partition access with (collection, s)
        return self.session.execute(
            self.get_s_stmt,
            (collection, s, limit)
        )

    def get_p(self, collection, p, limit=10):
        # Optimized: Use po_table for direct partition access
        return self.session.execute(
            self.get_p_stmt,
            (collection, p, limit)
        )

    def get_o(self, collection, o, limit=10):
        # Optimized: Use object_table for direct partition access
        return self.session.execute(
            self.get_o_stmt,
            (collection, o, limit)
        )

    def get_sp(self, collection, s, p, limit=10):
        # Optimized: Use subject_table with clustering key access
        return self.session.execute(
            self.get_sp_stmt,
            (collection, s, p, limit)
        )

    def get_po(self, collection, p, o, limit=10):
        # Note: Now requires ALLOW FILTERING because document_id is between partition key and o
        # This is a trade-off for document-specific deletion capability
        return self.session.execute(
            self.get_po_stmt,
            (collection, p, o, limit)
        )

    def get_os(self, collection, o, s, limit=10):
        # Note: Now requires ALLOW FILTERING because document_id is between partition key and s
        # This is a trade-off for document-specific deletion capability
        return self.session.execute(
            self.get_os_stmt,
            (collection, o, s, limit)
        )

    def get_spo(self, collection, s, p, o, limit=10):
        # Optimized: Use collection_table for exact key lookup
        return self.session.execute(
            self.get_spo_stmt,
            (collection, s, p, o, limit)
        )

    def collection_exists(self, collection):
        """Check if collection exists by querying collection_metadata table"""
        try:
            result = self.session.execute(
                f"SELECT collection FROM {self.collection_metadata_table} WHERE collection = %s LIMIT 1",
                (collection,)
            )
            return bool(list(result))
        except Exception as e:
            logger.error(f"Error checking collection existence: {e}")
            return False

    def create_collection(self, collection):
        """Create collection by inserting metadata row"""
        try:
            import datetime
            self.session.execute(
                f"INSERT INTO {self.collection_metadata_table} (collection, created_at) VALUES (%s, %s)",
                (collection, datetime.datetime.now())
            )
            logger.info(f"Created collection metadata for {collection}")
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise e

    def delete_collection(self, collection):
        """Delete all triples for a specific collection

        Uses collection_table to enumerate all triples, then deletes from all 4 tables
        using full partition keys for optimal performance with compound keys.
        """
        # Step 1: Read all triples from collection_table (single partition read)
        rows = self.session.execute(
            f"SELECT document_id, s, p, o FROM {self.collection_table} WHERE collection = %s",
            (collection,)
        )

        # Step 2: Delete each triple from all 4 tables using full partition keys
        # Batch deletions for efficiency
        batch = BatchStatement()
        count = 0

        for row in rows:
            document_id, s, p, o = row.document_id, row.s, row.p, row.o

            # Delete from subject table (partition key: collection, s)
            batch.add(self.delete_subject_stmt, (collection, s, document_id, p, o))

            # Delete from predicate-object table (partition key: collection, p)
            batch.add(self.delete_po_stmt, (collection, p, document_id, o, s))

            # Delete from object table (partition key: collection, o)
            batch.add(self.delete_object_stmt, (collection, o, document_id, s, p))

            # Delete from collection table (partition key: collection only)
            batch.add(self.delete_collection_stmt, (collection, document_id, s, p, o))

            count += 1

            # Execute batch every 100 triples to avoid oversized batches
            if count % 100 == 0:
                self.session.execute(batch)
                batch = BatchStatement()

        # Execute remaining deletions
        if count % 100 != 0:
            self.session.execute(batch)

        # Step 3: Delete collection metadata
        self.session.execute(
            f"DELETE FROM {self.collection_metadata_table} WHERE collection = %s",
            (collection,)
        )

        logger.info(f"Deleted {count} triples from collection {collection}")

    def delete_document_triples(self, collection, document_id):
        """Delete all triples for a specific document from a collection
        
        This is the critical method that enables document-specific deletion.
        Uses document_id index for efficient lookup and deletion.
        
        Args:
            collection: Collection UUID
            document_id: Document ID to delete triples for
            
        Returns:
            int: Number of triples deleted
        """
        logger.info(f"Deleting triples for document {document_id} from collection {collection}")
        
        # Step 1: Query triples_collection for all triples belonging to this document
        # This is efficient because document_id is part of the clustering key
        rows = self.session.execute(
            f"SELECT s, p, o FROM {self.collection_table} "
            f"WHERE collection = %s AND document_id = %s",
            (collection, document_id)
        )
        
        # Step 2: Batch delete from all 4 tables
        # Each triple requires 4 deletes, so batch every 25 triples = 100 statements
        batch = BatchStatement()
        count = 0
        batch_count = 0
        
        for row in rows:
            s, p, o = row.s, row.p, row.o
            
            # Delete from all 4 tables using full partition + clustering keys
            # Note: Order of parameters matches the PRIMARY KEY definition
            batch.add(self.delete_subject_stmt, (collection, s, document_id, p, o))
            batch.add(self.delete_po_stmt, (collection, p, document_id, o, s))
            batch.add(self.delete_object_stmt, (collection, o, document_id, s, p))
            batch.add(self.delete_collection_stmt, (collection, document_id, s, p, o))
            
            count += 1
            batch_count += 4  # 4 statements per triple
            
            # Execute batch every 25 triples (100 statements) to avoid oversized batches
            if batch_count >= 100:
                self.session.execute(batch)
                batch = BatchStatement()
                batch_count = 0
        
        # Execute remaining deletions
        if batch_count > 0:
            self.session.execute(batch)
        
        logger.info(f"Deleted {count} triples for document {document_id} from collection {collection}")
        return count

    def close(self):
        """Close the Cassandra session and cluster connections properly"""
        if hasattr(self, 'session') and self.session:
            self.session.shutdown()
        if hasattr(self, 'cluster') and self.cluster:
            self.cluster.shutdown()
            # Remove from global tracking
            if self.cluster in _active_clusters:
                _active_clusters.remove(self.cluster)
