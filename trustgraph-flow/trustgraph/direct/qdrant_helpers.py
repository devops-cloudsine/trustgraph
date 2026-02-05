"""
Qdrant vector store helper for managing document and graph vectors.
Provides methods to delete vectors by document_id for cascade delete operations.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import logging

logger = logging.getLogger(__name__)


class QdrantVectorManager:
    """
    Helper class for managing Qdrant vector stores, specifically for deletion operations.
    """
    
    def __init__(self, qdrant_uri: str, api_key=None):
        """
        Initialize Qdrant client.
        
        Args:
            qdrant_uri: Qdrant server URI (e.g., "http://qdrant:6333")
            api_key: Optional API key for authentication
        """
        self.client = QdrantClient(url=qdrant_uri, api_key=api_key)
        logger.info(f"QdrantVectorManager initialized with URI: {qdrant_uri}")
    
    def delete_document_vectors(self, user: str, collection: str, document_id: str):
        """
        Delete all document chunk vectors for a specific document.
        
        Args:
            user: User ID
            collection: Collection name
            document_id: Document ID to delete vectors for
            
        Returns:
            Number of vectors deleted (0 if collection doesn't exist or no matches)
        """
        collection_name = f"d_{user}_{collection}"
        
        if not self.client.collection_exists(collection_name):
            logger.debug(f"Collection {collection_name} does not exist, skipping deletion")
            return 0
        
        try:
            # Get collection info before deletion to check point count
            collection_info_before = self.client.get_collection(collection_name)
            points_before = collection_info_before.points_count or 0
            
            # Delete using filter on document_id field
            # Qdrant delete returns UpdateResult with operation_id and status
            result = self.client.delete(
                collection_name=collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                ),
                wait=True  # Wait for operation to complete
            )
            
            # Check collection info after deletion
            collection_info_after = self.client.get_collection(collection_name)
            points_after = collection_info_after.points_count or 0
            deleted_count = points_before - points_after
            
            if deleted_count > 0:
                logger.info(f"Deleted {deleted_count} document vectors from {collection_name} for document {document_id}")
            elif points_after > 0:
                # Vectors remain but weren't deleted - likely old vectors without document_id
                logger.warning(
                    f"No vectors deleted from {collection_name} for document {document_id}. "
                    f"Collection still has {points_after} vectors. "
                    f"These may be old vectors without document_id payload. "
                    f"Consider recreating the collection or reprocessing documents."
                )
            else:
                logger.debug(f"Collection {collection_name} is now empty after deletion")
            
            return deleted_count
        except Exception as e:
            logger.warning(f"Failed to delete document vectors from {collection_name}: {e}")
            return 0
    
    def delete_graph_vectors(self, user: str, collection: str, document_id: str):
        """
        Delete all graph entity vectors for a specific document.
        
        Args:
            user: User ID
            collection: Collection name
            document_id: Document ID to delete vectors for
            
        Returns:
            Number of vectors deleted (0 if collection doesn't exist or no matches)
        """
        collection_name = f"t_{user}_{collection}"
        
        if not self.client.collection_exists(collection_name):
            logger.debug(f"Collection {collection_name} does not exist, skipping deletion")
            return 0
        
        try:
            # Get collection info before deletion to check point count
            collection_info_before = self.client.get_collection(collection_name)
            points_before = collection_info_before.points_count or 0
            
            # Delete using filter on document_id field
            # Qdrant delete returns UpdateResult with operation_id and status
            result = self.client.delete(
                collection_name=collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                ),
                wait=True  # Wait for operation to complete
            )
            
            # Check collection info after deletion
            collection_info_after = self.client.get_collection(collection_name)
            points_after = collection_info_after.points_count or 0
            deleted_count = points_before - points_after
            
            if deleted_count > 0:
                logger.info(f"Deleted {deleted_count} graph vectors from {collection_name} for document {document_id}")
            elif points_after > 0:
                # Vectors remain but weren't deleted - likely old vectors without document_id
                logger.warning(
                    f"No vectors deleted from {collection_name} for document {document_id}. "
                    f"Collection still has {points_after} vectors. "
                    f"These may be old vectors without document_id payload. "
                    f"Consider recreating the collection or reprocessing documents."
                )
            else:
                logger.debug(f"Collection {collection_name} is now empty after deletion")
            
            return deleted_count
        except Exception as e:
            logger.warning(f"Failed to delete graph vectors from {collection_name}: {e}")
            return 0

