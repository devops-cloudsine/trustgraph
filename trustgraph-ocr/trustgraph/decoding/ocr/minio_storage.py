"""
MinIO storage utility for OCR image storage.

Provides a unified interface for storing OCR-generated images in MinIO,
replacing the local filesystem storage. Images are stored in a bucket
with a path structure of: <bucket>/<document-id>/<filename>

Environment variables:
- MINIO_ENDPOINT: MinIO server endpoint (default: minio:9000)
- MINIO_ACCESS_KEY: Access key (default: minioadmin)
- MINIO_SECRET_KEY: Secret key (default: minioadmin)
- MINIO_BUCKET: Bucket name for OCR images (default: ocr-images)
- MINIO_SECURE: Use HTTPS (default: false)
"""

import logging
import os
import re
from io import BytesIO
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import minio client
try:
    from minio import Minio
    from minio.error import S3Error
    MINIO_AVAILABLE = True
except ImportError:
    MINIO_AVAILABLE = False
    Minio = None
    S3Error = Exception
    logger.warning("minio package not available - falling back to local storage")


class MinioStorage:
    """
    MinIO storage handler for OCR images.
    
    Provides methods to:
    - Initialize MinIO client and create bucket
    - Upload images to MinIO
    - Get the local filesystem path (for containers that mount minio-data volume)
    """
    
    def __init__(
        self,
        endpoint: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        bucket: Optional[str] = None,
        secure: Optional[bool] = None,
        local_mount_path: Optional[str] = None,
    ):
        """
        Initialize MinIO storage.
        
        Args:
            endpoint: MinIO server endpoint (host:port)
            access_key: MinIO access key
            secret_key: MinIO secret key
            bucket: Bucket name for OCR images
            secure: Use HTTPS connection
            local_mount_path: Local path where minio-data volume is mounted
                             (for generating filesystem paths for vLLM)
        """
        # Configuration from environment or defaults
        self.endpoint = endpoint or os.environ.get("MINIO_ENDPOINT", "minio:9000")
        self.access_key = access_key or os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
        self.secret_key = secret_key or os.environ.get("MINIO_SECRET_KEY", "minioadmin")
        self.bucket = bucket or os.environ.get("MINIO_BUCKET", "ocr-images")
        self.secure = secure if secure is not None else (
            os.environ.get("MINIO_SECURE", "false").lower() == "true"
        )
        self.local_mount_path = local_mount_path or os.environ.get(
            "MINIO_LOCAL_MOUNT_PATH", "/minio_data"
        )
        
        self.client: Optional[Minio] = None
        self._initialized = False
        
    # // ---> OCR Processor init > [initialize] > create MinIO client and bucket
    def initialize(self) -> bool:
        """
        Initialize MinIO client and ensure bucket exists.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if not MINIO_AVAILABLE:
            logger.error("MinIO package not available - cannot initialize storage")
            return False
            
        if self._initialized:
            return True
            
        try:
            logger.info(f"Initializing MinIO client: endpoint={self.endpoint}, bucket={self.bucket}")
            
            self.client = Minio(
                self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                secure=self.secure,
            )
            
            # Create bucket if it doesn't exist
            if not self.client.bucket_exists(self.bucket):
                logger.info(f"Creating MinIO bucket: {self.bucket}")
                self.client.make_bucket(self.bucket)
            else:
                logger.debug(f"MinIO bucket already exists: {self.bucket}")
                
            self._initialized = True
            logger.info("MinIO storage initialized successfully")
            return True
            
        except S3Error as e:
            # Handle race condition: bucket might already exist
            if hasattr(e, 'code') and e.code in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
                logger.debug(f"MinIO bucket already exists (confirmed): {self.bucket}")
                self._initialized = True
                return True
            logger.error(f"MinIO S3 error during initialization: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize MinIO storage: {e}")
            return False
    
    # // ---> on_message > [save_document] > upload original document bytes to MinIO bucket
    def save_document(
        self,
        doc_id: str,
        filename: str,
        document_data: bytes,
        content_type: str = "application/octet-stream",
    ) -> Optional[str]:
        """
        Save original document to MinIO storage.
        
        Args:
            doc_id: Document identifier (used as subdirectory)
            filename: Document filename
            document_data: Document binary data
            content_type: MIME type of the document
            
        Returns:
            The object path in MinIO (doc_id/filename), or None on failure
        """
        if not self._initialized:
            if not self.initialize():
                logger.error("Cannot save document - MinIO not initialized")
                return None
                
        # Sanitize doc_id for use as object path
        safe_doc_id = self._safe_id(doc_id)
        object_name = f"{safe_doc_id}/{filename}"
        
        try:
            data_stream = BytesIO(document_data)
            data_length = len(document_data)
            
            logger.debug(f"Uploading document to MinIO: {self.bucket}/{object_name} ({data_length} bytes)")
            
            self.client.put_object(
                self.bucket,
                object_name,
                data_stream,
                data_length,
                content_type=content_type,
            )
            
            logger.info(f"Successfully saved document: {object_name}")
            return object_name
            
        except S3Error as e:
            logger.error(f"MinIO S3 error uploading document {object_name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to upload document {object_name}: {e}")
            return None

    # // ---> on_message > [save_image] > upload image bytes to MinIO bucket
    def save_image(
        self,
        doc_id: str,
        filename: str,
        image_data: bytes,
        content_type: str = "image/png",
    ) -> Optional[str]:
        """
        Save image to MinIO storage.
        
        Args:
            doc_id: Document identifier (used as subdirectory)
            filename: Image filename
            image_data: Image binary data
            content_type: MIME type of the image
            
        Returns:
            The object path in MinIO (bucket/doc_id/filename), or None on failure
        """
        if not self._initialized:
            if not self.initialize():
                logger.error("Cannot save image - MinIO not initialized")
                return None
                
        # Sanitize doc_id for use as object path
        safe_doc_id = self._safe_id(doc_id)
        object_name = f"{safe_doc_id}/{filename}"
        
        try:
            data_stream = BytesIO(image_data)
            data_length = len(image_data)
            
            logger.debug(f"Uploading to MinIO: {self.bucket}/{object_name} ({data_length} bytes)")
            
            self.client.put_object(
                self.bucket,
                object_name,
                data_stream,
                data_length,
                content_type=content_type,
            )
            
            logger.debug(f"Successfully uploaded: {object_name}")
            return object_name
            
        except S3Error as e:
            logger.error(f"MinIO S3 error uploading {object_name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to upload {object_name}: {e}")
            return None
    
    # // ---> on_message > [get_local_path] > get filesystem path for mounted minio-data volume
    def get_local_path(self, object_name: str) -> str:
        """
        Get the local filesystem path for an object.
        
        This path is valid in containers that mount the minio-data volume.
        Used for vLLM's --allowed-local-media-path access.
        
        Args:
            object_name: Object name returned by save_image()
            
        Returns:
            Local filesystem path (e.g., /minio_data/ocr-images/doc123/page_1.png)
        """
        return os.path.join(self.local_mount_path, self.bucket, object_name)
    
    # // ---> on_message > [get_full_object_path] > get full bucket/object path
    def get_full_object_path(self, object_name: str) -> str:
        """
        Get the full object path including bucket.
        
        Args:
            object_name: Object name returned by save_image()
            
        Returns:
            Full path (e.g., ocr-images/doc123/page_1.png)
        """
        return f"{self.bucket}/{object_name}"
    
    # // ---> save_image > [_safe_id] > sanitize document ID for use as object path
    @staticmethod
    def _safe_id(value: str) -> str:
        """Sanitize a value for use as an object path component."""
        if value is None:
            return "unknown"
        return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))
    
    # // ---> on_message > [read_image] > read image from MinIO (for base64 conversion)
    def read_image(self, object_name: str) -> Optional[bytes]:
        """
        Read image data from MinIO.
        
        Args:
            object_name: Object name (doc_id/filename)
            
        Returns:
            Image bytes or None on failure
        """
        if not self._initialized:
            if not self.initialize():
                logger.error("Cannot read image - MinIO not initialized")
                return None
                
        try:
            response = self.client.get_object(self.bucket, object_name)
            data = response.read()
            response.close()
            response.release_conn()
            return data
        except S3Error as e:
            logger.error(f"MinIO S3 error reading {object_name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to read {object_name}: {e}")
            return None

    # // ---> on_message > [delete_document_images] > delete all images for a document
    def delete_document_images(self, doc_id: str) -> bool:
        """
        Delete all images for a document (cleanup).
        
        Args:
            doc_id: Document identifier
            
        Returns:
            True if successful, False otherwise
        """
        if not self._initialized:
            if not self.initialize():
                return False
                
        safe_doc_id = self._safe_id(doc_id)
        prefix = f"{safe_doc_id}/"
        
        try:
            objects = self.client.list_objects(self.bucket, prefix=prefix, recursive=True)
            for obj in objects:
                self.client.remove_object(self.bucket, obj.object_name)
                logger.debug(f"Deleted: {obj.object_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete images for {doc_id}: {e}")
            return False


# Global storage instance (lazy initialization)
_storage_instance: Optional[MinioStorage] = None


def get_minio_storage() -> MinioStorage:
    """
    Get the global MinIO storage instance.
    
    Returns:
        MinioStorage instance (creates one if needed)
    """
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = MinioStorage()
    return _storage_instance



