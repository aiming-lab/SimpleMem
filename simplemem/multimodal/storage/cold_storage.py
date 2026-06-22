"""
Cold Storage Manager for Omni-Memory.

Manages storage of heavy multimodal data (images, audio, video) with
pointer-based access for lazy loading.
"""

import os
import shutil
import hashlib
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
from datetime import datetime
import json

from simplemem.multimodal.core.config import StorageConfig
from simplemem.multimodal.core.mau import ModalityType

logger = logging.getLogger(__name__)


class ColdStorageManager:
    """
    Cold Storage Manager for heavy multimodal data.

    Design Principles:
    1. Heavy data (images, audio, video) stored externally
    2. Only pointers (URIs) kept in memory
    3. Organized by date and modality for efficient management
    4. Supports local filesystem and S3 backends
    """

    def __init__(self, config: Optional[StorageConfig] = None):
        self.config = config or StorageConfig()
        self.base_dir = Path(self.config.cold_storage_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Initialize S3 client if needed
        self._s3_client = None
        if self.config.use_s3:
            self._init_s3()

        # File type mappings
        self._modality_extensions = {
            ModalityType.VISUAL: [".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"],
            ModalityType.AUDIO: [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"],
            ModalityType.VIDEO: [".mp4", ".avi", ".mkv", ".mov", ".webm"],
            ModalityType.TEXT: [".txt", ".json", ".md"],
        }

    def _init_s3(self):
        """Initialize S3 client for cloud storage."""
        try:
            import boto3
            self._s3_client = boto3.client('s3')
            logger.info(f"S3 client initialized for bucket: {self.config.s3_bucket}")
        except ImportError:
            logger.warning("boto3 not installed, S3 storage disabled")
            self.config.use_s3 = False

    def _generate_path(
        self,
        modality: ModalityType,
        extension: str,
        session_id: Optional[str] = None,
        timestamp: Optional[float] = None
    ) -> Path:
        """Generate organized storage path."""
        parts = []

        if self.config.organize_by_date:
            dt = datetime.fromtimestamp(timestamp) if timestamp else datetime.now()
            parts.append(dt.strftime("%Y/%m/%d"))

        if self.config.organize_by_modality:
            parts.append(modality.value)

        if session_id:
            parts.append(session_id)

        # Generate unique filename
        ts = timestamp or datetime.now().timestamp()
        hash_part = hashlib.md5(f"{ts}".encode()).hexdigest()[:8]
        filename = f"{int(ts * 1000)}_{hash_part}{extension}"

        return self.base_dir / "/".join(parts) / filename

    def store(
        self,
        data: Union[bytes, str, Path],
        modality: ModalityType,
        extension: str = "",
        session_id: Optional[str] = None,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store data in cold storage.

        Args:
            data: Raw data (bytes, file path, or string content)
            modality: Type of modality
            extension: File extension (with dot)
            session_id: Optional session identifier
            timestamp: Optional timestamp for organization
            metadata: Optional metadata to store alongside

        Returns:
            Storage pointer (URI) for the stored data
        """
        # Determine extension if not provided
        if not extension:
            if isinstance(data, (str, Path)) and Path(data).exists():
                extension = Path(data).suffix
            else:
                extension = self._default_extension(modality)

        # Generate storage path
        storage_path = self._generate_path(modality, extension, session_id, timestamp)
        storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Write data
        if isinstance(data, bytes):
            storage_path.write_bytes(data)
        elif isinstance(data, (str, Path)) and Path(data).exists():
            shutil.copy2(str(data), str(storage_path))
        elif isinstance(data, str):
            storage_path.write_text(data, encoding='utf-8')
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        # Store metadata if provided
        if metadata:
            meta_path = storage_path.with_suffix(storage_path.suffix + '.meta.json')
            meta_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding='utf-8')

        # Upload to S3 if enabled
        if self.config.use_s3:
            s3_key = self._upload_to_s3(storage_path)
            return f"s3://{self.config.s3_bucket}/{s3_key}"

        return str(storage_path)

    def _upload_to_s3(self, local_path: Path) -> str:
        """Upload file to S3 and return key."""
        relative_path = local_path.relative_to(self.base_dir)
        s3_key = f"{self.config.s3_prefix}{relative_path}"

        self._s3_client.upload_file(str(local_path), self.config.s3_bucket, s3_key)
        logger.debug(f"Uploaded to S3: {s3_key}")

        return s3_key

    def retrieve(self, pointer: str) -> Optional[bytes]:
        """
        Retrieve data from cold storage.

        Args:
            pointer: Storage pointer (local path or S3 URI)

        Returns:
            Raw bytes of the stored data
        """
        if pointer.startswith("s3://"):
            return self._retrieve_from_s3(pointer)

        path = Path(pointer)
        if path.exists():
            return path.read_bytes()

        logger.warning(f"File not found: {pointer}")
        return None

    def _retrieve_from_s3(self, s3_uri: str) -> Optional[bytes]:
        """Retrieve file from S3."""
        if not self._s3_client:
            logger.error("S3 client not initialized")
            return None

        # Parse S3 URI
        parts = s3_uri.replace("s3://", "").split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""

        try:
            response = self._s3_client.get_object(Bucket=bucket, Key=key)
            return response['Body'].read()
        except Exception as e:
            logger.error(f"S3 retrieval error: {e}")
            return None

    def retrieve_to_file(self, pointer: str, target_path: Optional[str] = None) -> Optional[str]:
        """
        Retrieve data and save to local file.

        Args:
            pointer: Storage pointer
            target_path: Optional target path (uses temp if not provided)

        Returns:
            Path to the retrieved file
        """
        data = self.retrieve(pointer)
        if data is None:
            return None

        if target_path is None:
            import tempfile
            suffix = Path(pointer).suffix
            fd, target_path = tempfile.mkstemp(suffix=suffix)
            os.close(fd)

        Path(target_path).write_bytes(data)
        return target_path

    def get_metadata(self, pointer: str) -> Optional[Dict[str, Any]]:
        """Get metadata associated with stored data."""
        if pointer.startswith("s3://"):
            # TODO: Implement S3 metadata retrieval
            return None

        meta_path = Path(pointer).with_suffix(Path(pointer).suffix + '.meta.json')
        if meta_path.exists():
            return json.loads(meta_path.read_text(encoding='utf-8'))
        return None

    def delete(self, pointer: str) -> bool:
        """
        Delete data from cold storage.

        Args:
            pointer: Storage pointer

        Returns:
            True if deleted successfully
        """
        if pointer.startswith("s3://"):
            return self._delete_from_s3(pointer)

        path = Path(pointer)
        if path.exists():
            path.unlink()
            # Also delete metadata if exists
            meta_path = path.with_suffix(path.suffix + '.meta.json')
            if meta_path.exists():
                meta_path.unlink()
            return True

        return False

    def _delete_from_s3(self, s3_uri: str) -> bool:
        """Delete file from S3."""
        if not self._s3_client:
            return False

        parts = s3_uri.replace("s3://", "").split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""

        try:
            self._s3_client.delete_object(Bucket=bucket, Key=key)
            return True
        except Exception as e:
            logger.error(f"S3 deletion error: {e}")
            return False

    def exists(self, pointer: str) -> bool:
        """Check if data exists in storage."""
        if pointer.startswith("s3://"):
            return self._exists_in_s3(pointer)
        return Path(pointer).exists()

    def _exists_in_s3(self, s3_uri: str) -> bool:
        """Check if file exists in S3."""
        if not self._s3_client:
            return False

        parts = s3_uri.replace("s3://", "").split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""

        try:
            self._s3_client.head_object(Bucket=bucket, Key=key)
            return True
        except:
            return False

    def get_size(self, pointer: str) -> int:
        """Get size of stored data in bytes."""
        if pointer.startswith("s3://"):
            # TODO: Implement S3 size retrieval
            return 0

        path = Path(pointer)
        if path.exists():
            return path.stat().st_size
        return 0

    def list_files(
        self,
        modality: Optional[ModalityType] = None,
        session_id: Optional[str] = None,
        date_prefix: Optional[str] = None,
    ) -> List[str]:
        """
        List files in cold storage.

        Args:
            modality: Filter by modality
            session_id: Filter by session
            date_prefix: Filter by date (YYYY/MM/DD format)

        Returns:
            List of storage pointers
        """
        search_path = self.base_dir

        if date_prefix:
            search_path = search_path / date_prefix
        if modality and self.config.organize_by_modality:
            search_path = search_path / modality.value
        if session_id:
            search_path = search_path / session_id

        if not search_path.exists():
            return []

        files = []
        for f in search_path.rglob("*"):
            if f.is_file() and not f.suffix.endswith('.meta.json'):
                files.append(str(f))

        return files

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        total_size = 0
        file_count = 0
        by_modality = {}

        for f in self.base_dir.rglob("*"):
            if f.is_file() and not f.suffix.endswith('.meta.json'):
                size = f.stat().st_size
                total_size += size
                file_count += 1

                # Count by modality
                for modality, exts in self._modality_extensions.items():
                    if f.suffix.lower() in exts:
                        by_modality[modality.value] = by_modality.get(modality.value, 0) + 1
                        break

        return {
            "total_size_bytes": total_size,
            "total_size_gb": total_size / (1024 ** 3),
            "file_count": file_count,
            "by_modality": by_modality,
            "storage_path": str(self.base_dir),
        }

    def cleanup_old_files(self, days_old: int = 30) -> int:
        """
        Clean up files older than specified days.

        Returns:
            Number of files deleted
        """
        if not self.config.auto_cleanup_enabled:
            logger.warning("Auto cleanup is disabled")
            return 0

        from datetime import timedelta
        cutoff = datetime.now() - timedelta(days=days_old)
        deleted = 0

        for f in self.base_dir.rglob("*"):
            if f.is_file():
                mtime = datetime.fromtimestamp(f.stat().st_mtime)
                if mtime < cutoff:
                    f.unlink()
                    deleted += 1

        logger.info(f"Cleaned up {deleted} files older than {days_old} days")
        return deleted

    def _default_extension(self, modality: ModalityType) -> str:
        """Get default extension for modality."""
        defaults = {
            ModalityType.VISUAL: ".jpg",
            ModalityType.AUDIO: ".wav",
            ModalityType.VIDEO: ".mp4",
            ModalityType.TEXT: ".txt",
            ModalityType.MULTIMODAL: ".bin",
        }
        return defaults.get(modality, ".bin")
