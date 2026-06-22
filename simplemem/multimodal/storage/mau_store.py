"""
MAU Store - Storage for Multimodal Atomic Units.

Provides efficient storage and retrieval of MAU metadata while keeping
heavy data in cold storage.
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Iterator
from datetime import datetime
import threading

from ..core.mau import MultimodalAtomicUnit, ModalityType
from ..core.config import StorageConfig

logger = logging.getLogger(__name__)


class MAUStore:
    """
    Store for Multimodal Atomic Units.

    Stores lightweight MAU metadata in JSON format while heavy data
    remains in cold storage. Supports efficient lookup by ID, time range,
    and modality.
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
        config: Optional[StorageConfig] = None,
    ):
        self.config = config or StorageConfig()
        self.storage_path = Path(storage_path or self.config.index_dir) / "mau_store"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # In-memory index for fast lookup
        self._id_index: Dict[str, str] = {}  # mau_id -> file_path
        self._time_index: Dict[str, List[str]] = {}  # date_key -> [mau_ids]
        self._modality_index: Dict[str, List[str]] = {}  # modality -> [mau_ids]
        self._event_index: Dict[str, List[str]] = {}  # event_id -> [mau_ids]

        # Thread safety
        self._lock = threading.RLock()

        # Load existing index
        self._load_index()

    def _get_date_key(self, timestamp: float) -> str:
        """Get date key for indexing."""
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")

    def _get_storage_file(self, date_key: str) -> Path:
        """Get storage file path for a date."""
        return self.storage_path / f"mau_{date_key}.jsonl"

    def _load_index(self):
        """Load index from existing files."""
        with self._lock:
            for file_path in self.storage_path.glob("mau_*.jsonl"):
                date_key = file_path.stem.replace("mau_", "")
                self._time_index[date_key] = []

                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                            mau_id = data.get("id")
                            if mau_id:
                                self._id_index[mau_id] = str(file_path)
                                self._time_index[date_key].append(mau_id)

                                modality = data.get("modality_type", "text")
                                if modality not in self._modality_index:
                                    self._modality_index[modality] = []
                                self._modality_index[modality].append(mau_id)

                                event_id = data.get("links", {}).get("event_id")
                                if event_id:
                                    if event_id not in self._event_index:
                                        self._event_index[event_id] = []
                                    self._event_index[event_id].append(mau_id)

                        except json.JSONDecodeError:
                            continue

        logger.info(f"Loaded {len(self._id_index)} MAUs from storage")

    def add(self, mau: MultimodalAtomicUnit) -> str:
        """
        Add a MAU to the store.

        Args:
            mau: The MAU to store

        Returns:
            The MAU ID
        """
        with self._lock:
            date_key = self._get_date_key(mau.timestamp)
            file_path = self._get_storage_file(date_key)

            # Write to file
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(mau.to_json() + "\n")

            # Update indices
            self._id_index[mau.id] = str(file_path)

            if date_key not in self._time_index:
                self._time_index[date_key] = []
            self._time_index[date_key].append(mau.id)

            modality = mau.modality_type.value
            if modality not in self._modality_index:
                self._modality_index[modality] = []
            self._modality_index[modality].append(mau.id)

            if mau.links.event_id:
                if mau.links.event_id not in self._event_index:
                    self._event_index[mau.links.event_id] = []
                self._event_index[mau.links.event_id].append(mau.id)

        logger.debug(f"Added MAU: {mau.id}")
        return mau.id

    def get(self, mau_id: str) -> Optional[MultimodalAtomicUnit]:
        """
        Get a MAU by ID.

        Args:
            mau_id: The MAU ID

        Returns:
            The MAU if found, None otherwise
        """
        with self._lock:
            file_path = self._id_index.get(mau_id)
            if not file_path:
                return None

            # Search in file
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if data.get("id") == mau_id:
                        return MultimodalAtomicUnit.from_dict(data)

        return None

    def get_batch(self, mau_ids: List[str]) -> List[MultimodalAtomicUnit]:
        """
        Get multiple MAUs by IDs.

        Args:
            mau_ids: List of MAU IDs

        Returns:
            List of found MAUs
        """
        results = []
        for mau_id in mau_ids:
            mau = self.get(mau_id)
            if mau:
                results.append(mau)
        return results

    def get_active(self, limit: int = 100) -> List[MultimodalAtomicUnit]:
        """Get only ACTIVE (non-archived) MAUs."""
        results = []
        for mau in self.iter_all():
            if getattr(mau, "status", "ACTIVE") == "ACTIVE":
                results.append(mau)
                if len(results) >= limit:
                    break
        return results

    def update(self, mau: MultimodalAtomicUnit) -> bool:
        """
        Update an existing MAU.

        Note: This is an expensive operation that rewrites the file.
        """
        with self._lock:
            file_path = self._id_index.get(mau.id)
            if not file_path:
                return False

            # Read all MAUs from file
            maus = []
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if data.get("id") == mau.id:
                        maus.append(mau.to_dict())
                    else:
                        maus.append(data)

            # Rewrite file
            with open(file_path, "w", encoding="utf-8") as f:
                for m in maus:
                    f.write(json.dumps(m, ensure_ascii=False) + "\n")

        return True

    def delete(self, mau_id: str) -> bool:
        """
        Delete a MAU.

        Note: This is an expensive operation that rewrites the file.
        """
        with self._lock:
            file_path = self._id_index.get(mau_id)
            if not file_path:
                return False

            # Read all MAUs except the one to delete
            maus = []
            deleted_mau = None
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if data.get("id") == mau_id:
                        deleted_mau = data
                    else:
                        maus.append(data)

            if deleted_mau is None:
                return False

            # Rewrite file
            with open(file_path, "w", encoding="utf-8") as f:
                for m in maus:
                    f.write(json.dumps(m, ensure_ascii=False) + "\n")

            # Update indices
            del self._id_index[mau_id]
            for idx in [self._time_index, self._modality_index, self._event_index]:
                for key in idx:
                    if mau_id in idx[key]:
                        idx[key].remove(mau_id)

        return True

    def archive(
        self, mau_id: str, reason: str = "consolidation", run_id: Optional[str] = None
    ) -> bool:
        """
        Archive a MAU (soft delete). The MAU remains searchable by ID
        but is excluded from default listing/iteration.

        This is the ONLY allowed "removal" operation on episodic memories.
        Physical deletion is FORBIDDEN for benchmark safety.
        """
        mau = self.get(mau_id)
        if not mau:
            return False
        if getattr(mau, "status", "ACTIVE") == "PINNED":
            logger.warning(f"Cannot archive PINNED MAU {mau_id}")
            return False

        mau.status = "ARCHIVED"
        mau.storage_tier = "COLD"
        mau.archived_at = time.time()
        mau.archive_reason = reason
        mau.archive_run_id = run_id
        return self.update(mau)

    def get_by_time_range(
        self,
        start_time: float,
        end_time: float,
        modality: Optional[ModalityType] = None,
    ) -> List[MultimodalAtomicUnit]:
        """
        Get MAUs within a time range.

        Args:
            start_time: Start timestamp
            end_time: End timestamp
            modality: Optional modality filter

        Returns:
            List of MAUs in the time range
        """
        results = []

        start_date = datetime.fromtimestamp(start_time)
        end_date = datetime.fromtimestamp(end_time)

        # Get relevant date keys
        current = start_date
        date_keys = []
        while current <= end_date:
            date_keys.append(current.strftime("%Y-%m-%d"))
            current = datetime(current.year, current.month, current.day + 1)

        with self._lock:
            for date_key in date_keys:
                file_path = self._get_storage_file(date_key)
                if not file_path.exists():
                    continue

                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        ts = data.get("timestamp", 0)
                        if start_time <= ts <= end_time:
                            if modality and data.get("modality_type") != modality.value:
                                continue
                            results.append(MultimodalAtomicUnit.from_dict(data))

        return sorted(results, key=lambda x: x.timestamp)

    def get_by_modality(
        self, modality: ModalityType, limit: int = 100
    ) -> List[MultimodalAtomicUnit]:
        """Get MAUs by modality."""
        with self._lock:
            mau_ids = self._modality_index.get(modality.value, [])[:limit]
        return self.get_batch(mau_ids)

    def get_by_event(self, event_id: str) -> List[MultimodalAtomicUnit]:
        """Get all MAUs belonging to an event."""
        with self._lock:
            mau_ids = self._event_index.get(event_id, [])
        return self.get_batch(mau_ids)

    def iter_all(self) -> Iterator[MultimodalAtomicUnit]:
        """Iterate over all MAUs."""
        with self._lock:
            for file_path in self.storage_path.glob("mau_*.jsonl"):
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        yield MultimodalAtomicUnit.from_dict(data)

    def count(self) -> int:
        """Get total count of MAUs."""
        with self._lock:
            return len(self._id_index)

    def count_by_modality(self) -> Dict[str, int]:
        """Get count by modality."""
        with self._lock:
            return {k: len(v) for k, v in self._modality_index.items()}

    def count_by_status(self) -> Dict[str, int]:
        """Get count by status (ACTIVE, ARCHIVED, PINNED)."""
        counts: Dict[str, int] = {"ACTIVE": 0, "ARCHIVED": 0, "PINNED": 0}
        for mau in self.iter_all():
            status = getattr(mau, "status", "ACTIVE")
            counts[status] = counts.get(status, 0) + 1
        return counts

    def get_summaries(
        self,
        mau_ids: Optional[List[str]] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Get lightweight summaries for MAUs (for pyramid retrieval step 1).

        Returns only summary and metadata, not full MAU.
        """
        results = []

        if mau_ids:
            for mau_id in mau_ids[:limit]:
                mau = self.get(mau_id)
                if mau:
                    results.append(
                        {
                            "id": mau.id,
                            "summary": mau.summary,
                            "modality_type": mau.modality_type.value,
                            "timestamp": mau.timestamp,
                            "tags": mau.metadata.tags,
                        }
                    )
        else:
            count = 0
            for mau in self.iter_all():
                if count >= limit:
                    break
                results.append(
                    {
                        "id": mau.id,
                        "summary": mau.summary,
                        "modality_type": mau.modality_type.value,
                        "timestamp": mau.timestamp,
                        "tags": mau.metadata.tags,
                    }
                )
                count += 1

        return results

    def compact(self) -> None:
        """Compact storage by removing deleted entries."""
        # Implementation would rewrite files removing any gaps
        pass

    def export(self, output_path: str) -> int:
        """Export all MAUs to a single file."""
        count = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for mau in self.iter_all():
                f.write(mau.to_json() + "\n")
                count += 1
        return count

    def import_from_file(self, input_path: str) -> int:
        """Import MAUs from a file."""
        count = 0
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                mau = MultimodalAtomicUnit.from_json(line)
                self.add(mau)
                count += 1
        return count
