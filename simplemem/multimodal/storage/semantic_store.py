"""
Semantic Store - Storage for derived semantic memory nodes.

Stores distilled semantic artifacts (gist, concepts, KG edges, reflections)
with lineage back to episodic MAUs.
"""

import json
import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, cast

logger = logging.getLogger(__name__)


def _new_semantic_node_id() -> str:
    """Create a unique semantic node identifier."""
    return f"sem_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"


def _parse_json_object(line: str) -> Optional[dict[str, object]]:
    """Parse a JSON line into a dictionary payload."""
    payload_obj = cast(object, json.loads(line))
    if isinstance(payload_obj, dict):
        payload_dict = cast(dict[object, object], payload_obj)
        normalized: dict[str, object] = {}
        for key, value in payload_dict.items():
            if isinstance(key, str):
                normalized[key] = value
        return normalized
    return None


@dataclass
class SemanticNode:
    """Derived semantic node with lineage to episodic MAUs."""

    node_id: str = field(default_factory=_new_semantic_node_id)
    node_type: str = "gist"
    content: str = ""
    embedding: Optional[list[float]] = None
    source_mau_ids: list[str] = field(default_factory=list)
    consolidation_run_id: str = ""
    supersedes: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    version: int = 1

    def to_dict(self) -> dict[str, object]:
        """Serialize node to a Python dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "SemanticNode":
        """Create a semantic node from dictionary data."""
        node_id_raw = data.get("node_id")
        node_id = (
            node_id_raw
            if isinstance(node_id_raw, str) and node_id_raw
            else _new_semantic_node_id()
        )

        node_type_raw = data.get("node_type")
        node_type = node_type_raw if isinstance(node_type_raw, str) else "gist"

        content_raw = data.get("content")
        content = content_raw if isinstance(content_raw, str) else ""

        embedding_raw = data.get("embedding")
        embedding: Optional[list[float]] = None
        if isinstance(embedding_raw, list):
            embedding_values = cast(list[object], embedding_raw)
            parsed_embedding: list[float] = []
            for value in embedding_values:
                typed_value = value
                if isinstance(typed_value, (int, float)):
                    parsed_embedding.append(float(typed_value))
            embedding = parsed_embedding

        source_raw = data.get("source_mau_ids")
        source_mau_ids: list[str] = []
        if isinstance(source_raw, list):
            source_values = cast(list[object], source_raw)
            for value in source_values:
                typed_value = value
                if isinstance(typed_value, str):
                    source_mau_ids.append(typed_value)

        run_id_raw = data.get("consolidation_run_id")
        consolidation_run_id = run_id_raw if isinstance(run_id_raw, str) else ""

        supersedes_raw = data.get("supersedes")
        supersedes = supersedes_raw if isinstance(supersedes_raw, str) else None

        created_at_raw = data.get("created_at")
        created_at = (
            float(created_at_raw)
            if isinstance(created_at_raw, (int, float))
            else time.time()
        )

        version_raw = data.get("version")
        version = int(version_raw) if isinstance(version_raw, int) else 1

        return cls(
            node_id=node_id,
            node_type=node_type,
            content=content,
            embedding=embedding,
            source_mau_ids=source_mau_ids,
            consolidation_run_id=consolidation_run_id,
            supersedes=supersedes,
            created_at=created_at,
            version=version,
        )

    def to_json(self) -> str:
        """Serialize node to JSON."""
        return json.dumps(self.to_dict(), ensure_ascii=False)


class SemanticStore:
    """JSONL-backed store for semantic nodes with in-memory indices."""

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path: Path = (
            Path(storage_path)
            if storage_path
            else Path("./omni_memory_data/semantic_store")
        )
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self._id_index: dict[str, str] = {}
        self._type_index: dict[str, list[str]] = {}
        self._source_index: dict[str, list[str]] = {}

        self._lock: threading.RLock = threading.RLock()
        self._load_index()

    def _storage_file_for_run(self, run_id: str) -> Path:
        safe_run_id = run_id or "default"
        return self.storage_path / f"semantic_{safe_run_id}.jsonl"

    def _index_node(self, node: SemanticNode, file_path: Path) -> None:
        self._id_index[node.node_id] = str(file_path)

        if node.node_type not in self._type_index:
            self._type_index[node.node_type] = []
        self._type_index[node.node_type].append(node.node_id)

        for mau_id in node.source_mau_ids:
            if mau_id not in self._source_index:
                self._source_index[mau_id] = []
            self._source_index[mau_id].append(node.node_id)

    def _deindex_node(self, node: SemanticNode) -> None:
        _ = self._id_index.pop(node.node_id, None)

        node_ids = self._type_index.get(node.node_type)
        if node_ids and node.node_id in node_ids:
            _ = node_ids.remove(node.node_id)

        for mau_id in node.source_mau_ids:
            source_nodes = self._source_index.get(mau_id)
            if source_nodes and node.node_id in source_nodes:
                _ = source_nodes.remove(node.node_id)

    def _load_index(self) -> None:
        """Load index from all existing semantic JSONL files."""
        with self._lock:
            for file_path in self.storage_path.glob("semantic_*.jsonl"):
                with open(file_path, "r", encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        try:
                            data = _parse_json_object(line)
                        except json.JSONDecodeError:
                            continue
                        if data is None:
                            continue
                        node = SemanticNode.from_dict(data)
                        self._index_node(node, file_path)

        logger.info("Loaded %d semantic nodes from storage", len(self._id_index))

    def add(self, node: SemanticNode) -> str:
        """Add a semantic node and update indices."""
        with self._lock:
            file_path = self._storage_file_for_run(node.consolidation_run_id)
            with open(file_path, "a", encoding="utf-8") as handle:
                _ = handle.write(node.to_json() + "\n")

            self._index_node(node, file_path)

        logger.debug("Added semantic node: %s", node.node_id)
        return node.node_id

    def get(self, node_id: str) -> Optional[SemanticNode]:
        """Get a semantic node by ID."""
        with self._lock:
            file_path = self._id_index.get(node_id)
            if not file_path:
                return None

            with open(file_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    data = _parse_json_object(line)
                    if data is None:
                        continue
                    if data.get("node_id") == node_id:
                        return SemanticNode.from_dict(data)

        return None

    def get_by_source_mau(self, mau_id: str) -> list[SemanticNode]:
        """Get semantic nodes that derive from a source MAU."""
        with self._lock:
            node_ids = list(self._source_index.get(mau_id, []))

        nodes: list[SemanticNode] = []
        for node_id in node_ids:
            node = self.get(node_id)
            if node is not None:
                nodes.append(node)
        return nodes

    def get_by_type(self, node_type: str, limit: int = 100) -> list[SemanticNode]:
        """Get semantic nodes by type."""
        with self._lock:
            node_ids = list(self._type_index.get(node_type, []))[:limit]

        nodes: list[SemanticNode] = []
        for node_id in node_ids:
            node = self.get(node_id)
            if node is not None:
                nodes.append(node)
        return nodes

    def count(self) -> int:
        """Return total number of indexed semantic nodes."""
        with self._lock:
            return len(self._id_index)

    def delete(self, node_id: str) -> bool:
        """Delete a semantic node (allowed because semantic content is derived)."""
        with self._lock:
            file_path = self._id_index.get(node_id)
            if not file_path:
                return False

            file_obj = Path(file_path)
            kept_lines: list[str] = []
            deleted_node: Optional[SemanticNode] = None

            with open(file_obj, "r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    data = _parse_json_object(line)
                    if data is None:
                        continue
                    if data.get("node_id") == node_id:
                        deleted_node = SemanticNode.from_dict(data)
                    else:
                        kept_lines.append(json.dumps(data, ensure_ascii=False))

            if deleted_node is None:
                return False

            with open(file_obj, "w", encoding="utf-8") as handle:
                for entry in kept_lines:
                    _ = handle.write(entry + "\n")

            self._deindex_node(deleted_node)

        logger.debug("Deleted semantic node: %s", node_id)
        return True
