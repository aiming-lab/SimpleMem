from __future__ import annotations

import json
import logging
import math
import sqlite3
import threading
from pathlib import Path
from typing import Iterable

from .models import MemorySearchHit, MemoryStatus, MemoryType, MemoryUnit, utc_now_iso as _utc_now_iso

logger = logging.getLogger(__name__)


class MemoryStore:
    """SQLite-backed storage for long-term memory units."""

    def __init__(self, db_path: str, enable_event_log: bool = True):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._fts_available = False
        self._lock = threading.RLock()
        self._enable_event_log = enable_event_log
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            self._configure()
            self._migrate()
        except sqlite3.DatabaseError:
            logger.warning(
                "Corrupted memory store at %s; resetting to empty database.", self.db_path
            )
            self._reset_corrupted()
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            self._configure()
            self._migrate()

    def _reset_corrupted(self) -> None:
        """Back up a corrupted DB file and start fresh."""
        backup_path = self.db_path.with_suffix(".db.corrupt")
        try:
            if self.db_path.exists():
                self.db_path.rename(backup_path)
                logger.info("Backed up corrupted store to %s", backup_path)
        except OSError:
            try:
                self.db_path.unlink(missing_ok=True)
            except OSError:
                pass

    def close(self) -> None:
        self.conn.close()

    def get_db_size(self) -> dict:
        """Get database file size and page statistics."""
        try:
            size_bytes = self.db_path.stat().st_size if self.db_path.exists() else 0
        except OSError:
            size_bytes = 0
        page_count = self.conn.execute("PRAGMA page_count").fetchone()[0]
        page_size = self.conn.execute("PRAGMA page_size").fetchone()[0]
        freelist = self.conn.execute("PRAGMA freelist_count").fetchone()[0]
        return {
            "size_bytes": size_bytes,
            "size_mb": round(size_bytes / (1024 * 1024), 2),
            "page_count": page_count,
            "page_size": page_size,
            "freelist_pages": freelist,
            "freelist_ratio": round(freelist / max(page_count, 1), 4),
        }

    def backup(self, backup_path: str) -> bool:
        """Create a full backup of the database.

        Uses SQLite's online backup API for a consistent copy.
        """
        try:
            backup_p = Path(backup_path).expanduser()
            backup_p.parent.mkdir(parents=True, exist_ok=True)
            backup_conn = sqlite3.connect(str(backup_p))
            self.conn.backup(backup_conn)
            backup_conn.close()
            return True
        except Exception as exc:
            logger.warning("Backup failed: %s", exc)
            return False

    def compact(self) -> None:
        """Run VACUUM to reclaim space after deletions."""
        try:
            self.conn.execute("VACUUM")
        except sqlite3.OperationalError:
            pass

    def _configure(self) -> None:
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA synchronous=NORMAL")

    SCHEMA_VERSION = 6  # Bump on schema changes.

    def _migrate(self) -> None:
        # Schema version tracking.
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER NOT NULL,
                applied_at TEXT NOT NULL
            )
            """
        )
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT PRIMARY KEY,
                scope_id TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                content TEXT NOT NULL,
                summary TEXT DEFAULT '',
                source_session_id TEXT DEFAULT '',
                source_turn_start INTEGER DEFAULT 0,
                source_turn_end INTEGER DEFAULT 0,
                entities_json TEXT DEFAULT '[]',
                topics_json TEXT DEFAULT '[]',
                importance REAL DEFAULT 0.5,
                confidence REAL DEFAULT 0.7,
                access_count INTEGER DEFAULT 0,
                reinforcement_score REAL DEFAULT 0.0,
                status TEXT DEFAULT 'active',
                supersedes_json TEXT DEFAULT '[]',
                superseded_by TEXT DEFAULT '',
                embedding_json TEXT DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_accessed_at TEXT DEFAULT '',
                expires_at TEXT DEFAULT '',
                tags_json TEXT DEFAULT '[]'
            );

            CREATE INDEX IF NOT EXISTS idx_memories_scope
            ON memories(scope_id);

            CREATE INDEX IF NOT EXISTS idx_memories_scope_status
            ON memories(scope_id, status);

            CREATE INDEX IF NOT EXISTS idx_memories_scope_type
            ON memories(scope_id, memory_type);
            """
        )
        columns = {
            row["name"]
            for row in self.conn.execute("PRAGMA table_info(memories)").fetchall()
        }
        if "embedding_json" not in columns:
            self.conn.execute("ALTER TABLE memories ADD COLUMN embedding_json TEXT DEFAULT '[]'")
        if "expires_at" not in columns:
            self.conn.execute("ALTER TABLE memories ADD COLUMN expires_at TEXT DEFAULT ''")
        if "tags_json" not in columns:
            self.conn.execute("ALTER TABLE memories ADD COLUMN tags_json TEXT DEFAULT '[]'")
        # Event log for audit trail.
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                memory_id TEXT NOT NULL,
                scope_id TEXT DEFAULT '',
                detail TEXT DEFAULT ''
            )
            """
        )
        # Memory dependency/link tracking for relationship graphs.
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_links (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                link_type TEXT NOT NULL DEFAULT 'related',
                created_at TEXT NOT NULL,
                PRIMARY KEY (source_id, target_id, link_type)
            )
            """
        )
        # Memory watch/subscription table.
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_watches (
                watch_id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT NOT NULL,
                watcher TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(memory_id, watcher)
            )
            """
        )
        # Memory annotations table for user-defined notes.
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_annotations (
                annotation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT NOT NULL,
                author TEXT NOT NULL DEFAULT '',
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        # Scope access control table for multi-tenant deployments.
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scope_access (
                scope_id TEXT NOT NULL,
                principal TEXT NOT NULL,
                permission TEXT NOT NULL DEFAULT 'read',
                created_at TEXT NOT NULL,
                PRIMARY KEY (scope_id, principal, permission)
            )
            """
        )
        self._setup_fts()
        # Record schema version.
        current = self.conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
        if current is None or current < self.SCHEMA_VERSION:
            self.conn.execute(
                "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                (self.SCHEMA_VERSION, _utc_now_iso()),
            )
        self.conn.commit()

    def validate_integrity(self) -> dict:
        """Check store integrity: orphaned links, dangling references, etc."""
        issues: list[str] = []

        # Check for links referencing non-existent memories.
        orphaned_links = self.conn.execute(
            """SELECT COUNT(*) FROM memory_links l
               WHERE NOT EXISTS (SELECT 1 FROM memories m WHERE m.memory_id = l.source_id)
               OR NOT EXISTS (SELECT 1 FROM memories m WHERE m.memory_id = l.target_id)"""
        ).fetchone()[0]
        if orphaned_links:
            issues.append(f"{orphaned_links} orphaned link(s) reference missing memories")

        # Check for watches on non-existent memories.
        orphaned_watches = self.conn.execute(
            """SELECT COUNT(*) FROM memory_watches w
               WHERE NOT EXISTS (SELECT 1 FROM memories m WHERE m.memory_id = w.memory_id)"""
        ).fetchone()[0]
        if orphaned_watches:
            issues.append(f"{orphaned_watches} watch(es) on missing memories")

        # Check for annotations on non-existent memories.
        orphaned_annotations = self.conn.execute(
            """SELECT COUNT(*) FROM memory_annotations a
               WHERE NOT EXISTS (SELECT 1 FROM memories m WHERE m.memory_id = a.memory_id)"""
        ).fetchone()[0]
        if orphaned_annotations:
            issues.append(f"{orphaned_annotations} annotation(s) on missing memories")

        # Check for superseded_by pointing to non-existent memories.
        dangling_superseded = self.conn.execute(
            """SELECT COUNT(*) FROM memories m
               WHERE m.superseded_by != ''
               AND NOT EXISTS (SELECT 1 FROM memories m2 WHERE m2.memory_id = m.superseded_by)"""
        ).fetchone()[0]
        if dangling_superseded:
            issues.append(f"{dangling_superseded} memories with dangling superseded_by reference(s)")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "orphaned_links": orphaned_links,
            "orphaned_watches": orphaned_watches,
            "orphaned_annotations": orphaned_annotations,
            "dangling_superseded": dangling_superseded,
        }

    def cleanup_orphans(self) -> dict:
        """Remove orphaned links, watches, and annotations."""
        removed_links = self.conn.execute(
            """DELETE FROM memory_links
               WHERE NOT EXISTS (SELECT 1 FROM memories m WHERE m.memory_id = memory_links.source_id)
               OR NOT EXISTS (SELECT 1 FROM memories m WHERE m.memory_id = memory_links.target_id)"""
        ).rowcount
        removed_watches = self.conn.execute(
            """DELETE FROM memory_watches
               WHERE NOT EXISTS (SELECT 1 FROM memories m WHERE m.memory_id = memory_watches.memory_id)"""
        ).rowcount
        removed_annotations = self.conn.execute(
            """DELETE FROM memory_annotations
               WHERE NOT EXISTS (SELECT 1 FROM memories m WHERE m.memory_id = memory_annotations.memory_id)"""
        ).rowcount
        self.conn.commit()
        return {
            "removed_links": removed_links,
            "removed_watches": removed_watches,
            "removed_annotations": removed_annotations,
            "total_removed": removed_links + removed_watches + removed_annotations,
        }

    def export_csv(self, scope_id: str) -> str:
        """Export active memories as CSV text."""
        units = self.list_active(scope_id, limit=10000)
        lines = ["memory_id,type,content,importance,confidence,access_count,created_at,tags"]
        for u in units:
            # Escape content for CSV.
            content = u.content.replace('"', '""')
            tags = ";".join(u.tags)
            lines.append(
                f'"{u.memory_id}","{u.memory_type.value}","{content}",{u.importance},{u.confidence},{u.access_count},"{u.created_at}","{tags}"'
            )
        return "\n".join(lines)

    def get_schema_version(self) -> int:
        """Get the current schema version."""
        try:
            row = self.conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
            return row[0] or 0
        except sqlite3.OperationalError:
            return 0

    def _setup_fts(self) -> None:
        """Create FTS5 virtual table for fast full-text search if supported."""
        self._fts_available = False
        try:
            self.conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    memory_id,
                    scope_id,
                    content,
                    summary,
                    entities_text,
                    topics_text,
                    tokenize='unicode61'
                )
                """
            )
            self._fts_available = True
        except sqlite3.OperationalError:
            logger.debug("FTS5 not available; falling back to manual keyword search.")

    def _index_fts(self, unit: MemoryUnit) -> None:
        """Insert a single memory unit into the FTS index."""
        if not self._fts_available:
            return
        try:
            # Remove old entry first, then insert fresh.
            self.conn.execute(
                "DELETE FROM memories_fts WHERE memory_id = ?",
                (unit.memory_id,),
            )
            self.conn.execute(
                "INSERT INTO memories_fts(memory_id, scope_id, content, summary, entities_text, topics_text) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    unit.memory_id,
                    unit.scope_id,
                    unit.content,
                    unit.summary,
                    " ".join(unit.entities),
                    " ".join(unit.topics),
                ),
            )
        except sqlite3.OperationalError as e:
            logger.debug("FTS indexing failed for %s: %s", unit.memory_id, e)

    def _remove_fts(self, memory_id: str) -> None:
        """Remove a memory from the FTS index."""
        if not self._fts_available:
            return
        try:
            self.conn.execute(
                "DELETE FROM memories_fts WHERE memory_id = ?",
                (memory_id,),
            )
        except sqlite3.OperationalError as e:
            logger.debug("FTS removal failed for %s: %s", memory_id, e)

    def _log_event(self, event_type: str, memory_id: str, scope_id: str = "", detail: str = "") -> None:
        """Record a mutation event for audit purposes."""
        if not self._enable_event_log:
            return
        try:
            self.conn.execute(
                "INSERT INTO memory_events (timestamp, event_type, memory_id, scope_id, detail) VALUES (?, ?, ?, ?, ?)",
                (_utc_now_iso(), event_type, memory_id, scope_id, detail[:500]),
            )
        except Exception:
            pass  # Event logging is best-effort.

    def get_event_log(self, scope_id: str = "", limit: int = 50) -> list[dict]:
        """Read recent events, optionally filtered by scope."""
        if scope_id:
            rows = self.conn.execute(
                "SELECT * FROM memory_events WHERE scope_id = ? ORDER BY event_id DESC LIMIT ?",
                (scope_id, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM memory_events ORDER BY event_id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "event_id": row["event_id"],
                "timestamp": row["timestamp"],
                "event_type": row["event_type"],
                "memory_id": row["memory_id"],
                "scope_id": row["scope_id"],
                "detail": row["detail"],
            }
            for row in rows
        ]

    def add_memories(self, units: Iterable[MemoryUnit]) -> int:
        with self._lock:
            count = 0
            for unit in units:
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO memories (
                        memory_id, scope_id, memory_type, content, summary,
                        source_session_id, source_turn_start, source_turn_end,
                        entities_json, topics_json, importance, confidence,
                        access_count, reinforcement_score, status, supersedes_json,
                        superseded_by, embedding_json, created_at, updated_at,
                        last_accessed_at, expires_at, tags_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        unit.memory_id,
                        unit.scope_id,
                        unit.memory_type.value,
                        unit.content,
                        unit.summary,
                        unit.source_session_id,
                        unit.source_turn_start,
                        unit.source_turn_end,
                        json.dumps(unit.entities, ensure_ascii=False),
                        json.dumps(unit.topics, ensure_ascii=False),
                        unit.importance,
                        unit.confidence,
                        unit.access_count,
                        unit.reinforcement_score,
                        unit.status.value,
                        json.dumps(unit.supersedes, ensure_ascii=False),
                        unit.superseded_by,
                        json.dumps(unit.embedding),
                        unit.created_at,
                        unit.updated_at,
                        unit.last_accessed_at,
                        unit.expires_at,
                        json.dumps(unit.tags, ensure_ascii=False),
                    ),
                )
                self._index_fts(unit)
                self._log_event("create", unit.memory_id, unit.scope_id, f"type={unit.memory_type.value}")
                count += 1
            self.conn.commit()
            return count

    def list_active(self, scope_id: str, limit: int = 100) -> list[MemoryUnit]:
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT * FROM memories
                WHERE scope_id = ? AND status = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (scope_id, MemoryStatus.ACTIVE.value, limit),
            ).fetchall()
        units = [self._row_to_unit(row) for row in rows]
        # Filter out expired memories.
        now_iso = _utc_now_iso()
        return [u for u in units if not u.expires_at or u.expires_at > now_iso]

    def expire_stale(self, scope_id: str) -> int:
        """Archive memories that have passed their expires_at timestamp.

        Returns the number of memories archived.
        """
        now_iso = _utc_now_iso()
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT memory_id FROM memories
                WHERE scope_id = ? AND status = 'active'
                  AND expires_at != '' AND expires_at <= ?
                """,
                (scope_id, now_iso),
            ).fetchall()
        count = 0
        for row in rows:
            mid = row["memory_id"]
            self.conn.execute(
                "UPDATE memories SET status = 'archived', updated_at = ? WHERE memory_id = ?",
                (now_iso, mid),
            )
            self._remove_fts(mid)
            count += 1
        if count:
            self.conn.commit()
        return count

    def set_ttl(self, memory_id: str, expires_at: str) -> bool:
        """Set or update the expiry time for a memory.

        Args:
            memory_id: The memory to update.
            expires_at: ISO-8601 timestamp when the memory should expire.
                        Pass empty string to remove the TTL.
        """
        row = self.conn.execute(
            "SELECT memory_id FROM memories WHERE memory_id = ?",
            (memory_id,),
        ).fetchone()
        if row is None:
            return False
        self.conn.execute(
            "UPDATE memories SET expires_at = ?, updated_at = ? WHERE memory_id = ?",
            (expires_at, _utc_now_iso(), memory_id),
        )
        self.conn.commit()
        return True

    def share_to_scope(self, memory_id: str, target_scope_id: str) -> str | None:
        """Copy a memory to another scope for cross-scope knowledge sharing.

        Creates a new memory unit in the target scope with the same content
        but a fresh ID. Returns the new memory ID, or None if the source
        memory was not found.
        """
        source = self._get_by_id(memory_id)
        if source is None:
            return None
        import uuid as _uuid
        new_id = str(_uuid.uuid4())
        shared = MemoryUnit(
            memory_id=new_id,
            scope_id=target_scope_id,
            memory_type=source.memory_type,
            content=source.content,
            summary=source.summary,
            source_session_id=source.source_session_id,
            source_turn_start=source.source_turn_start,
            source_turn_end=source.source_turn_end,
            entities=list(source.entities),
            topics=list(source.topics),
            importance=source.importance,
            confidence=max(source.confidence - 0.05, 0.5),  # slight confidence reduction for shared
            embedding=list(source.embedding),
        )
        self.add_memories([shared])
        self._log_event("share", memory_id, target_scope_id, f"new_id={new_id[:36]}")
        return new_id

    def import_memories_json(self, data: list[dict], target_scope_id: str | None = None) -> int:
        """Import memories from JSON dicts (as produced by export_scope_json).

        If target_scope_id is provided, all imported memories are placed in that scope.
        Otherwise each memory keeps its original scope_id.
        Returns the number of memories imported.
        """
        import uuid as _uuid
        units = []
        for item in data:
            try:
                mt = MemoryType(item.get("memory_type", "episodic"))
            except ValueError:
                mt = MemoryType.EPISODIC
            scope = target_scope_id or item.get("scope_id", "default")
            unit = MemoryUnit(
                memory_id=str(_uuid.uuid4()),  # always generate fresh ID
                scope_id=scope,
                memory_type=mt,
                content=item.get("content", ""),
                summary=item.get("summary", ""),
                source_session_id=item.get("source_session_id", ""),
                source_turn_start=int(item.get("source_turn_start", 0)),
                source_turn_end=int(item.get("source_turn_end", 0)),
                entities=item.get("entities", []),
                topics=item.get("topics", []),
                importance=float(item.get("importance", 0.5)),
                confidence=float(item.get("confidence", 0.7)),
                access_count=0,  # reset access stats on import
                reinforcement_score=0.0,
                expires_at=item.get("expires_at", ""),
                tags=item.get("tags", []),
            )
            units.append(unit)
        return self.add_memories(units)

    def set_type_ttl(self, scope_id: str, memory_type: MemoryType, expires_at: str) -> int:
        """Set TTL on all active memories of a given type in a scope.

        Returns the number of memories updated.
        """
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT memory_id FROM memories
                WHERE scope_id = ? AND memory_type = ? AND status = 'active'
                """,
                (scope_id, memory_type.value),
            ).fetchall()
            count = 0
            now = _utc_now_iso()
            for row in rows:
                self.conn.execute(
                    "UPDATE memories SET expires_at = ?, updated_at = ? WHERE memory_id = ?",
                    (expires_at, now, row["memory_id"]),
                )
                count += 1
            if count:
                self.conn.commit()
            return count

    def merge_memories(self, id_a: str, id_b: str, merged_content: str, merged_summary: str = "") -> str | None:
        """Merge two memories into a new one, superseding both originals.

        Returns the new memory ID, or None if either source is not found.
        """
        with self._lock:
            import uuid as _uuid
            a = self._get_by_id(id_a)
            b = self._get_by_id(id_b)
            if a is None or b is None:
                return None

            now = _utc_now_iso()
            new_id = str(_uuid.uuid4())
            # Combine entities and topics, deduplicated.
            entities = list(dict.fromkeys(a.entities + b.entities))[:12]
            topics = list(dict.fromkeys(a.topics + b.topics))[:12]
            merged = MemoryUnit(
                memory_id=new_id,
                scope_id=a.scope_id,
                memory_type=a.memory_type,
                content=merged_content,
                summary=merged_summary or f"Merged from {id_a[:8]} and {id_b[:8]}",
                source_session_id=a.source_session_id,
                source_turn_start=min(a.source_turn_start, b.source_turn_start),
                source_turn_end=max(a.source_turn_end, b.source_turn_end),
                entities=entities,
                topics=topics,
                importance=max(a.importance, b.importance),
                confidence=max(a.confidence, b.confidence),
                supersedes=[id_a, id_b],
                embedding=list(a.embedding) if a.embedding else list(b.embedding),
            )
            self.add_memories([merged])
            # Supersede both originals.
            self.supersede(id_a, new_id, now)
            self.supersede(id_b, new_id, now)
            self._log_event("merge", new_id, a.scope_id, f"from={id_a[:18]}+{id_b[:18]}")
            return new_id

    def add_tags(self, memory_id: str, tags: list[str]) -> bool:
        """Add tags to a memory (deduplicating with existing tags)."""
        with self._lock:
            unit = self._get_by_id(memory_id)
            if unit is None:
                return False
            existing = set(unit.tags)
            for tag in tags:
                existing.add(tag.lower().strip())
            self.conn.execute(
                "UPDATE memories SET tags_json = ?, updated_at = ? WHERE memory_id = ?",
                (json.dumps(sorted(existing), ensure_ascii=False), _utc_now_iso(), memory_id),
            )
            self.conn.commit()
            return True

    def remove_tags(self, memory_id: str, tags: list[str]) -> bool:
        """Remove tags from a memory."""
        with self._lock:
            unit = self._get_by_id(memory_id)
            if unit is None:
                return False
            to_remove = {t.lower().strip() for t in tags}
            remaining = [t for t in unit.tags if t not in to_remove]
            self.conn.execute(
                "UPDATE memories SET tags_json = ?, updated_at = ? WHERE memory_id = ?",
                (json.dumps(remaining, ensure_ascii=False), _utc_now_iso(), memory_id),
            )
            self.conn.commit()
            return True

    def search_by_tag(self, scope_id: str, tag: str, limit: int = 50) -> list[MemoryUnit]:
        """Find all active memories in a scope that have a given tag."""
        units = self.list_active(scope_id, limit=limit)
        tag_lower = tag.lower().strip()
        return [u for u in units if tag_lower in {t.lower() for t in u.tags}]

    def get_memory_history(self, memory_id: str) -> list[dict]:
        """Get version history for a memory (supersedes chain and modifications).

        Traverses the supersedes chain backward to find the evolution history
        of a concept through merges and updates.
        """
        history: list[dict] = []
        visited: set[str] = set()
        queue = [memory_id]
        while queue:
            mid = queue.pop(0)
            if mid in visited:
                continue
            visited.add(mid)
            unit = self._get_by_id(mid)
            if unit is None:
                continue
            history.append({
                "memory_id": unit.memory_id,
                "content": unit.content[:200],
                "status": unit.status.value,
                "created_at": unit.created_at,
                "updated_at": unit.updated_at,
                "importance": unit.importance,
                "supersedes": unit.supersedes,
                "superseded_by": unit.superseded_by,
            })
            # Walk backward through supersedes chain.
            for sid in unit.supersedes:
                if sid not in visited:
                    queue.append(sid)
        return sorted(history, key=lambda h: h["created_at"])

    def get_scope_analytics(self, scope_id: str) -> dict:
        """Generate analytics for a scope including growth, access, and quality metrics."""
        from datetime import datetime as _dt, timezone as _tz

        with self._lock:
            all_rows = self.conn.execute(
                "SELECT * FROM memories WHERE scope_id = ?",
                (scope_id,),
            ).fetchall()

        if not all_rows:
            return {"total": 0}

        units = [self._row_to_unit(row) for row in all_rows]
        now = _dt.now(_tz.utc)

        active = [u for u in units if u.status == MemoryStatus.ACTIVE]
        superseded = [u for u in units if u.status == MemoryStatus.SUPERSEDED]
        archived = [u for u in units if u.status == MemoryStatus.ARCHIVED]

        # Type distribution.
        type_dist: dict[str, int] = {}
        for u in active:
            type_dist[u.memory_type.value] = type_dist.get(u.memory_type.value, 0) + 1

        # Access stats.
        total_accesses = sum(u.access_count for u in active)
        avg_access = total_accesses / max(len(active), 1)
        never_accessed = sum(1 for u in active if u.access_count == 0)
        highly_accessed = sum(1 for u in active if u.access_count >= 5)

        # Importance distribution.
        if active:
            avg_importance = sum(u.importance for u in active) / len(active)
            high_importance = sum(1 for u in active if u.importance >= 0.8)
            low_importance = sum(1 for u in active if u.importance < 0.3)
        else:
            avg_importance = 0.0
            high_importance = 0
            low_importance = 0

        # TTL stats.
        with_ttl = sum(1 for u in active if u.expires_at)
        with_tags = sum(1 for u in active if u.tags)

        # Pinned.
        pinned = sum(1 for u in active if u.importance >= 0.99)

        return {
            "total": len(units),
            "active": len(active),
            "superseded": len(superseded),
            "archived": len(archived),
            "type_distribution": type_dist,
            "access": {
                "total_accesses": total_accesses,
                "avg_access_count": round(avg_access, 2),
                "never_accessed": never_accessed,
                "highly_accessed": highly_accessed,
            },
            "importance": {
                "average": round(avg_importance, 4),
                "high_count": high_importance,
                "low_count": low_importance,
            },
            "features": {
                "with_ttl": with_ttl,
                "with_tags": with_tags,
                "pinned": pinned,
            },
        }

    def bulk_archive(self, memory_ids: list[str]) -> int:
        """Archive multiple memories at once. Returns count archived."""
        now = _utc_now_iso()
        count = 0
        for mid in memory_ids:
            row = self.conn.execute(
                "SELECT memory_id, scope_id FROM memories WHERE memory_id = ? AND status = 'active'",
                (mid,),
            ).fetchone()
            if row is None:
                continue
            self.conn.execute(
                "UPDATE memories SET status = 'archived', updated_at = ? WHERE memory_id = ?",
                (now, mid),
            )
            self._remove_fts(mid)
            self._log_event("archive", mid, row["scope_id"])
            count += 1
        if count:
            self.conn.commit()
        return count

    def bulk_add_tags(self, memory_ids: list[str], tags: list[str]) -> int:
        """Add tags to multiple memories at once. Returns count updated."""
        count = 0
        for mid in memory_ids:
            if self.add_tags(mid, tags):
                count += 1
        return count

    def snapshot_scope(self, scope_id: str) -> dict:
        """Create a point-in-time snapshot of a scope for potential rollback.

        Returns a snapshot dict containing all active memory data and metadata.
        """
        units = self.export_scope_json(scope_id)
        stats = self.get_stats(scope_id)
        return {
            "snapshot_at": _utc_now_iso(),
            "scope_id": scope_id,
            "stats": stats,
            "memories": units,
        }

    def restore_snapshot(self, snapshot: dict) -> int:
        """Restore a scope from a previous snapshot.

        Archives all current active memories, then imports the snapshot.
        Returns the number of memories restored.
        """
        scope_id = snapshot.get("scope_id", "")
        if not scope_id:
            return 0
        memories = snapshot.get("memories", [])
        if not memories:
            return 0
        # Archive current active memories.
        current = self.list_active(scope_id, limit=10000)
        if current:
            now = _utc_now_iso()
            for u in current:
                self.conn.execute(
                    "UPDATE memories SET status = 'archived', updated_at = ? WHERE memory_id = ?",
                    (now, u.memory_id),
                )
                self._remove_fts(u.memory_id)
            self.conn.commit()
        # Import snapshot memories.
        return self.import_memories_json(memories, target_scope_id=scope_id)

    def find_similar(self, memory_id: str, limit: int = 5) -> list[tuple[MemoryUnit, float]]:
        """Find memories similar to a given memory based on topic/entity overlap.

        Returns list of (unit, similarity_score) tuples sorted by score descending.
        """
        source = self._get_by_id(memory_id)
        if source is None:
            return []
        source_terms = set(t.lower() for t in source.topics + source.entities)
        if not source_terms:
            return []
        units = self.list_active(source.scope_id, limit=500)
        scored: list[tuple[MemoryUnit, float]] = []
        for u in units:
            if u.memory_id == memory_id:
                continue
            u_terms = set(t.lower() for t in u.topics + u.entities)
            if not u_terms:
                continue
            overlap = len(source_terms & u_terms) / float(len(source_terms | u_terms))
            if overlap > 0.1:
                scored.append((u, round(overlap, 4)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def compute_health_score(self, scope_id: str) -> dict:
        """Compute a single health score (0-100) for a scope.

        Factors: access coverage, importance distribution, type diversity,
        freshness, and absence of stale/expired memories.
        """
        from datetime import datetime as _dt, timezone as _tz

        units = self.list_active(scope_id, limit=5000)
        if not units:
            return {"score": 0, "components": {}, "active_count": 0}

        now = _dt.now(_tz.utc)
        n = float(len(units))

        # 1. Access coverage (0-25): what % of memories have been accessed?
        accessed = sum(1 for u in units if u.access_count > 0)
        access_score = min(25, 25 * (accessed / n))

        # 2. Importance distribution (0-25): healthy pool has diverse importance.
        importances = [u.importance for u in units]
        avg_imp = sum(importances) / n
        # Score peaks when avg importance is in 0.4-0.7 range (not all high or all low).
        imp_center_dist = abs(avg_imp - 0.55)
        importance_score = max(0, 25 * (1 - imp_center_dist / 0.45))

        # 3. Type diversity (0-25): more types = healthier.
        types_present = len({u.memory_type.value for u in units})
        type_diversity_score = min(25, 25 * types_present / 4.0)  # 4+ types = max

        # 4. Freshness (0-25): recent updates indicate active maintenance.
        fresh_count = 0
        for u in units:
            try:
                updated = _dt.fromisoformat(u.updated_at.replace("Z", "+00:00"))
                if updated.tzinfo is None:
                    updated = updated.replace(tzinfo=_tz.utc)
                age_days = (now - updated).total_seconds() / 86400.0
                if age_days < 30:
                    fresh_count += 1
            except (ValueError, TypeError):
                pass
        freshness_score = min(25, 25 * (fresh_count / n))

        total = round(access_score + importance_score + type_diversity_score + freshness_score, 1)
        return {
            "score": total,
            "components": {
                "access_coverage": round(access_score, 1),
                "importance_health": round(importance_score, 1),
                "type_diversity": round(type_diversity_score, 1),
                "freshness": round(freshness_score, 1),
            },
            "active_count": len(units),
        }

    def find_duplicates(self, scope_id: str, threshold: float = 0.80) -> list[dict]:
        """Find near-duplicate memory pairs based on content similarity.

        Uses Jaccard similarity on word tokens. Returns pairs above threshold.
        """
        units = self.list_active(scope_id, limit=500)
        if len(units) < 2:
            return []

        # Precompute word sets for each unit.
        word_sets: list[set[str]] = []
        for u in units:
            words = set(u.content.lower().split())
            word_sets.append(words)

        duplicates: list[dict] = []
        for i in range(len(units)):
            if not word_sets[i]:
                continue
            for j in range(i + 1, len(units)):
                if not word_sets[j]:
                    continue
                intersection = len(word_sets[i] & word_sets[j])
                union = len(word_sets[i] | word_sets[j])
                if union == 0:
                    continue
                sim = intersection / float(union)
                if sim >= threshold:
                    duplicates.append({
                        "id_a": units[i].memory_id,
                        "id_b": units[j].memory_id,
                        "similarity": round(sim, 4),
                        "type": units[i].memory_type.value,
                        "content_a": units[i].content[:100],
                        "content_b": units[j].content[:100],
                    })
        duplicates.sort(key=lambda d: d["similarity"], reverse=True)
        return duplicates

    def save_stats_snapshot(self, scope_id: str) -> dict:
        """Save a timestamped stats snapshot for trend tracking.

        Stores the snapshot in a dedicated table and returns the data.
        """
        # Ensure stats_snapshots table exists.
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS stats_snapshots (
                snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                scope_id TEXT NOT NULL,
                data_json TEXT NOT NULL
            )
            """
        )
        stats = self.get_stats(scope_id)
        health = self.compute_health_score(scope_id)
        snapshot = {
            "timestamp": _utc_now_iso(),
            "scope_id": scope_id,
            "active": stats.get("active", 0),
            "total": stats.get("total", 0),
            "active_by_type": stats.get("active_by_type", {}),
            "health_score": health.get("score", 0),
        }
        self.conn.execute(
            "INSERT INTO stats_snapshots (timestamp, scope_id, data_json) VALUES (?, ?, ?)",
            (snapshot["timestamp"], scope_id, json.dumps(snapshot, ensure_ascii=False)),
        )
        self.conn.commit()
        return snapshot

    def get_stats_trend(self, scope_id: str, limit: int = 20) -> list[dict]:
        """Get recent stats snapshots for trend analysis."""
        try:
            rows = self.conn.execute(
                """
                SELECT data_json FROM stats_snapshots
                WHERE scope_id = ?
                ORDER BY snapshot_id DESC
                LIMIT ?
                """,
                (scope_id, limit),
            ).fetchall()
        except Exception:
            return []
        snapshots = []
        for row in rows:
            try:
                snapshots.append(json.loads(row["data_json"]))
            except Exception:
                pass
        snapshots.reverse()  # oldest first
        return snapshots

    def search_advanced(
        self,
        scope_id: str,
        keyword: str = "",
        memory_type: str = "",
        tag: str = "",
        min_importance: float = 0.0,
        limit: int = 50,
    ) -> list[MemoryUnit]:
        """Search memories with combined criteria.

        All non-empty criteria are ANDed together.
        """
        units = self.list_active(scope_id, limit=limit * 5)  # over-fetch for filtering
        results = []
        keyword_lower = keyword.lower() if keyword else ""
        tag_lower = tag.lower().strip() if tag else ""
        for u in units:
            if keyword_lower and keyword_lower not in u.content.lower() and keyword_lower not in u.summary.lower():
                continue
            if memory_type and u.memory_type.value != memory_type:
                continue
            if tag_lower and tag_lower not in {t.lower() for t in u.tags}:
                continue
            if u.importance < min_importance:
                continue
            results.append(u)
            if len(results) >= limit:
                break
        return results

    def compare_scopes(self, scope_a: str, scope_b: str) -> dict:
        """Compare two scopes to find shared and unique content.

        Uses content hashing to identify shared facts.
        """
        units_a = self.list_active(scope_a, limit=1000)
        units_b = self.list_active(scope_b, limit=1000)

        content_a = {u.content.strip().lower(): u for u in units_a}
        content_b = {u.content.strip().lower(): u for u in units_b}

        shared_keys = set(content_a.keys()) & set(content_b.keys())
        unique_a = set(content_a.keys()) - shared_keys
        unique_b = set(content_b.keys()) - shared_keys

        return {
            "scope_a": scope_a,
            "scope_b": scope_b,
            "scope_a_count": len(units_a),
            "scope_b_count": len(units_b),
            "shared_count": len(shared_keys),
            "unique_to_a": len(unique_a),
            "unique_to_b": len(unique_b),
            "shared_content": [content_a[k].content[:100] for k in list(shared_keys)[:5]],
        }

    def export_scope_json(self, scope_id: str) -> list[dict]:
        """Export all active memories for a scope as JSON-serializable dicts.

        Useful for external backup, analysis, or migration.
        """
        units = self.list_active(scope_id, limit=10000)
        result = []
        for u in units:
            result.append({
                "memory_id": u.memory_id,
                "scope_id": u.scope_id,
                "memory_type": u.memory_type.value,
                "content": u.content,
                "summary": u.summary,
                "source_session_id": u.source_session_id,
                "source_turn_start": u.source_turn_start,
                "source_turn_end": u.source_turn_end,
                "entities": u.entities,
                "topics": u.topics,
                "importance": u.importance,
                "confidence": u.confidence,
                "access_count": u.access_count,
                "reinforcement_score": u.reinforcement_score,
                "status": u.status.value,
                "created_at": u.created_at,
                "updated_at": u.updated_at,
                "last_accessed_at": u.last_accessed_at,
                "expires_at": u.expires_at,
                "tags": u.tags,
            })
        return result

    def search_keyword(self, scope_id: str, query_text: str, limit: int = 6) -> list[MemorySearchHit]:
        terms = [t.lower() for t in _tokenize(query_text) if t]
        if not terms:
            return []

        # Try FTS-accelerated search first for large pools.
        if self._fts_available:
            fts_hits = self._search_fts(scope_id, terms, limit)
            if fts_hits is not None:
                return fts_hits

        # Fallback: manual IDF-weighted keyword scan.
        return self._search_keyword_manual(scope_id, terms, limit)

    def _search_fts(self, scope_id: str, terms: list[str], limit: int) -> list[MemorySearchHit] | None:
        """Use FTS5 MATCH to pre-filter candidates, then IDF-rank them."""
        try:
            # Build FTS query with scope filter (escape quotes to prevent injection).
            escaped_scope = scope_id.replace('"', '""')
            escaped_terms = [t.replace('"', '""') for t in terms[:12]]
            term_clause = " OR ".join(f'"{t}"' for t in escaped_terms)
            fts_query = f'scope_id:"{escaped_scope}" AND ({term_clause})'
            with self._lock:
                rows = self.conn.execute(
                    "SELECT memory_id FROM memories_fts WHERE memories_fts MATCH ?",
                    (fts_query,),
                ).fetchall()
        except sqlite3.OperationalError:
            return None

        if not rows:
            # FTS found nothing; fall back to manual scan since FTS tokenization
            # may differ from our keyword tokenizer.
            return None

        fts_ids = {row[0] for row in rows}
        # Filter to active-only from the main table.
        units = [
            u for u in self.list_active(scope_id, limit=500)
            if u.memory_id in fts_ids
        ]
        if not units:
            return []

        return self._rank_with_idf(units, terms, limit)

    def _search_keyword_manual(self, scope_id: str, terms: list[str], limit: int) -> list[MemorySearchHit]:
        """Manual IDF-weighted keyword scan over all active units."""
        units = self.list_active(scope_id, limit=500)
        if not units:
            return []

        # Build IDF weights: rarer terms across the corpus score higher.
        doc_freq: dict[str, int] = {}
        haystacks: list[str] = []
        for unit in units:
            haystack = " ".join(
                [
                    unit.content.lower(),
                    unit.summary.lower(),
                    " ".join(x.lower() for x in unit.entities),
                    " ".join(x.lower() for x in unit.topics),
                ]
            )
            haystacks.append(haystack)
            for term in set(terms):
                if term in haystack:
                    doc_freq[term] = doc_freq.get(term, 0) + 1

        num_docs = float(len(units))
        hits: list[MemorySearchHit] = []
        for idx, unit in enumerate(units):
            haystack = haystacks[idx]
            matched = [term for term in terms if term in haystack]
            if not matched:
                continue
            idf_score = sum(
                _log2(num_docs / float(doc_freq.get(term, 1)))
                for term in matched
            )
            score = idf_score + unit.importance + unit.reinforcement_score
            hits.append(MemorySearchHit(unit=unit, score=score, matched_terms=matched))

        hits.sort(key=lambda h: (h.score, h.unit.updated_at), reverse=True)
        return hits[:limit]

    def _rank_with_idf(self, units: list[MemoryUnit], terms: list[str], limit: int) -> list[MemorySearchHit]:
        """IDF-weighted ranking for a pre-filtered set of units."""
        doc_freq: dict[str, int] = {}
        haystacks: list[str] = []
        for unit in units:
            haystack = " ".join(
                [
                    unit.content.lower(),
                    unit.summary.lower(),
                    " ".join(x.lower() for x in unit.entities),
                    " ".join(x.lower() for x in unit.topics),
                ]
            )
            haystacks.append(haystack)
            for term in set(terms):
                if term in haystack:
                    doc_freq[term] = doc_freq.get(term, 0) + 1

        num_docs = float(len(units))
        hits: list[MemorySearchHit] = []
        for idx, unit in enumerate(units):
            haystack = haystacks[idx]
            matched = [term for term in terms if term in haystack]
            if not matched:
                continue
            idf_score = sum(
                _log2(num_docs / float(doc_freq.get(term, 1)))
                for term in matched
            )
            score = idf_score + unit.importance + unit.reinforcement_score
            hits.append(MemorySearchHit(unit=unit, score=score, matched_terms=matched))

        hits.sort(key=lambda h: (h.score, h.unit.updated_at), reverse=True)
        return hits[:limit]

    def list_scopes(self) -> list[dict]:
        """List all scopes in the store with their memory counts."""
        rows = self.conn.execute(
            """
            SELECT scope_id, COUNT(*) as total,
                   SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) as active
            FROM memories
            GROUP BY scope_id
            ORDER BY total DESC
            """
        ).fetchall()
        return [
            {"scope_id": row["scope_id"], "total": row["total"], "active": row["active"]}
            for row in rows
        ]

    def update_content(self, memory_id: str, content: str, summary: str = "") -> bool:
        """Update the content of an existing memory."""
        with self._lock:
            row = self.conn.execute(
                "SELECT memory_id FROM memories WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
            if row is None:
                return False
            if summary:
                self.conn.execute(
                    "UPDATE memories SET content = ?, summary = ?, updated_at = ? WHERE memory_id = ?",
                    (content, summary, _utc_now_iso(), memory_id),
                )
            else:
                self.conn.execute(
                    "UPDATE memories SET content = ?, updated_at = ? WHERE memory_id = ?",
                    (content, _utc_now_iso(), memory_id),
                )
            # Re-index FTS before commit to keep them in sync.
            unit = self._get_by_id(memory_id)
            if unit:
                self._index_fts(unit)
            self.conn.commit()
            return True

    def _get_by_id(self, memory_id: str) -> MemoryUnit | None:
        """Get a single memory unit by ID."""
        row = self.conn.execute(
            "SELECT * FROM memories WHERE memory_id = ?",
            (memory_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_unit(row)

    def get_by_ids(self, memory_ids: list[str]) -> list[MemoryUnit]:
        """Retrieve multiple memory units by their IDs in a single query."""
        if not memory_ids:
            return []
        placeholders = ",".join("?" for _ in memory_ids)
        rows = self.conn.execute(
            f"SELECT * FROM memories WHERE memory_id IN ({placeholders})",
            memory_ids,
        ).fetchall()
        return [self._row_to_unit(r) for r in rows]

    def garbage_collect(self, scope_id: str) -> dict:
        """Remove orphaned superseded memories and return cleanup stats."""
        rows = self.conn.execute(
            "SELECT memory_id FROM memories WHERE scope_id = ? AND status = 'superseded'",
            (scope_id,),
        ).fetchall()
        superseded_ids = {row["memory_id"] for row in rows}

        active_units = self.list_active(scope_id, limit=10000)
        referenced = set()
        for unit in active_units:
            for sid in unit.supersedes:
                referenced.add(sid)

        orphans = superseded_ids - referenced
        for oid in orphans:
            self.conn.execute("DELETE FROM memories WHERE memory_id = ?", (oid,))
            self._remove_fts(oid)
        if orphans:
            self.conn.commit()
        return {"removed": len(orphans), "kept_superseded": len(superseded_ids) - len(orphans)}

    def pin_memory(self, memory_id: str) -> bool:
        """Pin a memory by setting its importance to maximum (0.99).

        Pinned memories are effectively guaranteed to appear in retrieval.
        """
        row = self.conn.execute(
            "SELECT memory_id FROM memories WHERE memory_id = ?",
            (memory_id,),
        ).fetchone()
        if row is None:
            return False
        self.conn.execute(
            "UPDATE memories SET importance = 0.99, updated_at = ? WHERE memory_id = ?",
            (_utc_now_iso(), memory_id),
        )
        self._log_event("pin", memory_id)
        self.conn.commit()
        return True

    def unpin_memory(self, memory_id: str, restore_importance: float = 0.7) -> bool:
        """Unpin a memory by restoring its importance to a default value."""
        row = self.conn.execute(
            "SELECT memory_id, importance FROM memories WHERE memory_id = ?",
            (memory_id,),
        ).fetchone()
        if row is None:
            return False
        if float(row["importance"]) >= 0.99:
            self.conn.execute(
                "UPDATE memories SET importance = ?, updated_at = ? WHERE memory_id = ?",
                (restore_importance, _utc_now_iso(), memory_id),
            )
            self.conn.commit()
        return True

    def record_feedback(self, memory_id: str, helpful: bool) -> None:
        """Record retrieval feedback by adjusting importance.

        Positive feedback boosts importance by 0.03 (capped at 0.95).
        Negative feedback reduces importance by 0.05 (floored at 0.1).
        """
        with self._lock:
            row = self.conn.execute(
                "SELECT importance, scope_id FROM memories WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
            if row is None:
                return
            current = float(row["importance"])
            scope = row["scope_id"] or ""
            if helpful:
                new_importance = min(0.95, current + 0.03)
            else:
                new_importance = max(0.1, current - 0.05)
            self.conn.execute(
                "UPDATE memories SET importance = ?, updated_at = ? WHERE memory_id = ?",
                (round(new_importance, 4), _utc_now_iso(), memory_id),
            )
            self.conn.commit()
            direction = "positive" if helpful else "negative"
            self._log_event("feedback", memory_id, scope_id=scope, detail=f"{direction}: {current:.4f} -> {new_importance:.4f}")

    def mark_accessed(self, memory_ids: Iterable[str], accessed_at: str) -> None:
        ids = list(memory_ids)
        if not ids:
            return
        with self._lock:
            self.conn.executemany(
                """
                UPDATE memories
                SET access_count = access_count + 1,
                    last_accessed_at = ?,
                    updated_at = ?
                WHERE memory_id = ?
                """,
                [(accessed_at, accessed_at, memory_id) for memory_id in ids],
            )
            self.conn.commit()

    def update_importance(self, memory_id: str, importance: float, updated_at: str) -> None:
        with self._lock:
            self.conn.execute(
                """
                UPDATE memories
                SET importance = ?, updated_at = ?
                WHERE memory_id = ?
                """,
                (importance, updated_at, memory_id),
            )
            self.conn.commit()

    def update_reinforcement(self, memory_id: str, reinforcement_score: float, updated_at: str) -> None:
        with self._lock:
            self.conn.execute(
                """
                UPDATE memories
                SET reinforcement_score = ?, updated_at = ?
                WHERE memory_id = ?
                """,
                (reinforcement_score, updated_at, memory_id),
            )
            self.conn.commit()

    def supersede(self, memory_id: str, superseded_by: str, updated_at: str) -> None:
        with self._lock:
            self.conn.execute(
                """
                UPDATE memories
                SET status = ?, superseded_by = ?, updated_at = ?
                WHERE memory_id = ?
                """,
                (MemoryStatus.SUPERSEDED.value, superseded_by, updated_at, memory_id),
            )
            self._remove_fts(memory_id)
            self._log_event("supersede", memory_id, detail=f"by={superseded_by[:36]}")
            self.conn.commit()

    def get_stats(self, scope_id: str) -> dict:
        with self._lock:
            row = self.conn.execute(
                """
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) AS active
                FROM memories
                WHERE scope_id = ?
                """,
                (scope_id,),
            ).fetchone()
            type_rows = self.conn.execute(
                """
                SELECT memory_type, COUNT(*) AS count
                FROM memories
                WHERE scope_id = ? AND status = 'active'
                GROUP BY memory_type
                ORDER BY count DESC, memory_type ASC
                """,
                (scope_id,),
            ).fetchall()
        return {
            "total": int(row["total"] or 0),
            "active": int(row["active"] or 0),
            "active_by_type": {
                str(type_row["memory_type"]): int(type_row["count"] or 0)
                for type_row in type_rows
            },
        }

    def _row_to_unit(self, row: sqlite3.Row) -> MemoryUnit:
        return MemoryUnit(
            memory_id=row["memory_id"],
            scope_id=row["scope_id"],
            memory_type=MemoryType(row["memory_type"]),
            content=row["content"],
            summary=row["summary"] or "",
            source_session_id=row["source_session_id"] or "",
            source_turn_start=int(row["source_turn_start"] or 0),
            source_turn_end=int(row["source_turn_end"] or 0),
            entities=_json_list(row["entities_json"]),
            topics=_json_list(row["topics_json"]),
            importance=float(row["importance"] or 0.0),
            confidence=float(row["confidence"] or 0.0),
            access_count=int(row["access_count"] or 0),
            reinforcement_score=float(row["reinforcement_score"] or 0.0),
            status=MemoryStatus(row["status"]),
            supersedes=_json_list(row["supersedes_json"]),
            superseded_by=row["superseded_by"] or "",
            embedding=_json_float_list(row["embedding_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_accessed_at=row["last_accessed_at"] or "",
            expires_at=row["expires_at"] or "" if "expires_at" in row.keys() else "",
            tags=_json_list(row["tags_json"]) if "tags_json" in row.keys() else [],
        )

    # --- Scope access control ---

    def grant_access(self, scope_id: str, principal: str, permission: str = "read") -> bool:
        """Grant a permission to a principal for a scope.

        Permissions: 'read', 'write', 'admin'.
        Returns True if newly granted, False if already existed.
        """
        if permission not in ("read", "write", "admin"):
            return False
        try:
            self.conn.execute(
                "INSERT INTO scope_access (scope_id, principal, permission, created_at) VALUES (?, ?, ?, ?)",
                (scope_id, principal, permission, _utc_now_iso()),
            )
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def revoke_access(self, scope_id: str, principal: str, permission: str | None = None) -> int:
        """Revoke permissions from a principal for a scope.

        If permission is None, revokes all permissions.
        Returns number of revoked grants.
        """
        if permission:
            cursor = self.conn.execute(
                "DELETE FROM scope_access WHERE scope_id = ? AND principal = ? AND permission = ?",
                (scope_id, principal, permission),
            )
        else:
            cursor = self.conn.execute(
                "DELETE FROM scope_access WHERE scope_id = ? AND principal = ?",
                (scope_id, principal),
            )
        self.conn.commit()
        return cursor.rowcount

    def check_access(self, scope_id: str, principal: str, permission: str = "read") -> bool:
        """Check if a principal has a specific permission for a scope.

        Admin permission implies all others.
        """
        row = self.conn.execute(
            "SELECT COUNT(*) FROM scope_access WHERE scope_id = ? AND principal = ? AND (permission = ? OR permission = 'admin')",
            (scope_id, principal, permission),
        ).fetchone()
        return row[0] > 0

    def list_scope_grants(self, scope_id: str) -> list[dict]:
        """List all access grants for a scope."""
        rows = self.conn.execute(
            "SELECT scope_id, principal, permission, created_at FROM scope_access WHERE scope_id = ? ORDER BY principal, permission",
            (scope_id,),
        ).fetchall()
        return [
            {"scope_id": r["scope_id"], "principal": r["principal"], "permission": r["permission"], "created_at": r["created_at"]}
            for r in rows
        ]

    def list_principal_scopes(self, principal: str) -> list[dict]:
        """List all scopes a principal has access to."""
        rows = self.conn.execute(
            "SELECT scope_id, principal, permission, created_at FROM scope_access WHERE principal = ? ORDER BY scope_id, permission",
            (principal,),
        ).fetchall()
        return [
            {"scope_id": r["scope_id"], "principal": r["principal"], "permission": r["permission"], "created_at": r["created_at"]}
            for r in rows
        ]

    # --- Memory links / dependency tracking ---

    def add_link(self, source_id: str, target_id: str, link_type: str = "related") -> bool:
        """Create a directed link between two memories.

        Link types: 'related', 'depends_on', 'elaborates', 'contradicts'.
        Returns True if newly created, False if already existed.
        """
        try:
            self.conn.execute(
                "INSERT INTO memory_links (source_id, target_id, link_type, created_at) VALUES (?, ?, ?, ?)",
                (source_id, target_id, link_type, _utc_now_iso()),
            )
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def remove_link(self, source_id: str, target_id: str, link_type: str | None = None) -> int:
        """Remove a link between two memories. If link_type is None, removes all links."""
        if link_type:
            cursor = self.conn.execute(
                "DELETE FROM memory_links WHERE source_id = ? AND target_id = ? AND link_type = ?",
                (source_id, target_id, link_type),
            )
        else:
            cursor = self.conn.execute(
                "DELETE FROM memory_links WHERE source_id = ? AND target_id = ?",
                (source_id, target_id),
            )
        self.conn.commit()
        return cursor.rowcount

    def get_links(self, memory_id: str, direction: str = "both") -> list[dict]:
        """Get all links for a memory.

        direction: 'outgoing' (source=memory_id), 'incoming' (target=memory_id), 'both'.
        """
        results = []
        if direction in ("outgoing", "both"):
            rows = self.conn.execute(
                "SELECT source_id, target_id, link_type, created_at FROM memory_links WHERE source_id = ?",
                (memory_id,),
            ).fetchall()
            results.extend(
                {"source_id": r["source_id"], "target_id": r["target_id"], "link_type": r["link_type"], "direction": "outgoing"}
                for r in rows
            )
        if direction in ("incoming", "both"):
            rows = self.conn.execute(
                "SELECT source_id, target_id, link_type, created_at FROM memory_links WHERE target_id = ?",
                (memory_id,),
            ).fetchall()
            results.extend(
                {"source_id": r["source_id"], "target_id": r["target_id"], "link_type": r["link_type"], "direction": "incoming"}
                for r in rows
            )
        return results

    def get_linked_memories(self, memory_id: str, link_type: str | None = None) -> list[MemoryUnit]:
        """Get all memory units linked to a given memory (both directions).

        Optionally filter by link type.
        """
        if link_type:
            rows = self.conn.execute(
                """SELECT DISTINCT m.* FROM memories m
                   JOIN memory_links l ON (m.memory_id = l.target_id AND l.source_id = ?)
                      OR (m.memory_id = l.source_id AND l.target_id = ?)
                   WHERE l.link_type = ? AND m.status = 'active'""",
                (memory_id, memory_id, link_type),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """SELECT DISTINCT m.* FROM memories m
                   JOIN memory_links l ON (m.memory_id = l.target_id AND l.source_id = ?)
                      OR (m.memory_id = l.source_id AND l.target_id = ?)
                   WHERE m.status = 'active'""",
                (memory_id, memory_id),
            ).fetchall()
        return [self._row_to_unit(r) for r in rows]


    def sample_memories(self, scope_id: str, count: int = 5) -> list[MemoryUnit]:
        """Return a random sample of active memories for exploration/testing."""
        units = self.list_active(scope_id, limit=500)
        if len(units) <= count:
            return units
        import random
        return random.sample(units, count)

    # --- Memory watches ---

    def add_watch(self, memory_id: str, watcher: str) -> bool:
        """Watch a memory for changes. Returns True if newly added."""
        try:
            self.conn.execute(
                "INSERT INTO memory_watches (memory_id, watcher, created_at) VALUES (?, ?, ?)",
                (memory_id, watcher, _utc_now_iso()),
            )
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def remove_watch(self, memory_id: str, watcher: str) -> bool:
        """Stop watching a memory."""
        cursor = self.conn.execute(
            "DELETE FROM memory_watches WHERE memory_id = ? AND watcher = ?",
            (memory_id, watcher),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def get_watchers(self, memory_id: str) -> list[str]:
        """Get all watchers for a memory."""
        rows = self.conn.execute(
            "SELECT watcher FROM memory_watches WHERE memory_id = ? ORDER BY watcher",
            (memory_id,),
        ).fetchall()
        return [r["watcher"] for r in rows]

    def get_watched_memories(self, watcher: str) -> list[str]:
        """Get all memory IDs watched by a watcher."""
        rows = self.conn.execute(
            "SELECT memory_id FROM memory_watches WHERE watcher = ? ORDER BY memory_id",
            (watcher,),
        ).fetchall()
        return [r["memory_id"] for r in rows]

    # --- Memory annotations ---

    def add_annotation(self, memory_id: str, content: str, author: str = "") -> int:
        """Add an annotation to a memory. Returns the annotation ID."""
        cursor = self.conn.execute(
            "INSERT INTO memory_annotations (memory_id, author, content, created_at) VALUES (?, ?, ?, ?)",
            (memory_id, author, content, _utc_now_iso()),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_annotations(self, memory_id: str) -> list[dict]:
        """Get all annotations for a memory, ordered by creation time."""
        rows = self.conn.execute(
            "SELECT annotation_id, memory_id, author, content, created_at FROM memory_annotations WHERE memory_id = ? ORDER BY created_at",
            (memory_id,),
        ).fetchall()
        return [
            {"annotation_id": r["annotation_id"], "memory_id": r["memory_id"], "author": r["author"],
             "content": r["content"], "created_at": r["created_at"]}
            for r in rows
        ]

    def delete_annotation(self, annotation_id: int) -> bool:
        """Delete a specific annotation by ID."""
        cursor = self.conn.execute(
            "DELETE FROM memory_annotations WHERE annotation_id = ?",
            (annotation_id,),
        )
        self.conn.commit()
        return cursor.rowcount > 0


def _json_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except Exception:
        return []
    return [str(x) for x in data] if isinstance(data, list) else []


def _json_float_list(raw: str | None) -> list[float]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    out: list[float] = []
    for item in data:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            return []
    return out


def _log2(x: float) -> float:
    return math.log2(max(x, 1.0))


def _tokenize(text: str) -> list[str]:
    token = []
    out: list[str] = []
    for ch in text.lower():
        if ch.isalnum() or ch in {"_", "-"}:
            token.append(ch)
            continue
        if token:
            out.append("".join(token))
            token = []
    if token:
        out.append("".join(token))
    return out
