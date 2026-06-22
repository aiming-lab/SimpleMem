"""
Knowledge Graph for Omni-Memory.

Provides semantic storage and retrieval of entities and relations
extracted from multimodal content.
"""

import logging
import json
import time
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import threading

from simplemem.multimodal.knowledge.entity_extractor import ExtractedEntity, ExtractedRelation

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """
    Entity node in the knowledge graph.
    
    Represents a semantic entity extracted from multimodal content.
    """
    entity_id: str = field(default_factory=lambda: f"ent_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}")
    name: str = ""
    entity_type: str = "CONCEPT"
    
    # Attributes and metadata
    attributes: Dict[str, Any] = field(default_factory=dict)
    aliases: List[str] = field(default_factory=list)
    
    # Provenance tracking
    source_mau_ids: List[str] = field(default_factory=list)
    first_seen: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    mention_count: int = 1
    
    # Embeddings for semantic search
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "entity_type": self.entity_type,
            "attributes": self.attributes,
            "aliases": self.aliases,
            "source_mau_ids": self.source_mau_ids,
            "first_seen": self.first_seen,
            "last_updated": self.last_updated,
            "mention_count": self.mention_count,
            "embedding": self.embedding,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        return cls(**data)
    
    @classmethod
    def from_extracted(cls, extracted: ExtractedEntity) -> "Entity":
        """Create Entity from ExtractedEntity."""
        return cls(
            name=extracted.name,
            entity_type=extracted.entity_type,
            attributes=extracted.attributes,
            source_mau_ids=[extracted.source_mau_id] if extracted.source_mau_id else [],
        )
    
    def merge(self, other: "Entity") -> None:
        """Merge another entity into this one."""
        # Merge source MAUs
        for mau_id in other.source_mau_ids:
            if mau_id not in self.source_mau_ids:
                self.source_mau_ids.append(mau_id)
        
        # Merge attributes
        for key, value in other.attributes.items():
            if key not in self.attributes:
                self.attributes[key] = value
        
        # Merge aliases
        if other.name.lower() != self.name.lower():
            if other.name not in self.aliases:
                self.aliases.append(other.name)
        
        for alias in other.aliases:
            if alias not in self.aliases and alias.lower() != self.name.lower():
                self.aliases.append(alias)
        
        # Update stats
        self.mention_count += other.mention_count
        self.last_updated = time.time()
        self.first_seen = min(self.first_seen, other.first_seen)


@dataclass
class Relation:
    """
    Relation edge in the knowledge graph.
    
    Represents a semantic relation between two entities.
    """
    relation_id: str = field(default_factory=lambda: f"rel_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}")
    subject_id: str = ""
    predicate: str = ""
    object_id: str = ""
    
    # Attributes and metadata
    attributes: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    
    # Provenance
    source_mau_ids: List[str] = field(default_factory=list)
    first_seen: float = field(default_factory=time.time)
    mention_count: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "relation_id": self.relation_id,
            "subject_id": self.subject_id,
            "predicate": self.predicate,
            "object_id": self.object_id,
            "attributes": self.attributes,
            "confidence": self.confidence,
            "source_mau_ids": self.source_mau_ids,
            "first_seen": self.first_seen,
            "mention_count": self.mention_count,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relation":
        return cls(**data)
    
    def get_triple(self) -> Tuple[str, str, str]:
        """Get (subject_id, predicate, object_id) triple."""
        return (self.subject_id, self.predicate, self.object_id)


class KnowledgeGraph:
    """
    Semantic Knowledge Graph for Omni-Memory.
    
    Provides:
    - Entity storage and retrieval
    - Relation management
    - Graph-based reasoning
    - Multi-hop traversal
    - Entity resolution and merging
    
    Key Features:
    - Supports multimodal entity linking (text + visual)
    - Tracks provenance to source MAUs
    - Enables cross-modal knowledge reasoning
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        config: Optional[Any] = None,
    ):
        self.config = config
        self.storage_path = Path(storage_path) if storage_path else Path("./omni_memory_data/knowledge_graph")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Entity storage
        self._entities: Dict[str, Entity] = {}
        self._entity_name_index: Dict[str, str] = {}  # normalized_name -> entity_id
        self._entity_type_index: Dict[str, Set[str]] = defaultdict(set)  # type -> entity_ids
        self._mau_entity_index: Dict[str, Set[str]] = defaultdict(set)  # mau_id -> entity_ids
        
        # Relation storage
        self._relations: Dict[str, Relation] = {}
        self._subject_index: Dict[str, Set[str]] = defaultdict(set)  # entity_id -> relation_ids
        self._object_index: Dict[str, Set[str]] = defaultdict(set)  # entity_id -> relation_ids
        self._predicate_index: Dict[str, Set[str]] = defaultdict(set)  # predicate -> relation_ids
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Load existing data
        self._load()
    
    def _normalize_name(self, name: str) -> str:
        """Normalize entity name for matching."""
        return name.lower().strip()
    
    def add_entity(self, entity: Entity) -> str:
        """
        Add an entity to the graph.
        
        If similar entity exists, merges them.
        
        Returns:
            Entity ID
        """
        with self._lock:
            normalized = self._normalize_name(entity.name)
            
            # Check for existing entity
            if normalized in self._entity_name_index:
                existing_id = self._entity_name_index[normalized]
                existing = self._entities[existing_id]
                existing.merge(entity)
                self._update_indices(existing)
                return existing_id
            
            # Add new entity
            self._entities[entity.entity_id] = entity
            self._entity_name_index[normalized] = entity.entity_id
            self._update_indices(entity)
            
            logger.debug(f"Added entity: {entity.name} ({entity.entity_id})")
            return entity.entity_id
    
    def add_extracted_entity(self, extracted: ExtractedEntity) -> str:
        """Add entity from extraction result."""
        entity = Entity.from_extracted(extracted)
        return self.add_entity(entity)
    
    def _update_indices(self, entity: Entity) -> None:
        """Update indices for an entity."""
        # Type index
        self._entity_type_index[entity.entity_type].add(entity.entity_id)
        
        # MAU index
        for mau_id in entity.source_mau_ids:
            self._mau_entity_index[mau_id].add(entity.entity_id)
        
        # Alias index
        for alias in entity.aliases:
            normalized = self._normalize_name(alias)
            if normalized not in self._entity_name_index:
                self._entity_name_index[normalized] = entity.entity_id
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        return self._entities.get(entity_id)
    
    def get_entity_by_name(self, name: str) -> Optional[Entity]:
        """Get entity by name (case-insensitive)."""
        normalized = self._normalize_name(name)
        entity_id = self._entity_name_index.get(normalized)
        if entity_id:
            return self._entities.get(entity_id)
        return None
    
    def get_entities_by_type(self, entity_type: str) -> List[Entity]:
        """Get all entities of a type."""
        entity_ids = self._entity_type_index.get(entity_type, set())
        return [self._entities[eid] for eid in entity_ids if eid in self._entities]
    
    def get_entities_by_mau(self, mau_id: str) -> List[Entity]:
        """Get all entities linked to a MAU."""
        entity_ids = self._mau_entity_index.get(mau_id, set())
        return [self._entities[eid] for eid in entity_ids if eid in self._entities]
    
    def add_relation(self, relation: Relation) -> str:
        """
        Add a relation to the graph.
        
        Returns:
            Relation ID
        """
        with self._lock:
            # Verify entities exist
            if relation.subject_id not in self._entities:
                logger.warning(f"Subject entity not found: {relation.subject_id}")
            if relation.object_id not in self._entities:
                logger.warning(f"Object entity not found: {relation.object_id}")
            
            # Check for duplicate relation
            existing = self._find_relation(
                relation.subject_id, 
                relation.predicate,
                relation.object_id
            )
            if existing:
                # Update existing
                existing.mention_count += 1
                existing.confidence = max(existing.confidence, relation.confidence)
                for mau_id in relation.source_mau_ids:
                    if mau_id not in existing.source_mau_ids:
                        existing.source_mau_ids.append(mau_id)
                return existing.relation_id
            
            # Add new relation
            self._relations[relation.relation_id] = relation
            self._subject_index[relation.subject_id].add(relation.relation_id)
            self._object_index[relation.object_id].add(relation.relation_id)
            self._predicate_index[relation.predicate].add(relation.relation_id)
            
            logger.debug(f"Added relation: {relation.subject_id} -{relation.predicate}-> {relation.object_id}")
            return relation.relation_id
    
    def add_extracted_relation(
        self,
        extracted: ExtractedRelation,
        subject_id: Optional[str] = None,
        object_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Add relation from extraction result.
        
        Resolves entity names to IDs if not provided.
        """
        # Resolve subject
        if not subject_id:
            subject_entity = self.get_entity_by_name(extracted.subject)
            if not subject_entity:
                return None
            subject_id = subject_entity.entity_id
        
        # Resolve object  
        if not object_id:
            object_entity = self.get_entity_by_name(extracted.object)
            if not object_entity:
                return None
            object_id = object_entity.entity_id
        
        relation = Relation(
            subject_id=subject_id,
            predicate=extracted.predicate,
            object_id=object_id,
            attributes=extracted.attributes,
            confidence=extracted.confidence,
            source_mau_ids=[extracted.source_mau_id] if extracted.source_mau_id else [],
        )
        
        return self.add_relation(relation)
    
    def _find_relation(
        self,
        subject_id: str,
        predicate: str,
        object_id: str,
    ) -> Optional[Relation]:
        """Find existing relation."""
        subject_rels = self._subject_index.get(subject_id, set())
        for rel_id in subject_rels:
            rel = self._relations.get(rel_id)
            if rel and rel.predicate == predicate and rel.object_id == object_id:
                return rel
        return None
    
    def get_relation(self, relation_id: str) -> Optional[Relation]:
        """Get relation by ID."""
        return self._relations.get(relation_id)
    
    def get_relations_for_entity(
        self,
        entity_id: str,
        direction: str = "both",  # "outgoing", "incoming", "both"
    ) -> List[Relation]:
        """Get all relations involving an entity."""
        relations = []
        
        if direction in ["outgoing", "both"]:
            for rel_id in self._subject_index.get(entity_id, set()):
                rel = self._relations.get(rel_id)
                if rel:
                    relations.append(rel)
        
        if direction in ["incoming", "both"]:
            for rel_id in self._object_index.get(entity_id, set()):
                rel = self._relations.get(rel_id)
                if rel:
                    relations.append(rel)
        
        return relations
    
    def get_neighbors(
        self,
        entity_id: str,
        max_hops: int = 1,
        relation_types: Optional[List[str]] = None,
    ) -> Dict[str, Tuple[Entity, int]]:
        """
        Get neighboring entities within N hops.
        
        Returns:
            Dict mapping entity_id to (entity, hop_distance)
        """
        visited = {entity_id: (self._entities.get(entity_id), 0)}
        frontier = {entity_id}
        
        for hop in range(1, max_hops + 1):
            next_frontier = set()
            
            for eid in frontier:
                relations = self.get_relations_for_entity(eid)
                
                for rel in relations:
                    if relation_types and rel.predicate not in relation_types:
                        continue
                    
                    # Get neighbor ID
                    neighbor_id = rel.object_id if rel.subject_id == eid else rel.subject_id
                    
                    if neighbor_id not in visited:
                        entity = self._entities.get(neighbor_id)
                        if entity:
                            visited[neighbor_id] = (entity, hop)
                            next_frontier.add(neighbor_id)
            
            frontier = next_frontier
            if not frontier:
                break
        
        # Remove starting entity
        del visited[entity_id]
        return visited
    
    def search_entities(
        self,
        query: str,
        entity_types: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[Entity]:
        """
        Search entities by name/alias.
        
        Simple substring matching; for better results use embedding search.
        """
        query_lower = query.lower()
        results = []
        
        with self._lock:
            for entity in self._entities.values():
                if entity_types and entity.entity_type not in entity_types:
                    continue
                
                # Check name and aliases
                if query_lower in entity.name.lower():
                    results.append(entity)
                    continue
                
                for alias in entity.aliases:
                    if query_lower in alias.lower():
                        results.append(entity)
                        break
        
        # Sort by mention count
        results.sort(key=lambda e: e.mention_count, reverse=True)
        return results[:limit]
    
    def get_subgraph(
        self,
        entity_ids: List[str],
        include_relations: bool = True,
    ) -> Dict[str, Any]:
        """
        Get subgraph containing specified entities and their relations.
        """
        entities = []
        relations = []
        
        entity_set = set(entity_ids)
        
        for eid in entity_ids:
            entity = self._entities.get(eid)
            if entity:
                entities.append(entity.to_dict())
        
        if include_relations:
            seen_relations = set()
            for eid in entity_ids:
                for rel in self.get_relations_for_entity(eid):
                    if rel.relation_id not in seen_relations:
                        # Only include if both ends are in the subgraph
                        if rel.subject_id in entity_set and rel.object_id in entity_set:
                            relations.append(rel.to_dict())
                            seen_relations.add(rel.relation_id)
        
        return {
            "entities": entities,
            "relations": relations,
        }
    
    def count_entities(self) -> int:
        """Get total entity count."""
        return len(self._entities)
    
    def count_relations(self) -> int:
        """Get total relation count."""
        return len(self._relations)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        type_counts = {t: len(ids) for t, ids in self._entity_type_index.items()}
        predicate_counts = {p: len(ids) for p, ids in self._predicate_index.items()}
        
        return {
            "total_entities": len(self._entities),
            "total_relations": len(self._relations),
            "entity_types": type_counts,
            "relation_types": predicate_counts,
        }
    
    def save(self) -> None:
        """Save graph to disk."""
        with self._lock:
            # Save entities
            entities_file = self.storage_path / "entities.jsonl"
            with open(entities_file, 'w', encoding='utf-8') as f:
                for entity in self._entities.values():
                    f.write(json.dumps(entity.to_dict(), ensure_ascii=False) + '\n')
            
            # Save relations
            relations_file = self.storage_path / "relations.jsonl"
            with open(relations_file, 'w', encoding='utf-8') as f:
                for relation in self._relations.values():
                    f.write(json.dumps(relation.to_dict(), ensure_ascii=False) + '\n')
        
        logger.info(f"Saved knowledge graph: {len(self._entities)} entities, {len(self._relations)} relations")
    
    def _load(self) -> None:
        """Load graph from disk."""
        entities_file = self.storage_path / "entities.jsonl"
        relations_file = self.storage_path / "relations.jsonl"
        
        # Load entities
        if entities_file.exists():
            with open(entities_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        entity = Entity.from_dict(data)
                        self._entities[entity.entity_id] = entity
                        self._entity_name_index[self._normalize_name(entity.name)] = entity.entity_id
                        self._update_indices(entity)
        
        # Load relations
        if relations_file.exists():
            with open(relations_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        relation = Relation.from_dict(data)
                        self._relations[relation.relation_id] = relation
                        self._subject_index[relation.subject_id].add(relation.relation_id)
                        self._object_index[relation.object_id].add(relation.relation_id)
                        self._predicate_index[relation.predicate].add(relation.relation_id)
        
        if self._entities:
            logger.info(f"Loaded knowledge graph: {len(self._entities)} entities, {len(self._relations)} relations")
    
    def clear(self) -> None:
        """Clear all data."""
        with self._lock:
            self._entities.clear()
            self._entity_name_index.clear()
            self._entity_type_index.clear()
            self._mau_entity_index.clear()
            self._relations.clear()
            self._subject_index.clear()
            self._object_index.clear()
            self._predicate_index.clear()
