"""
Graph-based Retriever for Omni-Memory.

Combines vector search with knowledge graph traversal for
enhanced semantic retrieval.
"""

import logging
from typing import Optional, List, Dict, Any, Tuple, Set
from dataclasses import dataclass, field

from simplemem.multimodal.knowledge.knowledge_graph import KnowledgeGraph, Entity, Relation
from simplemem.multimodal.knowledge.entity_extractor import EntityExtractor

logger = logging.getLogger(__name__)


@dataclass
class GraphRetrievalResult:
    """Result from graph-based retrieval."""
    query: str
    
    # Matched entities
    entities: List[Entity] = field(default_factory=list)
    
    # Related entities (via graph traversal)
    related_entities: List[Tuple[Entity, int, str]] = field(default_factory=list)  # (entity, hops, relation_path)
    
    # Relations found
    relations: List[Relation] = field(default_factory=list)
    
    # Source MAU IDs to retrieve
    mau_ids: List[str] = field(default_factory=list)
    
    # Subgraph for visualization/reasoning
    subgraph: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "entities": [e.to_dict() for e in self.entities],
            "related_entities": [
                {"entity": e.to_dict(), "hops": h, "path": p}
                for e, h, p in self.related_entities
            ],
            "relations": [r.to_dict() for r in self.relations],
            "mau_ids": self.mau_ids,
            "subgraph": self.subgraph,
        }


class GraphRetriever:
    """
    Graph-based Retriever for Knowledge-Enhanced Retrieval.
    
    Combines:
    - Query entity extraction
    - Knowledge graph traversal
    - Multi-hop reasoning
    - MAU linking for evidence retrieval
    
    This enables semantic queries like:
    - "What happened near the coffee shop?" (location-based)
    - "Who did I meet at the conference?" (entity + event)
    - "Show me all interactions with John" (person-centric)
    """
    
    def __init__(
        self,
        knowledge_graph: KnowledgeGraph,
        entity_extractor: Optional[EntityExtractor] = None,
        config: Optional[Any] = None,
    ):
        self.kg = knowledge_graph
        self.extractor = entity_extractor or EntityExtractor(config=config)
        self.config = config
    
    def retrieve(
        self,
        query: str,
        max_entities: int = 5,
        max_hops: int = 2,
        include_subgraph: bool = False,
    ) -> GraphRetrievalResult:
        """
        Retrieve relevant knowledge from the graph based on query.
        
        Process:
        1. Extract entities from query
        2. Match to known entities in graph
        3. Traverse graph to find related entities
        4. Collect source MAU IDs
        
        Args:
            query: Natural language query
            max_entities: Max number of seed entities
            max_hops: Max graph traversal depth
            include_subgraph: Include subgraph for visualization
            
        Returns:
            GraphRetrievalResult
        """
        result = GraphRetrievalResult(query=query)
        
        # Step 1: Extract entities from query
        extraction = self.extractor.extract(query)
        query_entities = extraction.entities
        
        # Step 2: Match to known entities
        matched_entities = []
        for qe in query_entities:
            entity = self.kg.get_entity_by_name(qe.name)
            if entity:
                matched_entities.append(entity)
            else:
                # Try fuzzy search
                candidates = self.kg.search_entities(qe.name, limit=1)
                if candidates:
                    matched_entities.append(candidates[0])
        
        # Also try direct name search from query terms
        if not matched_entities:
            # Fallback: search for keywords
            words = query.lower().split()
            for word in words:
                if len(word) > 3:  # Skip short words
                    candidates = self.kg.search_entities(word, limit=2)
                    matched_entities.extend(candidates)
        
        # Deduplicate
        seen_ids = set()
        unique_matched = []
        for e in matched_entities[:max_entities]:
            if e.entity_id not in seen_ids:
                seen_ids.add(e.entity_id)
                unique_matched.append(e)
        
        result.entities = unique_matched
        
        # Step 3: Traverse graph for related entities
        all_entity_ids = set(e.entity_id for e in unique_matched)
        related_with_path = []
        
        for seed_entity in unique_matched:
            neighbors = self.kg.get_neighbors(
                seed_entity.entity_id,
                max_hops=max_hops,
            )
            
            for neighbor_id, (neighbor, hops) in neighbors.items():
                if neighbor and neighbor_id not in all_entity_ids:
                    # Find path description
                    path = self._get_path_description(seed_entity.entity_id, neighbor_id)
                    related_with_path.append((neighbor, hops, path))
                    all_entity_ids.add(neighbor_id)
        
        # Sort by hop distance
        related_with_path.sort(key=lambda x: x[1])
        result.related_entities = related_with_path[:max_entities * 2]
        
        # Step 4: Collect relations
        relations = []
        for eid in all_entity_ids:
            rels = self.kg.get_relations_for_entity(eid)
            for rel in rels:
                if rel.subject_id in all_entity_ids and rel.object_id in all_entity_ids:
                    if rel not in relations:
                        relations.append(rel)
        result.relations = relations
        
        # Step 5: Collect source MAU IDs
        mau_ids = set()
        for e in result.entities:
            mau_ids.update(e.source_mau_ids)
        for e, _, _ in result.related_entities:
            mau_ids.update(e.source_mau_ids)
        for r in result.relations:
            mau_ids.update(r.source_mau_ids)
        
        result.mau_ids = list(mau_ids)
        
        # Step 6: Build subgraph if requested
        if include_subgraph:
            result.subgraph = self.kg.get_subgraph(
                list(all_entity_ids),
                include_relations=True,
            )
        
        return result
    
    def _get_path_description(
        self,
        from_id: str,
        to_id: str,
    ) -> str:
        """Get description of path between two entities."""
        # Simple: find direct relation
        relations = self.kg.get_relations_for_entity(from_id, direction="outgoing")
        
        for rel in relations:
            if rel.object_id == to_id:
                return rel.predicate
        
        # Check incoming
        relations = self.kg.get_relations_for_entity(from_id, direction="incoming")
        for rel in relations:
            if rel.subject_id == to_id:
                return f"inverse_{rel.predicate}"
        
        return "related"
    
    def retrieve_by_entity_type(
        self,
        entity_type: str,
        limit: int = 10,
    ) -> GraphRetrievalResult:
        """
        Retrieve all entities of a specific type.
        
        Useful for queries like "Show me all people" or "List all locations".
        """
        entities = self.kg.get_entities_by_type(entity_type)[:limit]
        
        mau_ids = set()
        for e in entities:
            mau_ids.update(e.source_mau_ids)
        
        return GraphRetrievalResult(
            query=f"All {entity_type} entities",
            entities=entities,
            mau_ids=list(mau_ids),
        )
    
    def retrieve_by_mau(
        self,
        mau_id: str,
    ) -> GraphRetrievalResult:
        """
        Get all knowledge linked to a specific MAU.
        
        Useful for understanding what entities/relations were extracted
        from a particular memory.
        """
        entities = self.kg.get_entities_by_mau(mau_id)
        
        # Get relations involving these entities
        relations = []
        entity_ids = set(e.entity_id for e in entities)
        for e in entities:
            rels = self.kg.get_relations_for_entity(e.entity_id)
            for rel in rels:
                if rel.subject_id in entity_ids or rel.object_id in entity_ids:
                    if rel not in relations:
                        relations.append(rel)
        
        return GraphRetrievalResult(
            query=f"Knowledge from MAU {mau_id}",
            entities=entities,
            relations=relations,
            mau_ids=[mau_id],
        )
    
    def reason_multi_hop(
        self,
        start_entity: str,
        target_type: Optional[str] = None,
        relation_path: Optional[List[str]] = None,
        max_hops: int = 3,
    ) -> List[Tuple[Entity, List[Relation]]]:
        """
        Multi-hop reasoning through the graph.
        
        Find entities reachable from start_entity following
        specific relation paths or entity types.
        
        Example:
        - Start: "John", Target type: "LOCATION" -> Find all locations John visited
        - Start: "Conference", Relations: ["attended_by", "works_at"] -> 
          Find organizations of people who attended the conference
        """
        start = self.kg.get_entity_by_name(start_entity)
        if not start:
            return []
        
        # BFS with path tracking
        results = []
        visited = {start.entity_id}
        queue = [(start, [])]  # (entity, relation_path)
        
        while queue:
            current, path = queue.pop(0)
            
            if len(path) >= max_hops:
                continue
            
            relations = self.kg.get_relations_for_entity(current.entity_id)
            
            for rel in relations:
                # Check relation path constraint
                if relation_path and len(path) < len(relation_path):
                    if rel.predicate != relation_path[len(path)]:
                        continue
                
                # Get neighbor
                neighbor_id = (
                    rel.object_id if rel.subject_id == current.entity_id 
                    else rel.subject_id
                )
                
                if neighbor_id in visited:
                    continue
                
                neighbor = self.kg.get_entity(neighbor_id)
                if not neighbor:
                    continue
                
                new_path = path + [rel]
                
                # Check target type constraint
                if target_type and neighbor.entity_type == target_type:
                    results.append((neighbor, new_path))
                elif not target_type and len(new_path) >= 1:
                    results.append((neighbor, new_path))
                
                visited.add(neighbor_id)
                queue.append((neighbor, new_path))
        
        return results
    
    def format_for_llm(
        self,
        result: GraphRetrievalResult,
        max_entities: int = 10,
        include_relations: bool = True,
    ) -> str:
        """
        Format graph retrieval result for LLM context.
        """
        parts = []
        
        # Add matched entities
        if result.entities:
            parts.append("Relevant entities found:")
            for i, e in enumerate(result.entities[:max_entities], 1):
                attrs = ", ".join(f"{k}={v}" for k, v in e.attributes.items()) if e.attributes else ""
                parts.append(f"  {i}. {e.name} ({e.entity_type}){': ' + attrs if attrs else ''}")
        
        # Add related entities
        if result.related_entities:
            parts.append("\nRelated entities:")
            for e, hops, path in result.related_entities[:max_entities]:
                parts.append(f"  - {e.name} ({e.entity_type}) via {path} ({hops} hops)")
        
        # Add relations
        if include_relations and result.relations:
            parts.append("\nKnown relationships:")
            for rel in result.relations[:max_entities]:
                subj = self.kg.get_entity(rel.subject_id)
                obj = self.kg.get_entity(rel.object_id)
                if subj and obj:
                    parts.append(f"  - {subj.name} --[{rel.predicate}]--> {obj.name}")
        
        if not parts:
            return "No relevant knowledge found in the graph."
        
        return "\n".join(parts)
