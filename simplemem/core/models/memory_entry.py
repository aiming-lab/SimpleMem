"""
Core Data Structure - MemoryEntry (Memory Unit)

Section 3.1: Semantic Structured Compression
Each MemoryEntry represents a compact, context-independent memory unit
with multi-view indexing (Semantic, Lexical, Symbolic layers)
"""
from typing import List, Optional
from pydantic import BaseModel, Field
import uuid


class MemoryEntry(BaseModel):
    """
    Memory Unit - Self-contained entry indexed via multi-view indexing (Section 3.1)

    Indexed via: I(m_k) = {s_k (Semantic), l_k (Lexical), r_k (Symbolic)}
    """
    entry_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # [Semantic Layer] - Dense embedding base (v_k = E_dense(S_k))
    lossless_restatement: str = Field(
        ...,
        description="Self-contained fact with Φ_coref (no pronouns) and Φ_time (absolute timestamps)"
    )

    # [Lexical Layer] - Sparse keyword vectors (h_k = Sparse(S_k))
    keywords: List[str] = Field(
        default_factory=list,
        description="Core keywords for BM25-style exact matching"
    )

    # [Symbolic Layer] - Metadata constraints (R_k = {(key, val)})
    timestamp: Optional[str] = Field(
        None,
        description="Standardized time in ISO 8601 format (YYYY-MM-DDTHH:MM:SS)"
    )
    location: Optional[str] = Field(
        None,
        description="Natural language location description"
    )
    persons: List[str] = Field(
        default_factory=list,
        description="List of extracted persons"
    )
    entities: List[str] = Field(
        default_factory=list,
        description="List of extracted entities (companies, products, etc.)"
    )
    topic: Optional[str] = Field(
        None,
        description="Topic phrase summarized by LLM"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "entry_id": "550e8400-e29b-41d4-a716-446655440000",
                "lossless_restatement": "Alice discussed the marketing strategy for new product XYZ with Bob at Starbucks in Shanghai on November 15, 2025 at 14:30.",
                "keywords": ["Alice", "Bob", "product XYZ", "marketing strategy", "discussion"],
                "timestamp": "2025-11-15T14:30:00",
                "location": "Starbucks, Shanghai",
                "persons": ["Alice", "Bob"],
                "entities": ["product XYZ"],
                "topic": "Product marketing strategy discussion"
            }
        }


class Dialogue(BaseModel):
    """
    Original dialogue entry
    """
    dialogue_id: int
    speaker: str
    content: str
    timestamp: Optional[str] = None  # ISO 8601 format

    def __str__(self) -> str:
        time_str = f"[{self.timestamp}] " if self.timestamp else ""
        return f"{time_str}{self.speaker}: {self.content}"
