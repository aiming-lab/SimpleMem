"""Mini-KG layer for memory retrieval.

Builds entity/topic/time indices over extracted memories and provides
KG-based expansion to augment vector retrieval for multi-hop queries.

Data source: each extracted memory is expected to carry the fields
``persons`` (list[str]), ``topic`` (str|None), ``location`` (str|None),
``timestamp`` (str|None), ``session_id`` (str|None). These are already
produced by the LoCoMo extractor — no extra LLM pass needed offline.

Design principle (post-path-1 diagnosis): Cat 3 bottleneck is retrieval
*precision*, not recall. A widened top-k hurts Cat 3 by injecting noise;
the KG provides precision by letting us pull memories that share
specific entities with the question rather than generic semantic
similarity.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable


def _norm(s: str | None) -> str:
    return (s or "").strip().lower()


@dataclass
class KGIndex:
    """In-memory index over a list of extracted memories.

    Memories are keyed by their position (int) in the input list; the
    caller is responsible for mapping these back to ``RetrievedMemory``
    objects via the same index used for vector retrieval.
    """

    by_person: dict[str, set[int]] = field(default_factory=lambda: defaultdict(set))
    by_topic: dict[str, set[int]] = field(default_factory=lambda: defaultdict(set))
    by_location: dict[str, set[int]] = field(default_factory=lambda: defaultdict(set))
    by_session: dict[str, set[int]] = field(default_factory=lambda: defaultdict(set))
    by_date: dict[str, set[int]] = field(default_factory=lambda: defaultdict(set))
    # canonical name -> list of aliases seen in data; lookup is done via
    # the lowercase form so callers can pass whatever casing they got.
    person_aliases: dict[str, str] = field(default_factory=dict)
    topic_list: list[str] = field(default_factory=list)  # original casing
    n_memories: int = 0

    @classmethod
    def build(cls, memories: list[dict]) -> "KGIndex":
        idx = cls()
        idx.n_memories = len(memories)
        seen_topics: set[str] = set()
        for i, m in enumerate(memories):
            for p in m.get("persons") or []:
                key = _norm(p)
                if not key:
                    continue
                idx.by_person[key].add(i)
                if key not in idx.person_aliases:
                    idx.person_aliases[key] = p
            t = m.get("topic")
            if t:
                idx.by_topic[_norm(t)].add(i)
                if t not in seen_topics:
                    idx.topic_list.append(t)
                    seen_topics.add(t)
            loc = m.get("location")
            if loc:
                idx.by_location[_norm(loc)].add(i)
            sid = m.get("session_id")
            if sid:
                idx.by_session[str(sid)].add(i)
            ts = m.get("timestamp")
            if ts:
                idx.by_date[str(ts)[:10]].add(i)
        return idx

    # ── Lookup ──

    def memories_with_person(self, name: str) -> set[int]:
        return set(self.by_person.get(_norm(name), ()))

    def memories_with_persons_any(self, names: Iterable[str]) -> set[int]:
        out: set[int] = set()
        for n in names:
            out |= self.memories_with_person(n)
        return out

    def memories_with_persons_all(self, names: Iterable[str]) -> set[int]:
        """Intersection — the real bridge operator for multi-hop."""
        names = list(names)
        if not names:
            return set()
        result: set[int] | None = None
        for n in names:
            mem = self.memories_with_person(n)
            result = mem if result is None else result & mem
            if not result:
                return set()
        return result or set()

    def memories_with_topic(self, topic: str) -> set[int]:
        return set(self.by_topic.get(_norm(topic), ()))

    def memories_with_topics_any(self, topics: Iterable[str]) -> set[int]:
        out: set[int] = set()
        for t in topics:
            out |= self.memories_with_topic(t)
        return out

    def memories_with_location(self, loc: str) -> set[int]:
        return set(self.by_location.get(_norm(loc), ()))

    # ── Two-hop bridge: A and B co-appear in which memories? ──

    def bridge_between_persons(self, a: str, b: str) -> set[int]:
        """Memories where both persons co-appear (explicit 2-hop bridge)."""
        return self.memories_with_person(a) & self.memories_with_person(b)

    def bridge_person_topic(self, person: str, topic: str) -> set[int]:
        return self.memories_with_person(person) & self.memories_with_topic(topic)

    # ── Fuzzy topic match via embedder ──

    def topics_similar_to(
        self, query: str, embedder, top_k: int = 5, min_sim: float = 0.55
    ) -> list[tuple[str, float]]:
        """Return top-k topic strings semantically similar to `query`.

        Uses the same embedder the retrieval stack uses. Filters below
        ``min_sim`` so spurious topics don't pollute the expansion.
        """
        if not self.topic_list:
            return []
        try:
            import numpy as np
        except Exception:
            return []
        qv = embedder.encode([query])
        tv = embedder.encode(self.topic_list)
        qv = np.asarray(qv, dtype="float32")
        tv = np.asarray(tv, dtype="float32")
        # cosine: assume embedder returns already-normalized vectors, but
        # normalize defensively.
        qv = qv / (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-9)
        tv = tv / (np.linalg.norm(tv, axis=1, keepdims=True) + 1e-9)
        sims = (tv @ qv.T).flatten()
        order = sims.argsort()[::-1][:top_k]
        return [
            (self.topic_list[i], float(sims[i]))
            for i in order
            if sims[i] >= min_sim
        ]

    # ── Whole-memory expansion ──

    def expand(
        self,
        persons: Iterable[str] = (),
        topics: Iterable[str] = (),
        locations: Iterable[str] = (),
        session_ids: Iterable[str] = (),
    ) -> set[int]:
        """Union expansion across all specified axes."""
        out: set[int] = set()
        for p in persons:
            out |= self.memories_with_person(p)
        for t in topics:
            out |= self.memories_with_topic(t)
        for loc in locations:
            out |= self.memories_with_location(loc)
        for sid in session_ids:
            out |= set(self.by_session.get(str(sid), ()))
        return out


__all__ = ["KGIndex"]
