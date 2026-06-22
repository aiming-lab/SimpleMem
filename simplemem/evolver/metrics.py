from __future__ import annotations

from .store import MemoryStore


def summarize_memory_store(store: MemoryStore, scope_id: str) -> dict:
    stats = store.get_stats(scope_id)
    active_by_type = stats.get("active_by_type", {})
    active = int(stats.get("active", 0))
    total = int(stats.get("total", 0))

    stats["memory_density"] = round(
        float(active) / max(float(total), 1.0),
        4,
    )
    stats["dominant_type"] = (
        max(active_by_type, key=active_by_type.get)
        if active_by_type
        else ""
    )

    # Type diversity: number of distinct active types.
    stats["type_count"] = len(active_by_type)

    # Type distribution ratios.
    if active > 0 and active_by_type:
        stats["type_ratios"] = {
            t: round(count / float(active), 4)
            for t, count in active_by_type.items()
        }
    else:
        stats["type_ratios"] = {}

    # Superseded count (for GC awareness).
    stats["superseded"] = max(total - active, 0)

    return stats
