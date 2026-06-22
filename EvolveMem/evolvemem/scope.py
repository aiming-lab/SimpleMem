from __future__ import annotations


def derive_memory_scope(
    default_scope: str,
    session_id: str = "",
    memory_scope: str = "",
    user_id: str = "",
    workspace_id: str = "",
) -> str:
    explicit = _clean(memory_scope)
    if explicit:
        return explicit

    user = _clean(user_id)
    workspace = _clean(workspace_id)
    if user and workspace:
        return f"user:{user}|workspace:{workspace}"
    if workspace:
        return f"workspace:{workspace}"
    if user:
        return f"user:{user}"

    remembered = _clean(session_id)
    if remembered:
        return f"{_clean(default_scope) or 'default'}|session:{remembered}"
    return _clean(default_scope) or "default"


def base_scope(scope_id: str) -> str:
    """Strip session-specific suffix to get the shared base scope.

    ``"default|session:abc"`` → ``"default"``
    ``"user:alice|workspace:proj|session:xyz"`` → ``"user:alice|workspace:proj"``
    ``"default"`` → ``"default"``
    """
    parts = scope_id.split("|")
    base_parts = [p for p in parts if not p.startswith("session:")]
    return "|".join(base_parts) if base_parts else "default"


def _clean(value: str) -> str:
    cleaned = str(value or "").strip()
    return cleaned.replace("\n", " ").replace("\r", " ")
