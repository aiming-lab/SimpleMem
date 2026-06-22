"""
Utilities for loading OpenVision models via OpenVision's open_clip package.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_OPENVISION_MODEL = "hf-hub:UCSC-VLAA/openvision-vit-large-patch14-224"
_OPENVISION_MODEL_ALIASES = {
    "openvision-l-14": DEFAULT_OPENVISION_MODEL,
    "openvision-l14": DEFAULT_OPENVISION_MODEL,
    "openvision-vit-large-patch14-224": DEFAULT_OPENVISION_MODEL,
}


def is_openvision_model(model_name: str) -> bool:
    """Return whether a model name refers to the OpenVision family."""
    normalized = (model_name or "").strip().lower()
    return "openvision" in normalized or normalized in _OPENVISION_MODEL_ALIASES


def resolve_openvision_model_name(model_name: Optional[str]) -> str:
    """Normalize aliases and convert OpenVision HF repos to hf-hub format."""
    raw_name = (model_name or "").strip()
    if not raw_name:
        return DEFAULT_OPENVISION_MODEL

    lowered = raw_name.lower()
    if raw_name.startswith("hf-hub:"):
        return raw_name

    if lowered in _OPENVISION_MODEL_ALIASES:
        return _OPENVISION_MODEL_ALIASES[lowered]

    if lowered.startswith("ucsc-vlaa/openvision"):
        return f"hf-hub:{raw_name}"

    if lowered.startswith("openvision"):
        return f"hf-hub:UCSC-VLAA/{raw_name}"

    return raw_name


def _candidate_openvision_roots() -> List[Path]:
    candidates: List[Path] = []
    env_path = os.getenv("OPENVISION_CODEBASE_PATH") or os.getenv("OPENVISION_REPO_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    cwd = Path.cwd()
    candidates.extend(
        [
            cwd / "OpenVision",
            cwd.parent / "OpenVision",
            cwd / "third_party" / "OpenVision",
        ]
    )
    return candidates


def _inject_openvision_paths() -> None:
    """
    Add OpenVision import paths to sys.path if a local codebase exists.

    This enables importing:
    - src.convert_upload.open_clip
    - open_clip (from OpenVision's bundled open_clip package)
    """
    for repo_root in _candidate_openvision_roots():
        open_clip_dir = repo_root / "src" / "convert_upload" / "open_clip"
        convert_upload_dir = repo_root / "src" / "convert_upload"

        if not open_clip_dir.exists():
            continue

        root_str = str(repo_root)
        convert_upload_str = str(convert_upload_dir)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        if convert_upload_str not in sys.path:
            sys.path.insert(0, convert_upload_str)

        logger.info("Detected OpenVision codebase at %s", repo_root)
        return


def _import_create_model_from_pretrained() -> Callable[..., Any]:
    """
    Import OpenVision's create_model_from_pretrained API.

    Preference order:
    1) src.convert_upload.open_clip (OpenVision repository)
    2) open_clip (if OpenVision open_clip is importable via PYTHONPATH)
    3) open_clip_torch fallback
    """
    _inject_openvision_paths()

    import_errors: List[str] = []

    try:
        from src.convert_upload.open_clip import create_model_from_pretrained  # type: ignore

        return create_model_from_pretrained
    except Exception as exc:  # pragma: no cover - depends on local env
        import_errors.append(f"src.convert_upload.open_clip import failed: {exc}")

    try:
        from src.convert_upload.open_clip.factory import create_model_from_pretrained  # type: ignore

        return create_model_from_pretrained
    except Exception as exc:  # pragma: no cover - depends on local env
        import_errors.append(f"src.convert_upload.open_clip.factory import failed: {exc}")

    try:
        from open_clip import create_model_from_pretrained  # type: ignore

        logger.warning(
            "Falling back to import `open_clip` directly. "
            "Set OPENVISION_CODEBASE_PATH to ensure OpenVision's customized open_clip is used."
        )
        return create_model_from_pretrained
    except Exception as exc:
        import_errors.append(f"open_clip import failed: {exc}")

    raise ImportError(
        "Could not import OpenVision open_clip. "
        "Clone https://github.com/UCSC-VLAA/OpenVision and set OPENVISION_CODEBASE_PATH, "
        "or install open_clip_torch. "
        f"Errors: {' | '.join(import_errors)}"
    )


def _import_get_tokenizer() -> Callable[..., Any]:
    """Import OpenVision's get_tokenizer API."""
    _inject_openvision_paths()

    import_errors: List[str] = []

    try:
        from src.convert_upload.open_clip import get_tokenizer  # type: ignore

        return get_tokenizer
    except Exception as exc:  # pragma: no cover - depends on local env
        import_errors.append(f"src.convert_upload.open_clip import failed: {exc}")

    try:
        from src.convert_upload.open_clip.factory import get_tokenizer  # type: ignore

        return get_tokenizer
    except Exception as exc:  # pragma: no cover - depends on local env
        import_errors.append(f"src.convert_upload.open_clip.factory import failed: {exc}")

    try:
        from open_clip import get_tokenizer  # type: ignore

        logger.warning(
            "Falling back to import `open_clip` directly. "
            "Set OPENVISION_CODEBASE_PATH to ensure OpenVision's customized open_clip is used."
        )
        return get_tokenizer
    except Exception as exc:
        import_errors.append(f"open_clip import failed: {exc}")

    raise ImportError(
        "Could not import OpenVision get_tokenizer. "
        "Clone https://github.com/UCSC-VLAA/OpenVision and set OPENVISION_CODEBASE_PATH, "
        "or install open_clip_torch. "
        f"Errors: {' | '.join(import_errors)}"
    )


def load_openvision_clip_model(
    model_name: Optional[str],
    device: str = "cpu",
) -> Tuple[Any, Any, str]:
    """Load an OpenVision CLIP model and preprocess transform."""
    create_model_from_pretrained = _import_create_model_from_pretrained()
    resolved_model_name = resolve_openvision_model_name(model_name)

    model_and_preprocess = create_model_from_pretrained(
        model_name=resolved_model_name,
        device=device,
    )

    if isinstance(model_and_preprocess, tuple):
        model = model_and_preprocess[0]
        preprocess = model_and_preprocess[1]
    else:
        model = model_and_preprocess
        preprocess = None

    if preprocess is None:
        raise RuntimeError(
            "OpenVision open_clip did not return a preprocess transform. "
            "Please use a create_model_from_pretrained implementation that returns (model, preprocess)."
        )

    model.eval()
    return model, preprocess, resolved_model_name


def load_openvision_tokenizer(model_name: Optional[str]) -> Tuple[Any, str]:
    """Load OpenVision tokenizer for a given model."""
    get_tokenizer = _import_get_tokenizer()
    resolved_model_name = resolve_openvision_model_name(model_name)
    tokenizer = get_tokenizer(resolved_model_name)
    return tokenizer, resolved_model_name


def to_pil_image(image: Any):
    """Convert path/array/PIL-like input to PIL.Image.Image."""
    from PIL import Image

    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")

    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")

    # PIL Image and compatible objects pass through.
    if hasattr(image, "convert"):
        return image.convert("RGB")

    if hasattr(image, "__array__"):
        return Image.fromarray(np.array(image)).convert("RGB")

    raise ValueError(f"Unsupported image format: {type(image)}")


def extract_openvision_image_embedding(
    model: Any,
    preprocess: Any,
    image: Any,
    device: str,
) -> np.ndarray:
    """Extract a single OpenVision image embedding."""
    import torch

    pil_image = to_pil_image(image)
    image_tensor = preprocess(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        if hasattr(model, "encode_image"):
            outputs = model.encode_image(image_tensor)
        else:
            outputs = model(image_tensor)

        if isinstance(outputs, tuple):
            outputs = outputs[0]

    return outputs.detach().cpu().numpy().flatten()


def extract_openvision_image_embeddings_batch(
    model: Any,
    preprocess: Any,
    images: List[Any],
    device: str,
) -> np.ndarray:
    """Extract batched OpenVision image embeddings."""
    import torch

    if not images:
        return np.zeros((0, 0), dtype=np.float32)

    image_tensors = [preprocess(to_pil_image(img)) for img in images]
    batch = torch.stack(image_tensors, dim=0).to(device)

    with torch.no_grad():
        if hasattr(model, "encode_image"):
            outputs = model.encode_image(batch)
        else:
            outputs = model(batch)
        if isinstance(outputs, tuple):
            outputs = outputs[0]

    return outputs.detach().cpu().numpy()
