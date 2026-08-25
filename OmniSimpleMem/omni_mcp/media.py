"""
Media resolution for the Omni-SimpleMem MCP server.

Turns a user-supplied media reference into a local file path that the
Omni-Memory processors can consume. Supported reference forms:

    /abs/path/img.png            local file
    file:///abs/path/img.png     local file (URI form)
    https://host/img.png         HTTP(S) download
    https://drive.google.com/... Google Drive share link (converted to direct download)
    s3://bucket/key              S3 / MinIO / any S3-compatible endpoint (boto3)
    gs://bucket/key              Google Cloud Storage

Remote backends are imported lazily so the server runs without boto3 or
google-cloud-storage installed; a missing backend produces a clear,
actionable error instead of an import crash at startup.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, parse_qs

# Maximum number of bytes we will pull from a remote source. Guards against a
# mistyped URL streaming a multi-GB object into the agent's temp dir.
DEFAULT_MAX_BYTES = 512 * 1024 * 1024  # 512 MB
DEFAULT_TIMEOUT = 60  # seconds


class MediaError(Exception):
    """Raised when a media reference cannot be resolved."""


@dataclass
class ResolvedMedia:
    """A media reference resolved to a local file."""

    path: str
    source: str
    is_temporary: bool

    def cleanup(self) -> None:
        """Delete the file if we created it in a temp dir."""
        if self.is_temporary:
            try:
                os.unlink(self.path)
            except OSError:
                pass


def _max_bytes() -> int:
    raw = os.getenv("OMNI_MCP_MAX_DOWNLOAD_BYTES")
    if raw:
        try:
            return int(raw)
        except ValueError:
            pass
    return DEFAULT_MAX_BYTES


def _remote_enabled() -> bool:
    """Remote fetching can be disabled entirely for locked-down deployments."""
    return os.getenv("OMNI_MCP_DISABLE_REMOTE", "").strip().lower() not in ("1", "true", "yes")


def _tempfile_for(suffix: str) -> str:
    fd, path = tempfile.mkstemp(prefix="omni_mcp_", suffix=suffix)
    os.close(fd)
    return path


def _suffix_from(name: str, default: str = "") -> str:
    suffix = Path(urlparse(name).path).suffix
    return suffix or default


# --- Google Drive -----------------------------------------------------------

_GDRIVE_FILE_RE = re.compile(r"/file/d/([A-Za-z0-9_-]+)")


def _gdrive_direct_url(url: str) -> Optional[str]:
    """Convert a Google Drive share link into a direct-download URL."""
    parsed = urlparse(url)
    if "drive.google.com" not in parsed.netloc and "docs.google.com" not in parsed.netloc:
        return None

    match = _GDRIVE_FILE_RE.search(parsed.path)
    file_id = match.group(1) if match else None
    if not file_id:
        file_id = (parse_qs(parsed.query).get("id") or [None])[0]
    if not file_id:
        return None
    return f"https://drive.google.com/uc?export=download&id={file_id}"


# --- Backends ---------------------------------------------------------------

def _fetch_http(url: str) -> ResolvedMedia:
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - requests is a base dependency
        raise MediaError("HTTP downloads require the 'requests' package.") from exc

    direct = _gdrive_direct_url(url)
    target = direct or url

    limit = _max_bytes()
    try:
        response = requests.get(target, stream=True, timeout=DEFAULT_TIMEOUT, allow_redirects=True)
        response.raise_for_status()
    except Exception as exc:
        raise MediaError(f"Failed to download {url}: {exc}") from exc

    suffix = _suffix_from(target)
    if not suffix:
        content_type = (response.headers.get("content-type") or "").split(";")[0].strip()
        suffix = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/gif": ".gif",
            "image/webp": ".webp",
            "audio/mpeg": ".mp3",
            "audio/wav": ".wav",
            "audio/x-wav": ".wav",
            "video/mp4": ".mp4",
            "application/pdf": ".pdf",
            "text/plain": ".txt",
        }.get(content_type, "")

    path = _tempfile_for(suffix)
    written = 0
    try:
        with open(path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 256):
                if not chunk:
                    continue
                written += len(chunk)
                if written > limit:
                    raise MediaError(
                        f"Remote object exceeds the {limit} byte limit "
                        "(raise OMNI_MCP_MAX_DOWNLOAD_BYTES to allow larger files)."
                    )
                handle.write(chunk)
    except Exception:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise

    return ResolvedMedia(path=path, source=url, is_temporary=True)


def _fetch_s3(url: str) -> ResolvedMedia:
    """Fetch s3://bucket/key. Works with MinIO and other S3-compatible stores.

    Endpoint/credentials come from the standard environment:
      AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION
      S3_ENDPOINT_URL (or AWS_ENDPOINT_URL_S3) -> point this at MinIO
    """
    try:
        import boto3
    except ImportError as exc:
        raise MediaError(
            "s3:// references require boto3. Install it with: pip install boto3"
        ) from exc

    parsed = urlparse(url)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if not bucket or not key:
        raise MediaError(f"Malformed S3 URI: {url} (expected s3://bucket/key)")

    endpoint = os.getenv("S3_ENDPOINT_URL") or os.getenv("AWS_ENDPOINT_URL_S3")
    client_kwargs = {}
    if endpoint:
        client_kwargs["endpoint_url"] = endpoint

    path = _tempfile_for(_suffix_from(key))
    try:
        client = boto3.client("s3", **client_kwargs)
        client.download_file(bucket, key, path)
    except Exception as exc:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise MediaError(f"Failed to download {url}: {exc}") from exc

    return ResolvedMedia(path=path, source=url, is_temporary=True)


def _fetch_gcs(url: str) -> ResolvedMedia:
    try:
        from google.cloud import storage  # type: ignore
    except ImportError as exc:
        raise MediaError(
            "gs:// references require google-cloud-storage. "
            "Install it with: pip install google-cloud-storage"
        ) from exc

    parsed = urlparse(url)
    bucket_name = parsed.netloc
    blob_name = parsed.path.lstrip("/")
    if not bucket_name or not blob_name:
        raise MediaError(f"Malformed GCS URI: {url} (expected gs://bucket/object)")

    path = _tempfile_for(_suffix_from(blob_name))
    try:
        client = storage.Client()
        client.bucket(bucket_name).blob(blob_name).download_to_filename(path)
    except Exception as exc:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise MediaError(f"Failed to download {url}: {exc}") from exc

    return ResolvedMedia(path=path, source=url, is_temporary=True)


# --- Public API -------------------------------------------------------------

def resolve_media(reference: str) -> ResolvedMedia:
    """Resolve a media reference to a local file.

    The caller owns the result and should call ``cleanup()`` when done; only
    downloaded files are actually removed, local inputs are left untouched.
    """
    if not reference or not str(reference).strip():
        raise MediaError("Empty media reference.")

    reference = str(reference).strip()
    parsed = urlparse(reference)
    scheme = parsed.scheme.lower()

    # Local file (plain path, Windows drive letter, or file:// URI)
    if scheme in ("", "file") or (len(scheme) == 1 and reference[1:2] == ":"):
        if scheme == "file":
            local = parsed.path
        else:
            local = reference
        local = os.path.abspath(os.path.expanduser(local))
        if not os.path.exists(local):
            raise MediaError(f"Local file not found: {local}")
        if not os.path.isfile(local):
            raise MediaError(f"Not a regular file: {local}")
        return ResolvedMedia(path=local, source=reference, is_temporary=False)

    if not _remote_enabled():
        raise MediaError(
            f"Remote media fetching is disabled (OMNI_MCP_DISABLE_REMOTE), cannot resolve {reference}"
        )

    if scheme in ("http", "https"):
        return _fetch_http(reference)
    if scheme == "s3":
        return _fetch_s3(reference)
    if scheme == "gs":
        return _fetch_gcs(reference)

    raise MediaError(
        f"Unsupported media scheme '{scheme}://'. "
        "Supported: local paths, file://, http(s)://, s3:// (S3/MinIO), gs:// (GCS)."
    )


# --- Document text extraction ----------------------------------------------

def extract_document_text(path: str) -> str:
    """Extract plain text from a document so it can be stored as text memory.

    Supports .txt/.md/.json/.csv natively, .pdf via pypdf, .docx via python-docx.
    """
    suffix = Path(path).suffix.lower()

    if suffix in (".txt", ".md", ".json", ".csv", ".log", ".rst", ".yaml", ".yml", ""):
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read()

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader  # type: ignore
        except ImportError:
            try:
                from PyPDF2 import PdfReader  # type: ignore
            except ImportError as exc:
                raise MediaError(
                    "Reading PDF documents requires pypdf. Install it with: pip install pypdf"
                ) from exc
        reader = PdfReader(path)
        return "\n\n".join((page.extract_text() or "") for page in reader.pages).strip()

    if suffix == ".docx":
        try:
            import docx  # type: ignore
        except ImportError as exc:
            raise MediaError(
                "Reading .docx documents requires python-docx. "
                "Install it with: pip install python-docx"
            ) from exc
        document = docx.Document(path)
        return "\n".join(paragraph.text for paragraph in document.paragraphs).strip()

    raise MediaError(
        f"Unsupported document type '{suffix}'. "
        "Supported: .txt, .md, .json, .csv, .yaml, .rst, .log, .pdf, .docx"
    )


# --- media kind validation --------------------------------------------------

# Extensions we can confidently attribute to a modality. Anything not listed is
# treated as "unknown" and passed through to the processor, which may support
# formats this table does not enumerate.
_KIND_EXTENSIONS = {
    "image": {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff", ".tif", ".heic", ".heif"},
    "audio": {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".oga", ".opus", ".aac", ".wma", ".aiff"},
    "video": {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv", ".m4v", ".mpeg", ".mpg"},
    "document": {".txt", ".md", ".json", ".csv", ".log", ".rst", ".yaml", ".yml", ".pdf", ".docx"},
}


def guess_kind(path: str) -> Optional[str]:
    """Return 'image' / 'audio' / 'video' / 'document', or None if unrecognized."""
    suffix = Path(path).suffix.lower()
    if not suffix:
        return None
    for kind, extensions in _KIND_EXTENSIONS.items():
        if suffix in extensions:
            return kind
    return None


def ensure_kind(path: str, expected: str, reference: str) -> None:
    """Reject a file whose extension clearly belongs to a different modality.

    Unknown extensions are allowed through so the processors can still handle
    formats this table does not list; only a confident mismatch is an error.
    """
    actual = guess_kind(path)
    if actual is not None and actual != expected:
        raise MediaError(
            f"{reference} looks like a {actual} file ({Path(path).suffix}), "
            f"but it was passed to the {expected} tool. "
            f"Use the omni_add_{'document' if actual == 'document' else actual} tool instead."
        )


def copy_to(path: str, destination_dir: str) -> str:
    """Copy a resolved file into a persistent directory, returning the new path."""
    os.makedirs(destination_dir, exist_ok=True)
    target = os.path.join(destination_dir, os.path.basename(path))
    shutil.copy2(path, target)
    return target
