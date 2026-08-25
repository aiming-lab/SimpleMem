"""
Tests for the Omni-SimpleMem MCP server.

These run fully offline. Operations that need an LLM API key (image/audio/video
captioning, query/answer) are covered only for their error handling, so the
suite stays green without credentials.
"""

import json
import os
import shutil
import sys
import tempfile
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omni_mcp.media import (  # noqa: E402
    MediaError,
    _gdrive_direct_url,
    extract_document_text,
    resolve_media,
)
from omni_mcp.namespaces import (  # noqa: E402
    NamespaceError,
    NamespaceManager,
    validate_namespace,
)
from omni_mcp.server import OmniMCPServer  # noqa: E402
from omni_mcp.tools import tool_definitions  # noqa: E402


# --- fixtures ---------------------------------------------------------------

@pytest.fixture
def workdir():
    path = tempfile.mkdtemp(prefix="omni_mcp_tests_")
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def server(workdir):
    instance = OmniMCPServer(base_dir=os.path.join(workdir, "memory"))
    yield instance
    instance.shutdown()


def call_tool(server, name, arguments=None):
    """Invoke tools/call and return (payload_text, is_error)."""
    response = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments or {}},
        }
    )
    result = response["result"]
    return result["content"][0]["text"], result["isError"]


def call_tool_json(server, name, arguments=None):
    text, is_error = call_tool(server, name, arguments)
    assert not is_error, f"{name} failed: {text}"
    return json.loads(text)


# --- protocol ---------------------------------------------------------------

def test_initialize_reports_server_info(server):
    response = server.handle_request(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}}
    )
    result = response["result"]
    assert result["serverInfo"]["name"] == "omni-simplemem"
    assert result["protocolVersion"] == "2024-11-05"
    assert "tools" in result["capabilities"]


def test_initialize_echoes_client_protocol_version(server):
    response = server.handle_request(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}}
    )
    assert response["result"]["protocolVersion"] == "2025-06-18"


def test_notifications_get_no_response(server):
    assert server.handle_request({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def test_unknown_method_returns_method_not_found(server):
    response = server.handle_request({"jsonrpc": "2.0", "id": 9, "method": "does/not/exist"})
    assert response["error"]["code"] == -32601


def test_tools_list_exposes_expected_tools(server):
    response = server.handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    names = {tool["name"] for tool in response["result"]["tools"]}
    assert {
        "omni_add_text",
        "omni_add_image",
        "omni_add_audio",
        "omni_add_video",
        "omni_add_document",
        "omni_query",
        "omni_answer",
        "omni_stats",
        "omni_list_namespaces",
        "omni_delete_namespace",
    } <= names


def test_every_tool_has_a_valid_schema():
    for tool in tool_definitions():
        assert tool["name"] and tool["description"]
        schema = tool["inputSchema"]
        assert schema["type"] == "object"
        for required in schema.get("required", []):
            assert required in schema["properties"], f"{tool['name']}: {required} not in properties"


def test_unknown_tool_is_reported_as_tool_error(server):
    text, is_error = call_tool(server, "nope")
    assert is_error and "Unknown tool" in text


# --- namespaces -------------------------------------------------------------

@pytest.mark.parametrize(
    "name", ["../escape", "a/b", "..", "x" * 65, "/abs", "a\\b", "a\x00b", ".", "-lead"]
)
def test_path_traversal_namespaces_raise(name):
    with pytest.raises(NamespaceError):
        validate_namespace(name)


@pytest.mark.parametrize("name", ["default", "agent_a", "Agent-1", "a.b_c-1"])
def test_valid_namespaces_accepted(name):
    assert validate_namespace(name) == name


def test_empty_namespace_defaults(workdir):
    assert validate_namespace(None) == "default"
    assert validate_namespace("") == "default"


def test_namespace_data_dirs_are_isolated(workdir):
    manager = NamespaceManager(os.path.join(workdir, "mem"))
    assert manager.data_dir_for("agent_a") != manager.data_dir_for("agent_b")
    with pytest.raises(NamespaceError):
        manager.data_dir_for("../outside")


def test_memories_do_not_leak_across_namespaces(server):
    call_tool_json(server, "omni_add_text", {"text": "Alice is in Paris.", "namespace": "agent_a"})
    call_tool_json(server, "omni_add_text", {"text": "Bob is in Tokyo.", "namespace": "agent_b"})
    call_tool_json(server, "omni_add_text", {"text": "Bob is also in Osaka.", "namespace": "agent_b"})

    stats_a = call_tool_json(server, "omni_stats", {"namespace": "agent_a"})
    stats_b = call_tool_json(server, "omni_stats", {"namespace": "agent_b"})

    assert stats_a["mau_count"] == 1
    assert stats_b["mau_count"] == 2
    assert stats_a["storage_stats"]["storage_path"] != stats_b["storage_stats"]["storage_path"]


def test_list_namespaces_reports_created_clusters(server):
    call_tool_json(server, "omni_add_text", {"text": "hello", "namespace": "agent_a"})
    listing = call_tool_json(server, "omni_list_namespaces")
    assert "agent_a" in {n["namespace"] for n in listing["namespaces"]}


def test_delete_namespace_requires_confirmation(server):
    call_tool_json(server, "omni_add_text", {"text": "temporary", "namespace": "scratch"})

    refused = call_tool_json(server, "omni_delete_namespace", {"namespace": "scratch", "confirm": False})
    assert refused["deleted"] is False

    deleted = call_tool_json(server, "omni_delete_namespace", {"namespace": "scratch", "confirm": True})
    assert deleted["deleted"] is True

    listing = call_tool_json(server, "omni_list_namespaces")
    assert "scratch" not in {n["namespace"] for n in listing["namespaces"]}


def test_memory_persists_across_server_restarts(workdir):
    base = os.path.join(workdir, "persist")

    first = OmniMCPServer(base_dir=base)
    call_tool_json(first, "omni_add_text", {"text": "Durable fact.", "namespace": "agent_a"})
    first.shutdown()

    second = OmniMCPServer(base_dir=base)
    try:
        stats = call_tool_json(second, "omni_stats", {"namespace": "agent_a"})
        assert stats["mau_count"] == 1
    finally:
        second.shutdown()


# --- ingestion --------------------------------------------------------------

def test_add_text_stores_a_memory_unit(server):
    payload = call_tool_json(server, "omni_add_text", {"text": "The sky is blue.", "namespace": "t"})
    assert payload["success"] is True
    assert payload["mau_id"]
    assert payload["modality"] == "text"


def test_add_text_rejects_empty_input(server):
    text, is_error = call_tool(server, "omni_add_text", {"text": "   "})
    assert is_error and "required" in text


def test_add_document_extracts_and_stores_text(server, workdir):
    doc = os.path.join(workdir, "note.md")
    with open(doc, "w", encoding="utf-8") as handle:
        handle.write("# Title\n\nThe Eiffel Tower is in Paris.\n")

    payload = call_tool_json(server, "omni_add_document", {"document": doc, "namespace": "docs"})
    assert payload["success"] is True
    assert payload["truncated"] is False
    assert payload["characters_stored"] > 0


def test_add_document_truncates_to_max_chars(server, workdir):
    doc = os.path.join(workdir, "big.txt")
    with open(doc, "w", encoding="utf-8") as handle:
        handle.write("word " * 5000)

    payload = call_tool_json(server, "omni_add_document", {"document": doc, "max_chars": 100})
    assert payload["characters_stored"] == 100
    assert payload["truncated"] is True


def test_missing_media_file_is_a_clean_error(server):
    text, is_error = call_tool(server, "omni_add_image", {"image": "/nope/missing.png"})
    assert is_error and "not found" in text.lower()


@pytest.mark.parametrize(
    "tool,argument",
    [("omni_add_video", "video"), ("omni_add_image", "image"), ("omni_add_audio", "audio")],
)
def test_media_tools_reject_mismatched_file_types(server, workdir, tool, argument):
    """A .txt must not be ingestible as video/image/audio (it would store junk)."""
    doc = os.path.join(workdir, "note.txt")
    with open(doc, "w", encoding="utf-8") as handle:
        handle.write("this is prose, not media")

    text, is_error = call_tool(server, tool, {argument: doc, "namespace": "mismatch"})
    assert is_error
    assert "omni_add_document" in text

    stats = call_tool_json(server, "omni_stats", {"namespace": "mismatch"})
    assert stats["mau_count"] == 0, "a rejected file must not create a memory unit"


def test_unknown_extensions_are_passed_through_to_the_processor(workdir):
    """Only a confident mismatch is rejected; unknown suffixes stay allowed."""
    from omni_mcp.media import ensure_kind, guess_kind

    assert guess_kind("/tmp/clip.mp4") == "video"
    assert guess_kind("/tmp/photo.JPG") == "image"
    assert guess_kind("/tmp/mystery.raw") is None
    ensure_kind("/tmp/mystery.raw", "video", "mystery.raw")  # must not raise


def test_llm_dependent_tools_explain_missing_api_key(server, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    call_tool_json(server, "omni_add_text", {"text": "seed", "namespace": "q"})

    text, is_error = call_tool(server, "omni_query", {"query": "anything?", "namespace": "q"})
    if is_error:
        assert "API key" in text


# --- media resolution -------------------------------------------------------

def test_resolve_local_file(workdir):
    path = os.path.join(workdir, "a.txt")
    with open(path, "w") as handle:
        handle.write("hi")

    resolved = resolve_media(path)
    assert resolved.path == os.path.abspath(path)
    assert resolved.is_temporary is False
    resolved.cleanup()
    assert os.path.exists(path), "cleanup must not delete local inputs"


def test_resolve_file_uri(workdir):
    path = os.path.join(workdir, "b.txt")
    with open(path, "w") as handle:
        handle.write("hi")
    assert resolve_media(f"file://{path}").path == os.path.abspath(path)


def test_resolve_missing_local_file_raises():
    with pytest.raises(MediaError):
        resolve_media("/definitely/not/here.png")


def test_resolve_rejects_unsupported_scheme():
    with pytest.raises(MediaError) as excinfo:
        resolve_media("ftp://host/file.png")
    assert "Unsupported media scheme" in str(excinfo.value)


def test_resolve_rejects_empty_reference():
    with pytest.raises(MediaError):
        resolve_media("")


def test_google_drive_share_links_convert_to_direct_download():
    direct = _gdrive_direct_url("https://drive.google.com/file/d/1AbC_dEF-123/view?usp=sharing")
    assert direct == "https://drive.google.com/uc?export=download&id=1AbC_dEF-123"

    by_id = _gdrive_direct_url("https://drive.google.com/open?id=XYZ789")
    assert by_id == "https://drive.google.com/uc?export=download&id=XYZ789"

    assert _gdrive_direct_url("https://example.com/img.png") is None


def test_http_download_fetches_remote_file(workdir):
    content = b"remote-bytes"
    with open(os.path.join(workdir, "remote.txt"), "wb") as handle:
        handle.write(content)

    handler = partial(SimpleHTTPRequestHandler, directory=workdir)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        resolved = resolve_media(f"http://127.0.0.1:{port}/remote.txt")
        assert resolved.is_temporary is True
        with open(resolved.path, "rb") as handle:
            assert handle.read() == content
        resolved.cleanup()
        assert not os.path.exists(resolved.path)
    finally:
        httpd.shutdown()


def test_http_download_respects_size_limit(workdir, monkeypatch):
    with open(os.path.join(workdir, "big.bin"), "wb") as handle:
        handle.write(b"x" * 4096)

    monkeypatch.setenv("OMNI_MCP_MAX_DOWNLOAD_BYTES", "100")

    handler = partial(SimpleHTTPRequestHandler, directory=workdir)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(MediaError) as excinfo:
            resolve_media(f"http://127.0.0.1:{port}/big.bin")
        assert "limit" in str(excinfo.value)
    finally:
        httpd.shutdown()


def test_remote_fetching_can_be_disabled(monkeypatch):
    monkeypatch.setenv("OMNI_MCP_DISABLE_REMOTE", "1")
    with pytest.raises(MediaError) as excinfo:
        resolve_media("https://example.com/a.png")
    assert "disabled" in str(excinfo.value)


def test_malformed_s3_uri_is_rejected():
    pytest.importorskip("boto3")
    with pytest.raises(MediaError) as excinfo:
        resolve_media("s3://bucket-only")
    assert "Malformed S3 URI" in str(excinfo.value)


# --- document extraction ----------------------------------------------------

def test_extract_text_from_plain_documents(workdir):
    path = os.path.join(workdir, "note.txt")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("plain text body")
    assert extract_document_text(path) == "plain text body"


def test_extract_rejects_unsupported_document_type(workdir):
    path = os.path.join(workdir, "thing.xyz")
    with open(path, "w") as handle:
        handle.write("data")
    with pytest.raises(MediaError) as excinfo:
        extract_document_text(path)
    assert "Unsupported document type" in str(excinfo.value)


# --- resources --------------------------------------------------------------

def test_resources_list_and_read(server):
    listing = server.handle_request({"jsonrpc": "2.0", "id": 3, "method": "resources/list"})
    uris = {r["uri"] for r in listing["result"]["resources"]}
    assert "omni://namespaces" in uris

    read = server.handle_request(
        {"jsonrpc": "2.0", "id": 4, "method": "resources/read", "params": {"uri": "omni://namespaces"}}
    )
    assert json.loads(read["result"]["contents"][0]["text"])["count"] >= 0


def test_reading_unknown_resource_errors(server):
    response = server.handle_request(
        {"jsonrpc": "2.0", "id": 5, "method": "resources/read", "params": {"uri": "omni://nope"}}
    )
    assert response["error"]["code"] == -32603
