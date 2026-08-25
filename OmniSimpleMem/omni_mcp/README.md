# Omni-SimpleMem MCP Server

Exposes the Omni-SimpleMem multimodal memory pipeline (text, image, audio, video,
documents) to any MCP client over the **stdio transport**, with **isolated
per-agent memory namespaces**.

This complements the existing text-oriented MCP server under `MCP/`: that one is
a multi-tenant HTTP service, this one is a local stdio server built directly on
`OmniMemoryOrchestrator`, so it can reach the multimodal processors and the local
filesystem (needed for image/audio/video/document ingestion).

## Install

```bash
cd OmniSimpleMem
pip install -r requirements.txt          # core Omni-Memory dependencies

# Optional, only if you use the matching feature:
pip install boto3                        # s3:// (AWS S3 or MinIO)
pip install google-cloud-storage         # gs:// (Google Cloud Storage)
pip install pypdf                        # .pdf documents
pip install python-docx                  # .docx documents
```

## Run

```bash
python -m omni_mcp --data-dir ~/.omni_simplemem/mcp
```

The server speaks newline-delimited JSON-RPC 2.0 on stdin/stdout. It is normally
launched by an MCP client rather than by hand.

### Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "omni-simplemem": {
      "command": "python",
      "args": ["-m", "omni_mcp", "--data-dir", "~/.omni_simplemem/mcp"],
      "cwd": "/absolute/path/to/SimpleMem/OmniSimpleMem",
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "PYTHONPATH": "/absolute/path/to/SimpleMem/OmniSimpleMem"
      }
    }
  }
}
```

## Tools

| Tool | Purpose |
|------|---------|
| `omni_add_text` | Store text as an atomic memory unit |
| `omni_add_image` | Store an image (captioned + embedded, entropy-triggered) |
| `omni_add_audio` | Store audio (transcribed, VAD-triggered) |
| `omni_add_video` | Store a video (only visually significant frames) |
| `omni_add_document` | Extract text from `.txt/.md/.json/.csv/.yaml/.pdf/.docx` and store it |
| `omni_query` | Retrieve relevant memory summaries |
| `omni_answer` | Retrieval-augmented answer over memory |
| `omni_stats` | Memory statistics for a namespace |
| `omni_list_events` | List event nodes (grouped memories) |
| `omni_consolidate` | Run importance-based consolidation |
| `omni_list_namespaces` | List all isolated memory clusters |
| `omni_delete_namespace` | Permanently delete a namespace (needs `confirm: true`) |

### Isolated memory clusters

Every tool takes an optional `namespace`. Each namespace is a **fully separate
memory cluster** — its own storage directory, MAU store, vector index and event
store — so multiple agents can share one server without seeing each other's
memories.

```jsonc
{"name": "omni_add_text",
 "arguments": {"text": "Design review moved to Friday.", "namespace": "agent_planner"}}
```

Namespace names are restricted to `[A-Za-z0-9][A-Za-z0-9._-]{0,63}`; anything
that could traverse the filesystem (`../`, `/`, absolute paths) is rejected.

Orchestrators are loaded lazily and kept in an LRU cache
(`--max-open-namespaces`, default 8); evicted namespaces are saved to disk and
transparently reloaded on next use.

### Media references

Any media argument accepts:

| Form | Example |
|------|---------|
| Local path | `/data/photo.png` |
| File URI | `file:///data/photo.png` |
| HTTP(S) | `https://example.com/clip.mp4` |
| Google Drive share link | `https://drive.google.com/file/d/<id>/view` |
| S3 / MinIO | `s3://bucket/key` |
| Google Cloud Storage | `gs://bucket/object` |

For **MinIO** (or any S3-compatible store) set the endpoint and credentials in
the server's environment:

```bash
export S3_ENDPOINT_URL=https://minio.internal:9000
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
```

Remote objects are downloaded to a temp file, ingested, then deleted. Local
files are never modified or removed.

## Configuration

| Variable | Default | Meaning |
|----------|---------|---------|
| `OMNI_MCP_DATA_DIR` | `~/.omni_simplemem/mcp` | Base dir holding namespaces |
| `OMNI_MCP_MAX_OPEN_NAMESPACES` | `8` | Orchestrators kept loaded |
| `OMNI_MCP_MAX_DOWNLOAD_BYTES` | `536870912` (512 MB) | Remote download cap |
| `OMNI_MCP_DISABLE_REMOTE` | unset | Set to `1` to forbid all remote fetching |
| `OMNI_MCP_LOG_LEVEL` | `INFO` | Log level (logs go to stderr) |
| `OPENAI_API_KEY` | — | Required for captioning, transcription, query/answer |
| `OPENAI_API_BASE` | — | Optional OpenAI-compatible gateway |

### What works without an API key

`omni_add_text`, `omni_add_document`, `omni_stats`, `omni_list_events`,
`omni_list_namespaces`, `omni_consolidate` and `omni_delete_namespace` run fully
offline. Image/audio/video captioning and `omni_query`/`omni_answer` call an LLM
and require a key; without one they return a clear, actionable error rather than
failing silently.

## Tests

```bash
cd OmniSimpleMem
python -m pytest tests/test_omni_mcp.py -q
```

The suite runs offline and covers the protocol layer, namespace isolation and
path-traversal rejection, media resolution (including a real local HTTP
download), document extraction, persistence across restarts, and error handling.

## Notes and limits

- The stdio transport owns stdout, so the server routes all library output and
  logging to stderr. Do not add `print()` calls that write to stdout.
- Video ingestion samples frames; it does not perform object segmentation or
  tracking. Cross-image object/person graph search, art-style classification and
  segmentation-based retrieval are **not** implemented.
- Retrieval quality depends on the configured embedding model; mixing embedding
  models across runs in one namespace will degrade recall.
