# Web research with long-term memory

Use SimpleMem and [Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp)
together in Cursor to research current information and recall selected findings in
later conversations. Cursor coordinates two independent MCP servers: Parallel
searches the web and fetches pages, while SimpleMem stores and retrieves memories.
Adding the search server does not make SimpleMem's memory queries browse the web
automatically.

## Set up Cursor

1. Complete the [SimpleMem MCP setup](../MCP/README.md) and obtain your SimpleMem
   token. You can use the hosted service or your existing self-hosted endpoint.
2. Open `~/.cursor/mcp.json`, Cursor's global MCP configuration. Merge the
   `parallel-search` entry below into your existing `mcpServers` object. Keep your
   existing SimpleMem URL, token, other servers, and enabled/disabled choices.
   If you are starting from scratch, the full example is:

   ```json
   {
     "mcpServers": {
       "simplemem": {
         "url": "https://mcp.simplemem.cloud/mcp",
         "headers": {
           "Authorization": "Bearer YOUR_TOKEN"
         }
       },
       "parallel-search": {
         "url": "https://search.parallel.ai/mcp"
       }
     }
   }
   ```

3. Replace `YOUR_TOKEN` with your SimpleMem token. Keep the file private. Parallel's
   anonymous endpoint needs no account or API key; its free tier is intended for
   exploration and light use, with search running in Fast mode. SimpleMem still
   needs its own authentication and backend configuration.
4. In Cursor's MCP settings, confirm both servers connect and their tools appear.
   Parallel exposes `web_search` and `web_fetch`; SimpleMem exposes `memory_add`,
   `memory_retrieve`, and `memory_query`, among other tools. See the
   [Cursor MCP documentation](https://cursor.com/docs/context/mcp) for client setup
   and tool approval controls.

## Research, then save selected findings

Start an Agent conversation with both servers enabled. For example:

> I'm documenting Python virtual environments for our team. First use SimpleMem's
> memory_retrieve to check for earlier team decisions about Python environments.
> Then use Parallel's web_search to find the current official Python venv
> documentation. Use web_fetch if the search excerpts are insufficient to explain
> how to create and activate an environment on Linux and Windows. Show the source
> URLs, retrieval date, and any differences from our saved decisions. Do not save
> anything yet.

Review the sources and choose what should persist. Then ask:

> Save only the findings I selected using SimpleMem's memory_add, with speaker
> "Research notes". Include the topic, a short self-contained finding, its source
> title and full URL, and today's absolute retrieval date in the content. Treat
> these as external research findings, not as my preferences or team decisions.

Keep notes focused so later queries can find them. Put provenance in the `content`
alongside each finding; `memory_add` has no separate source-URL field. SimpleMem
compresses input into facts, so it is not a verbatim source archive. After saving,
ask `memory_retrieve` for the topic and check that the finding, URL, and retrieval
date survived. If provenance is missing, correct the note and verify it again;
keep the original source separately when exact wording matters.

## Recall and refresh

In a later conversation using the same SimpleMem account, try:

> Use SimpleMem's memory_retrieve to recall our Python virtual-environment
> research. Show the stored source URLs and retrieval dates. Use Parallel to check
> the current documentation before making a recommendation. Explain any changes
> and ask me before saving updated findings.

`memory_retrieve` returns stored entries for inspection; `memory_query` generates
an answer from memory. Neither establishes that a saved fact is still current.
Use web research to recheck time-sensitive findings. If no memory is returned,
broaden the topic query and confirm you are using the same SimpleMem account.

If a server fails to connect, check its own configuration: the bearer token above
belongs to SimpleMem only. If Parallel returns a free-tier rate limit, follow the
retry guidance in the response or try again later; do not treat the failed request
as evidence for a finding.
