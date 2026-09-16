# MCP server: stdio and HTTP

[Package overview](../README.md) · [HTTP and deployment](http-and-deployment.md) · [Agent tools](agent-tools.md)

The engine speaks [Model Context Protocol](https://modelcontextprotocol.io), so
an MCP client -- Claude Code, Claude Desktop, or anything else that speaks it --
can store and recall memories, read the entity graph and submit distilled facts.
The tools, their argument names and their bounds match the Rust server in
`crates/scone/src/mcp.rs`, so a client configured against one can point at the
other unchanged.

Two transports serve the same tools:

| | `--transport stdio` (default) | `--transport http` |
|---|---|---|
| Who may connect | the process that started this one | anyone who reaches the address and holds a key |
| Key | none: the pipe is the boundary | required, unless `--allow-anonymous` on loopback |
| Space | any space a call names | the key's space, and no other |
| Role | full | the key's role; a role that does not write reaches no writing tool |

## stdio

```bash
python -m scone_memory.runtime.mcp --space default
```

Stores come from `SCONE_*` variables exactly as for the CLI; with nothing set,
memory persists to SQLite at `~/.scone-memory/memory.db`.

## HTTP

```bash
python -m scone_memory.runtime.mcp --transport http --space notes
```

JSON-RPC arrives in a `POST /mcp` and answers leave as a server-sent event
stream, which is what lets a long tool call report progress before it finishes.
The default address is `127.0.0.1:8765`; `--host` and `--port` change it, and
they are refused for stdio, where there is nothing to bind.

### The key

With `SCONE_API_KEYS` (or `SCONE_API_KEY`) set, the HTTP transport reads the
same table the REST API reads, in the same format:

```bash
export SCONE_API_KEYS="k-ana:ana,k-bot:ana:read"
```

With nothing set, the first HTTP start issues a key of its own, so self-hosting
does not begin with inventing a secret:

```
scone: no SCONE_API_KEYS set, so this server issued a key for space 'notes':
  key:  sk-scone-...
  file: ~/.scone-memory/mcp-key (yours alone; delete it to issue a new one)
  url:  http://127.0.0.1:8765/mcp
  in an MCP client: "headers": {"Authorization": "Bearer sk-scone-..."}
```

The key is saved beside the memory it opens (next to `SCONE_SQLITE_PATH`), with
permissions only its owner can read, and read back on later starts -- so a
client configured once keeps working, and moving the store moves the key with
it. Delete the file to issue a new one. A host that sets `SCONE_API_KEYS` is
never given one, and stdio never is: there the client is the process that
started this one, and has nothing to prove.

`--allow-anonymous` serves with no key at all, as stdio does. Like any keyless
server it may bind nothing but a loopback address; asked for a wider one it
refuses to start rather than answer whoever arrives.

### What a key bounds

A key names a space and carries a role, and both are enforced at the tool:

- a call that names another space is refused with `space 'X' is not this key's`,
  the same words the REST API uses, whether the space is named in a tool
  argument or in a resource address such as `scone://X/graph/schema`;
- the fixed addresses (`scone://graph/schema` and its siblings) mean the key's
  own space;
- a `read` or `review` role calling `memory_store`, `memory_store_facts` or
  `memory_forget` is told `key role read cannot write`. The tools stay in the
  catalogue: a model that can read the reason corrects itself, where a tool that
  had silently vanished would be guessed at.

A server is built per key holder rather than per request, and sessions belong to
the holder that opened them: a session id presented by another key reaches no
session at all. Two keys that name the same space with the same role share one
-- they are the same authority. A key added to the table while the server runs
is served if its space and role are already covered, and otherwise told to
restart the server.

### Pointing a client at it

```json
{
  "mcpServers": {
    "scone-memory": {
      "type": "http",
      "url": "http://127.0.0.1:8765/mcp",
      "headers": { "Authorization": "Bearer sk-scone-..." }
    }
  }
}
```

Behind a reverse proxy, forward `/mcp` without buffering responses: a buffered
proxy holds back the event stream, and a tool that reports progress before it
answers appears to hang. Terminate TLS there; this server speaks plain HTTP and
binds loopback unless told otherwise.
