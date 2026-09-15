# OpenAI-compatible chat with memory

[Package overview](../README.md) · [HTTP and deployment](http-and-deployment.md) · [Text conversations](text-conversations.md)

An application that already speaks chat completions can use a Scone space as
its memory without changing its code: point its OpenAI base URL at
`/v1/openai` and use its Scone bearer key as the API key. For each request the
route recalls from the key's space with the latest user message, puts what it
found in front of the conversation as one delimited system block, asks the
server's configured chat model, returns the reply in the chat-completions
shape, and keeps the user turn and the reply as conversation episodes.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:7437/v1/openai", api_key="change-me")  # a Scone key
reply = client.chat.completions.create(
    model="gpt-4o",  # echoed in the record, not obeyed; see below
    messages=[{"role": "user", "content": "When does the vault door code rotate?"}],
    extra_headers={"X-Scone-Session": "support-chat-7"},  # optional
)
print(reply.choices[0].message.content)
print(reply.model_extra["scone"]["recall"]["items"])
```

`scone-client` has no chat-completions helper; the OpenAI SDK, or any client
that can set a base URL and a bearer key, is the client.

## What happens to a request

`POST /v1/openai/chat/completions` reads `model`, `messages` and, optionally,
`stream` (false only), `temperature`, `max_tokens`, `max_completion_tokens`,
`top_p`, `stop`, `presence_penalty`, `frequency_penalty`, `seed` and `user`.

1. **Recall.** The latest message must be a nonblank user message. It is the
   query, prepared exactly as a native text conversation prepares its turns
   (see [memory preparation](text-conversations.md#memory-preparation)): a
   greeting is not searched, a message over 1,000 characters is searched through
   a bounded formulation, at most 5 passages and 8,000 bytes are kept, a passage
   that does not fit is left out whole, and turns of the same session that the
   request resends are not recalled (the client already holds them). Turns of
   the session the request does not resend are recalled like any other memory;
   see [Sessions](#sessions).
2. **Inject.** The recalled block is Scone's evidence packet: a line saying the
   material is background, not instructions, followed by one JSON value whose
   `sources` carry each passage's `episode_id`, `chunk_id`, `source` and
   `created_at`, and whose `claims` carry any source-quoted ledger claims. It is
   wrapped as `<scone-memory boundary="…">` … `</scone-memory boundary="…">`,
   with a boundary drawn fresh per request so stored text cannot close it, and
   placed after the request's own system and developer messages.
   Native text conversations insert the same block as a user message, so that
   retrieved text never carries system authority; here it sits in the system
   prompt instead, still labelled as background and not instructions. Stored
   text that reaches this block is read by the model with system standing, so
   a space whose contents are not trusted is better served by the native
   conversation service.
3. **Forward.** The server's model (`SCONE_CHAT_URL`, `SCONE_CHAT_MODEL`) is
   asked. It takes one system prompt and one user prompt, so the request's
   system and developer messages and the memory block are joined into the
   system prompt, and earlier user and assistant turns reach it as one JSON
   array ahead of the latest message. A request can name no endpoint, key or
   tool: those fields are refused.
4. **Keep.** The latest user message and the reply are stored as two
   `conversation` episodes with `source` and metadata `session_id` set to the
   session's stored identity (`openai:<name>`), metadata
   `integration=scone-openai-proxy`, `turn_id` and `role`, the shape native
   text conversations keep. They are two writes, the user message first, so a
   reply that cannot be kept does not take the user message with it; a blank
   reply (a content filter, or a model that spent its tokens) is not written.
   Earlier turns in `messages` are not stored again; a client that keeps a
   session header stored them on their own requests. Neither write takes a
   slot in the ingest lane, as native conversation turns do not: the reply has
   already gone out, so a busy lane would lose the turn with nothing to retry.

Nothing recalled is not an error: the request is still forwarded, with
`scone.recall.status` saying `empty` (nothing matched or fitted), `skipped`
(a greeting) or `failed` (recall raised; `error_type` names it).

## What comes back

The chat-completions shape (`id`, `object`, `created`, `model`, `choices`)
with these differences, each deliberate:

| Field | Meaning |
|---|---|
| `model` | the configured model's name, or `unknown` when the provider names none; never the requested model |
| `choices[0].finish_reason` | always `stop`; the configured model does not report one, so `scone.provider_completion` is `unverified` |
| `usage` | absent; nothing counted tokens |
| `scone.requested_model` | the `model` the request named |
| `scone.not_forwarded` | every request field the model did not receive: `model` and any sampling field sent |
| `scone.recall` | `status`, `items` (`episode_id`, `chunk_id`, `source`, `created_at`), `claim_ids`, `omitted_count` (candidates recall found that did not reach the block, by the limit, the byte budget or the same-session rule), `limit`, `max_context_bytes`, `low_confidence`, `degraded`, `error_type`, `session_turns` (see [Sessions](#sessions)), and `query_formulation` when a long message was reformulated |
| `scone.injected` | `null` when nothing was injected; otherwise `role` (`system`), `boundary`, `bytes` and `sha256` of the wrapped block exactly as the model received it |
| `scone.capture` | `status` (`captured`, `partial` when the user message was kept and the reply was not, or `failed` when the user message was not kept), `episode_ids` of what was kept, `error_type` of the first failed write, and `user` and `assistant`, each with `status` (`captured`, `failed`, `blank` for a reply with no text, or `not_attempted` for a reply whose user message failed), `episode_id` and `error_type` |
| `scone.session` | the session name: the header's value, or the name drawn for a request without one |
| `scone.session_id`, `scone.turn_id` | the stored identity the turn was kept under (`openai:` and the name), and this turn |

Headers repeat the essentials for a client that reads no body extensions:
`X-Scone-Session`, `X-Scone-Recall-Status`, `X-Scone-Recalled-Episodes`
(comma-separated episode ids), `X-Scone-Injected-Bytes` and
`X-Scone-Capture-Status`.

A reply is returned even when keeping it failed: the model has already
answered, so `scone.capture` says which message was not kept and
`error_type` says why. If the key's access is withdrawn while the model
answers, the request is a 401 and nothing is kept.

## Sessions

`X-Scone-Session` (1..121 letters, digits, `.`, `_`, `:` or `-`) names the
conversation. The response's `X-Scone-Session` header is the name to send
back. Without the header every request is a session of its own with a random
name, so a later request recalls earlier ones like any other memory.

The turns are kept under `openai:` and the name, never the name alone, so a
session named like a document's source (`ops-handbook`) does not hide that
document from recall or file its turns beside it, and a session can never take
a native conversation's id and land in its transcript. A document written with
a source that itself begins `openai:` is in this route's namespace, and a
session of that name treats it as its own turn.

A session's stored turns that the request resends are not recalled: the model
already has them in `messages`. Those it does not resend are recalled like any
other memory, so a client that trims its history, or sends only the latest
message under a fixed session name, still has its earlier turns. Before
recalling, the route looks up to 20 passages of the session's own turns
nearest the latest message (a recall scoped to the session, not a walk of the
space); a turn is held when any of its passages is inside a message the
request sent, and every other turn found is admitted. `scone.recall.session_turns`
records this: `status` (`probed`, `skipped` for a request without a session
name, or `failed` with `error_type`, in which case nothing is admitted),
`probe_limit`, `probed`, `limit_reached` (the probe filled, so turns beyond it
were not looked at and stay out of recall) and `admitted_turn_ids`.

## Refusals

Errors are `{"error": "..."}`. A refusal is a 422, as elsewhere in this API,
not the 400 OpenAI's own service returns; OpenAI SDKs raise it as an
`UnprocessableEntityError` carrying the message.

| Refused | Status |
|---|---|
| `stream: true` — streaming is not supported in this version | 422 |
| no messages, or a conversation that does not end with a nonblank user message | 422 |
| a body over 256,000 bytes, declared or streamed, before it is parsed | 422 |
| a message over 32,000 bytes of text, or more than 200 messages | 422 |
| a role other than `system`, `developer`, `user` or `assistant`; content that is not text (images, audio); any field not listed above (`tools`, `n`, `base_url`, `api_key`, …); malformed JSON; a malformed `X-Scone-Session` | 422 |
| no model configured (`SCONE_CHAT_URL` and `SCONE_CHAT_MODEL`) | 422 |
| a key with the `read` role: the route writes | 403 |
| the configured model failed or was unreachable; nothing is kept, and the body says only that the model did not return a reply (the provider's own message can carry its account details, so it stays in the server) | 502 |

`GET /v1/capabilities` lists `chat.openai_compatible`: the route is served,
which says nothing about whether a model is configured.

## Measured

A fixture space held one passage, "The vault door code rotates on the 14th of
each month", and `FakeChat` recorded what reached it. Asked "When does the
vault door code rotate?", the model's input carried `14th` with memory and not
without it (an empty space); the injected block was 707 bytes.

Over 685 paragraphs of six of these guides (`directory-sync`,
`retrieval-and-storage`, `pdf-ocr`, `file-ingestion`, `conversation-service`,
`pdf-ingestion`), eight questions whose answer term does not appear in the
question put the answer into the model's input 5 times out of 8 with memory
and 0 without; injected blocks were 1,968 to 4,338 bytes (median 3,073). The
three misses were recall ranking, not injection: plain `/v1/recall` ranked the
answering paragraph 7th, 15th and outside the top 20, beyond the 5 kept. The
embedder was the hashing test embedder, which is not semantic, and turns the
route had kept from earlier questions took one of the five places in two of
the answers.
