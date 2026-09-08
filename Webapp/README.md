# Scone Webapp

The repository-owned React + TypeScript frontend, using React Router and Vite.
Work here, not in generated native package HTML copies.

```sh
cd Webapp
pnpm install --frozen-lockfile
pnpm dev
```

Open http://127.0.0.1:5173/memory. Vite supplies React Fast Refresh during development.
The development proxy uses the existing local Scone API on port 7437. Start
that backend first; the app does not create a mock engine or seed conversations.
Set `SCONE_DEV_API=http://127.0.0.1:PORT` to select another loopback backend.
Remote upstreams are intentionally rejected by this development configuration.

In development, the app reads the explicit access-key bootstrap from the same
local console already serving this browser. It never reads host settings or
transcripts. If no single key is exposed, the access-key dialog remains available.
Keys stay in page memory, never `localStorage` or `VITE_` environment variables.

```sh
pnpm test
pnpm typecheck
pnpm build
pnpm check:assets
```

Production packaging embeds built JS, CSS and the original Scone mark into both
native playground distributions and the Python Memory console. Node is not required to run the packaged Rust
or Python server. `dist/` and `node_modules/` are ignored. The lockfile belongs to
this app; CI and the AWS image use pnpm 9.9.0 with `pnpm-lock.yaml` as the primary
lockfile. Keep the retained npm compatibility lock synchronized when changing
dependencies. Upstream projects under `../reference/` are ignored reference material.

To test a backend-dependent UI before publishing it to running native servers:

```sh
pnpm typecheck
pnpm exec vite build
node scripts/package.mjs --output /private/tmp/scone-preview.html
```

Set `SCONE_PLAYGROUND_HTML` to that file when running `scripts/test-playground.cjs`.
This stages an embedded artifact without replacing the packaged native pages.
Run the ordinary build/package command only after the target backend supports
the required contract.

The Memory page discovers supported operations through authenticated
`GET /v1/capabilities` before loading workflows. Unknown/malformed responses and
transport failures remain visible and retryable; a failed request never means
an operation is unsupported. Deep links stay intact. See
the versioned contract in `tests/fixtures/http-capabilities.json`. This is feature discovery,
not a replacement for server-side authorization or an uptime guarantee.

## Documents: retained-source library

Open `/memory#documents` on a server advertising `episodes.list`. The library
uses authenticated `GET /v1/sources`, not recall, to browse files, notes,
conversations, observations and connector sources. Pages contain up to 25 entries
ordered by descending stored episode ID. Type filters apply server-side before
pagination; Newer/Older navigate the cursor history, and Refresh restarts at the
newest IDs for the selected type. Page counts are not whole-library totals.

Open a source to read its retained text with Markdown formatting. Filenames and local paths
remain provenance labels; only absolute HTTP(S) source URLs become external
links. Available original images use the existing authenticated, digest-checked
preview. A source record is not an approved memory claim, and retained text is
not a downloadable copy of the original file. Newly added or removed records can
change the inventory; it is not a frozen export.

Source inspections, review quotes and retrieved passages render emphasis, code,
lists and tables without modifying stored evidence. **View original Markdown**
reveals the exact retained string. Formatted full-source previews are limited to
80,000 characters; the complete original remains available below that preview.
Embedded HTML is never executed. Only absolute HTTP(S) and mailto Markdown links
are navigable; relative links and unsafe schemes are inert. Markdown images show
their descriptions without fetching remote bytes. Saved image attachments still
use the separate authenticated image viewer.

On attachment-capable servers, Add source reuses the existing note/image and
UTF-8 text-file import flow. A verified save refreshes All sources at the newest
page. No delete controls or unsupported write operations are inferred from the
read-only inventory capability. Older servers without `episodes.list` do not
show the section or receive inventory requests.

The browser boundary suite is `scripts/test-documents.cjs`, with
`SCONE_DOCUMENTS_HTML` pointing to the isolated packaged artifact above.
`scripts/test-conversations-native.cjs` also covers Documents pagination, import,
literal readback, original images and cross-space denial against a disposable
native Python service. Neither suite uses the live memory database.

## Current boundaries

Source-image previews are implemented for Playground recall, graph source
inspection, Memory Search and recent-memory results. Select an episode node in
the graph or Records view, then expand **View source images** in its inspector.
Chunk, claim and interaction IDs are not treated as episode IDs: follow their
recorded source connection to an episode first. Switching the inspected record
closes its previews and releases their temporary URLs.

Expand **View source images** to fetch the episode's saved
attachment metadata, then authenticated raster bytes. PNG, JPEG, GIF and WebP
previews validate the recorded size and SHA-256 digest before creating a temporary
browser object URL. Downloads are bounded to 25 MB per image; URLs are revoked
when previews unmount. This byte limit is not a decoded-pixel memory guarantee.
Other media types remain metadata-only; no inline SVG, HTML or PDF rendering.

The Add source form appears when `episodes.attachments` is explicitly true. It
saves a note with an optional original raster image, verifies the attachment
receipt against the selected bytes, then reads back the saved episode link.
Identical text may reuse an existing episode and attach the image there. Closing
before submitting makes no write; while saving, controls lock against repetition.
Upload and episode/link writes are not atomic. An unconfirmed outcome locks the
form and directs the user to inspect Search rather than automatically retrying.
Do not navigate away during saving; aborting a request does not undo server work.
The uncertain-write lock is local to this mounted form, not a durable receipt or
cross-navigation guarantee. A reload/navigation can discard it; server-side
operation tracking and recovery remain follow-up work.

The same form offers **Import text file** for UTF-8 text, Markdown and code
(up to 1 MB). Selection reads locally and shows a literal, non-executing preview;
Save submits the complete decoded text with kind `file` and source basename.
The read-back must match exactly, including line endings and any UTF-8 BOM,
before success is shown. A deduplicated episode retains its earlier metadata;
the saved-text inspection shows its recorded source. No separate binary file
attachment or download is created. Invalid encodings, empty/binary content and
unsupported formats are rejected; PDF parsing and URL fetching are not included.

The native integration test uses the Python attachment API and an isolated
in-memory store, not the live workspace. Rust attachment parity, automatic image
capture from agent sessions, bulk imports, broader file parsing, URL import, OCR and media-inclusive backup
are still unfinished. Existing `[Image #1]` text cannot reconstruct missing bytes.
Missing metadata, missing bytes and transport failures remain distinct states.
This UI is staged until the running backend is reloaded with its required routes.

Submitted Memory Search queries have independent request state. A changed query
or filter immediately removes the previous results and image previews; obsolete
requests are cancelled and late replies ignored. Failed queries expose **Retry
search** and do not retain another query's error. Requests have a 10-second
deadline. Editing the input alone does not submit a search; press Search or Enter.

The Review inbox groups recorded subjects, filters by origin and checked quotes,
and renders 25 cards per page. Sources are loaded only when expanded. Bulk
approval freezes every matching ID across pages, confirms explicitly, and runs
oldest effective date first. Confirmed outcomes survive a later failure; uncertain
writes stop the batch and require a refresh, never an automatic retry. This does
not freeze server state or provide transaction/undo guarantees. Queue polling is
every 15 seconds while visible and idle; the API still returns the full queue.

Beliefs uses inline reason/confirmation forms for close, exclude and include.
Successful receipts are checked against the submitted ID; errors do not imply
that no write occurred. Refresh is required after an unconfirmed outcome.
Closing is not deletion or undo; including does not reopen historical beliefs.
Rust's HTTP server supports close but not exclude/include. Native consoles and
CLIs retain their own independently tested workflows.

- `src/main.tsx`, `src/App.tsx`: React entry, shared shell, authentication and routes.
- `src/memory/`: typed Memory page components, coordinated with Fable.
- `src/playground/`: scoped graph, record search, source inspection and recall.
- `src/conversations/`: capability-checked text session controls, saved
  transcripts, public-text previews, request polling and source evidence. Uses
  the optional Python conversation API; provider configuration is server-owned.
  No voice/video or token-level telemetry is claimed.
- `src/api.ts`, `src/types.ts`: authenticated API client and evidence wire types.
- `src/local-session.ts`: explicit local development bootstrap.
- `src/components/DevReload.tsx`: opt-in ETag reload for native development previews.
- `src/workspace.css`, `src/identity.css`: layout and original Scone identity.
- `src/assets/`: runtime images; `assets/`: original mark and generation record.
- `/memory`, `/playground` and `/conversations/:sid?` are React routes in one application; `/` redirects
  to `/memory`. Development proxies only API traffic and local auth bootstrap.
- The Python memory server serves Memory and Playground directly (the local preview uses port
  7437). Rust packages the same Playground app; its existing native root console
  remains until Memory-page capability parity is independently verified.

The optional Python conversation service accepts `console=True` to serve the
packaged webapp at `/memory`, `/playground`, `/conversations` and session deep
links, alongside its memory/conversation APIs. Page hosting defaults to off and
never embeds a space or provider key. A full reload asks for the space key again.
The ordinary native memory servers do not provide conversation hosting or a
configured provider. For separate hosting, route these page paths to the React
application and `/v1/conversations` to the optional service. Vite handles SPA
paths in development. Adding a route does not configure or launch a provider.

Saved sessions remain readable without a configured text model; starting and
sending require one. On reopening, the optional conversation API's
`latest_request_id` identifies the latest accepted turn without guessing from
request-ID order. The UI distinguishes a completed reply whose text was forgotten
from one that cannot currently be read, and never resends either automatically.
It rechecks known receipts, clears revoked reply evidence and protects newer
commands/session selections from stale polling responses. These are outcome
receipts, not proof of provider delivery or a reconstructed token stream.

When the service explicitly advertises `streaming: true` with an SSE
`active_window` contract, accepted turns also show a **Live preview**. This reads
real public-text chunks using authenticated fetch; the key is never put in a URL,
redirects are rejected, and no stream is opened on older aggregate-only servers.
The preview is separate from Saved messages. Only the existing polled receipt
and transcript establish a retained reply; streamed text is not a capture receipt.

**Reconnect preview** resumes the same turn from its last sequence without
posting another message. There is no automatic reconnect loop. If the server has
evicted a prefix, the UI labels the gap and shows only the latest continuous
segment instead of stitching disconnected passages together. A reload can read
the active window, not an unlimited history of chunks. The final saved reply
remains readable through ordinary receipts/transcripts when available.

Each SSE frame is bounded to 524,288 decoded UTF-16 code units including framing,
and the visible preview to 1 MiB of UTF-8 text. Reaching that limit stops the stream and waits for the
saved outcome. Thirty seconds without incoming stream bytes interrupts the
preview; receipt polling continues. These are client safeguards, not provider
queue or whole-browser memory guarantees. Cancel, end, session/space changes,
and terminal outcomes remove provisional text and close its reader. A failed,
forgotten or unreadable reply cannot be replaced with the old preview. Hidden
reasoning is not displayed; chunk counts are not token counts. Voice and video
remain separate, unfinished transport work.

For isolated conversation UI checks, build with Vite, package with `--output`
as above, and pass that path as `SCONE_CONVERSATIONS_HTML` to
`node --test scripts/test-conversations.cjs` from the repository root. Install
Playwright or set `SCONE_PLAYWRIGHT_MODULE` to an installed module; optionally set
`SCONE_BROWSER_PATH` and `SCONE_SCREENSHOT_DIR`. Fixtures use temporary localhost
services, never the live memory store. No screenshots are written by default.

When the conversation service explicitly advertises `recall_scope: true`, the
start dialog offers source type, literal source prefix, inclusive creation-date
bounds and exact metadata matches. All selected filters must match; an empty
result never broadens recall. Dates without times mean midnight UTC. Filters
are fixed when the start request is submitted, retained for uncertain retries,
and checked against the server's acknowledgement before opening the session.
Validation rejections allow correction; uncertain outcomes do not automatically
create another session. Saved conversations expose a read-only Memory selection
summary, distinguishing no extra filters from unreported scope on older servers.
These controls narrow retrieved knowledge, not transcript capture, the session's
own history or server-side access policy.

`scripts/test-conversations-native.cjs` additionally exercises the actual Python
conversation API, native memory and Pipecat scheduler through the browser. Set
`SCONE_TEST_PYTHON` to an environment with Scone's `api` and Pipecat dependencies,
plus `SCONE_CONVERSATIONS_HTML` and the browser settings above. The Python
fixture uses temporary state and scripted model frames, not provider inference.
For an existing split local test installation, `SCONE_TEST_EXTRA_SITEPACKAGES`
can explicitly append an API dependency directory; such a run does not verify a
fresh, self-contained dependency installation.

The native browser suite also tears down and recreates the conversation service
around its retained journal and memory engine, with the model then disabled.
It verifies latest-outcome discovery, forgotten text, browser reload and scope
isolation. This is service recreation, not a killed-process or live-provider test.

The graph renders stored relationships only. Depth is a spatial layout, not a
3D simulation or confidence value. API connectivity, observed agent events and
verified host capture are distinct. Current snapshots are bounded; lossless
event replay and current Codex App capture have not been verified.
