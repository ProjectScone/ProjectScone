# Recall semantics, storage and recovery

[Package overview](../README.md) · [Architecture](../ARCHITECTURE.md)

Examples below run from `packages/memory/` unless a section names another working directory.

## What recall returns

Items carry `score` (rank within this query; the top item is always 1.0)
and `similarity` (cosine from the vector lane, when that lane saw the
chunk). Two lanes, vector and lexical, are fused by reciprocal rank with a
small recency term, capped at two chunks per episode. A lane that fails is
named in `degraded` and the other lane still answers. `context_reduction`
is the share of the space's bytes that were left behind.

`top_similarity` is the best cosine the vector lane saw for the query.
With `SCONE_SIMILARITY_FLOOR` set (a cosine, e.g. `0.45`), a recall whose
best hit falls below it, or that finds nothing, carries
`low_confidence: true` so the reader can decline to answer from weak
evidence; without a floor the field is `null` and nothing is judged. The
floor has no default because the right value depends on the embedder:
`scone-memory bench` prints, for each candidate floor, how many
no-evidence questions it would catch and how many answerable ones it
would wrongly withhold, and that sweep is where a floor comes from.

`history=true` (CLI `--history`) adds, for every matched fact, the closed
facts that held before it for the same subject and predicate, oldest
first, each with its interval and closing reason, bounded by `as_of`.

`graph_boost=true` adds a third lane, the entity lane, for questions whose
answer sits one relation away. "Which city is Ana Alves's employer in?"
is answered by a passage about her employer that names neither her nor
any word of the question. The lane works in four steps:

1. It finds the entities the question names, the longest name first.
   Common words never count.
2. It adds up to three neighbours of each: those the most facts relate
   to it. At most 12 entities are used.
3. It makes one extra lexical search for passages naming any of them,
   under the same filters as the other lanes.
4. It keeps a passage only if the passage really names one of them: a run
   of the passage's words, read the same way names and questions are,
   must equal one of the entity's spellings.

Passages naming a neighbour but none of the question's entities rank
first, because the other lanes cannot reach them. The lane is fused at
weight 2 beside the others' 1. Its items carry `lanes.entity`, and the
response lists the entities searched under `entities`.

The projection is built for it within the graph budget and then kept. If
it is not ready in time, `degraded` says `entity: projection_building`
and the other lanes answer. The same holds if building it fails (a store
error, say): `degraded` names the error, and the lane is the only thing
lost. If the projection's read was capped, `degraded` also says
`entity: graph_read_capped fact_limit`, because the lane then saw only
part of the ledger. Names are matched by the same words the variant tier
uses. Symbols that make a name (`C++` against `C#`) and accents
(`José` against `Jose`, however the accent is encoded) keep names apart.
The lane is off by default; with it off, recall is unchanged. Advertised
as `recall.graph_boost`.

On the synthetic bridge set (`bridge-v1`: 20 people, their employers and
where those are based, with distractors repeating the question's words),
the second-hop passage reaches the top five for none of the questions
with the lane off and for all 20 with it on. A single synthetic set
shows the mechanism works; whether it helps on real questions is for
the retrieval benchmarks to show before the default changes.

## What survives a crash

A `remember` marks the episode's identity in the document store before
it writes, and clears the mark only after the rows and the vectors are
all durable. On the next open the engine finishes or forgets whatever
was cut off in between: chunks are rebuilt from the stored content when
they are missing, vectors are re-embedded when they are missing, and a
mark with no episode behind it is dropped. Each open that had anything
to repair records one `recover` event with the counts. The guarantee:
an episode you can see is complete, or it is absent; it is never
searchable by one lane and not the other. This holds on every document
store below (the recovery contract in `tests/memory/test_contract.py` plays the
crash at each step of the write).

## Stores and what each one promises

The following description and table retain the original validation record,
not a current test count or a guarantee that every optional service ran in a
particular CI invocation:

> Every store below runs the same 37 contract tests (`tests/memory/test_contract.py`)
> and every evidence sink the same 8 (`tests/memory/test_events.py`). "Verified"
> says how: embedded means in-process in the test suite and CI; container
> means against a real server in Docker locally and as a CI service.

Consult [current measurements and their scope](scaling-validation.md) separately.

| Store | Documents | Vectors | Evidence | Verified | Notes |
|---|---|---|---|---|---|
| in-memory | yes | yes | yes | embedded | reference implementation |
| SQLite | yes | yes | yes | embedded | FTS5 lexical lane; WAL; schema stamped, additive steps from v5 (quote column, inflight table) |
| MongoDB | yes | | yes | container (local; CI when `SCONE_TEST_MONGO_URL` is set) | `$text` lexical lane; TTL retention |
| PostgreSQL + pgvector | yes | yes | yes | container | tsvector lexical lane, HNSW cosine, one pool for all three |
| Elasticsearch 8 | yes | yes | yes | container | BM25 lexical lane, float32 HNSW (int8 would round cosine), refresh per write |
| Qdrant | | yes | | embedded (local mode) and container | payload filters server-side |
| Redis Stack | | yes | | container | TAG/NUMERIC prefilters inside the KNN query |
| Chroma | | yes | | embedded; server by URL | width recorded in collection metadata |
| LanceDB | | yes | | embedded | SQL predicates, quotes doubled |
| Milvus | | yes | | embedded (Milvus Lite); server by URI | filter expressions with JSON literals |
| any LangChain VectorStore | | yes | | embedded (`InMemoryVectorStore`, FAISS) | needs a `filter_builder` and a stated `score`; see [the bridge guide](integrations.md#any-langchain-vectorstore-as-the-vector-index) |

Query terms come from one shared tokenizer, `retrieval.lexical.tokenize`.
A term is a run of letters, combining marks and digits in any script, after
NFKC normalisation and case folding, so "Zürich", "हिन्दी" and "ＡＢＣ" stay
whole; underscores still split words. Scripts written without spaces
(Chinese, Japanese, Korean, Thai, Lao, Khmer, Myanmar) become overlapping
pairs of characters. The in-process BM25 lane, SQLite's fact index and the
hash embedder match those pairs directly. Stores that tokenize documents
themselves do not: SQLite FTS5 and PostgreSQL keep a whole unspaced run as
one token, so their chunk lanes still cannot match inside Chinese or
Japanese text, and Elasticsearch's standard analyzer re-splits the pairs by
its own Unicode word rules, so matches there are looser. Anything that stores tokens or values hashed from them
records `TOKENIZER_VERSION`; SQLite rebuilds its fact index when that
version or Python's Unicode tables change.

Intentional differences: lexical scores are each store's own (BM25,
`$text`, `ts_rank`); only their order reaches the shared fusion algorithm.
Both rankings and raw scores can differ across stores. Retention is a TTL
index on MongoDB and a clock-driven sweep elsewhere. No store migrates
another build's data: each stamps a schema version and refuses a
mismatch, SQLite excepted for the one recorded step.

## Knowledge graph: entities and how they relate

A space's knowledge map is derived from its fact ledger. It is never a
second store: the same facts always give the same entities, ids and digest,
and every item points back to the facts behind it.

- **Entities** are what claims are about. Every subject is an entity. An
  object is an entity or a value, and `entities.classify` decides which by
  ordered rules. Each decision names its rule: a name shape, a determiner
  name, a key some subject carries, a predicate whose object is a thing, and
  so on. Dates, amounts, identifiers, prose and pronouns are values.
- **Relations** are entity-to-entity claims grouped by
  `(subject, predicate, object)`, with direction kept.
- **Attributes** are value claims grouped by `(entity, predicate, exact value)`.
  Values keep their exact text: `3 MB` and `3 mb` are two attributes.
- **Names** are the entity's recorded spellings. The label is the most
  common one, and casing is recovered from the source quote, so `alice chen`
  is labelled `Alice Chen`.
- **Kinds** (person, organisation, place, project, product, event, concept)
  are inferred hints from the predicates around an entity. They carry
  `kind_status: "inferred"` and list the fact ids that suggested them. Hints
  that disagree give `"conflict"` and no kind, never a guess.

Names are one entity when their keys match: case folded and spacing
collapsed, nothing looser. `Lisbon` and `Lisboa` stay two entities. A value
whose case can carry meaning (`MB` against `mb`, versions, paths) never joins
by folding in either direction. The same rule, `memory.identity.join_match`,
decides every join in retrieval too.

### Routes

Every route here is read-only and scoped to the caller's key. Stored
text is answered exactly as the ledger holds it. The ledger accepts
lone surrogates, which UTF-8 cannot encode, and the API writes each
one as its JSON escape (`\ud800`), which reads back as the same text.
Every route, not only these, answers this way rather than failing with
a 500. The CLI's JSON output does the same. The view
and the entity list are advertised as `graph.knowledge` and
`entities.read`. The report, paths and export each have their own
capability, named with the route.

#### `GET /v1/graph/knowledge`

| Parameter | Default | Meaning |
| --- | --- | --- |
| `status` | `current` | Which facts count (see below) |
| `as_of` | now | RFC 3339 moment for `current` and `history`; anything else is a 422 |
| `limit` | 150 (1–1000) | Most entities shown, ranked by claims in this view |
| `attribute_limit` | 300 (0–5000) | Most attributes shown |
| `cursor` | none | The next page of the ranking, from `coverage.next_cursor` |
| `seed` | none | Names or ids (repeatable, up to 24) to walk out from instead of ranking |
| `hub_degree` | 64 | In a seeded view, an entity with more relations is shown but not walked through |
| `direction` | `both` | In a seeded view, which way relations are followed: `out` (subject to object), `in` (object to subject) or `both` |
| `hops` | none (1–8) | In a seeded view, the most steps from a seed |
| `usage`, `usage_since` | off, all | Count, on each entity and relation, the recent recalls that returned one of its facts |

Without `seed`, the view pages through the ranking. `coverage.next_cursor`
names the next page and is absent on the last. A cursor encodes the
projection digest it was issued for. Once the graph has changed, the
cursor is refused with a 409 (`cursor_stale`), because a page from a
different graph would repeat or skip entities. Each page shows the
relations among its own entities.

With `seed`, the view walks out from the named entities breadth first, in
both directions, taking each entity's neighbours in order of the facts
behind the relation, until `limit` entities are shown. An entity with
more than `hub_degree` relations is shown but not walked through, unless
it is a seed, and `hub_skipped` says so. Entities the walk never reaches
from its seeds are left out as `outside_walk`, so a seeded view is
always marked truncated when it is not the whole space. Up to 24 seeds
are taken; more is a 422. An unknown seed is a 404, and an ambiguous one
a 409 listing the candidates. Both keep the read's coverage, so a capped
read is never taken for absence or a complete list. `filters.seeds`
gives the resolved ids and `filters.hub_degree` the degree the walk used.
A walk follows relations in `direction`. `in` finds what depends on a
seed: what is based in Lisbon, who lives there, then who works at what
is based there. `out` follows only what the seed points at.

`hops` stops the walk after that many steps. When a step more would
have reached more, `hop_limit` is among the reasons. Each entity in a
seeded view carries its `hop` from the nearest seed (0 for a seed).
`filters.direction` and `filters.hops` echo the walk. `direction` or
`hops` without a seed is a 422.

With `usage=true`, every shown entity and relation carries `recalled`:
how many of the space's recent recalls returned a fact it stands on.
This shows which parts of memory answer questions and which sit unread
(graphify's usage overlay).

- The recalls counted are the most recent 1,000, or those since
  `usage_since`. `coverage.usage` says how many were read and whether
  more were left.
- A recall that returned two facts of one entity counts once for it.
- Only counts leave the event log. No query text is read.
- An engine that keeps no events answers `recalled: null` with
  `usage.available: false`.
- Advertised as `graph.knowledge_usage`.

Paging, seeded walks and walk shapes are advertised as
`graph.knowledge_paging`, `graph.knowledge_seeds` and
`graph.knowledge_walk`: a server without them ignores those parameters.

Status modes:

- `current`: facts that hold at `as_of` within their validity interval, and are not excluded.
- `history`: every fact that ever held and had begun by `as_of`, not excluded.
- `proposed`: proposals awaiting review, not excluded.
- `all`: everything, excluded facts included.

The view is projected from the counted facts alone, so classification and
kind hints come only from them. An excluded claim, or one that begins after
`as_of`, cannot turn a value into an entity or suggest a kind.
`coverage.facts_counted` reports how many facts counted. A relation or
attribute appears when at least one of its facts counts, and its `fact_ids`
and `support` cover only those facts. Relations are shown only when both ends
are among the shown entities.

The response (`api.entity_routes.KnowledgeView`):

```json
{
  "schema_version": 1,
  "space": "alpha",
  "projection": {"version": "scone.entities/1", "classifier": "objects/1", "kinds": "kinds/1",
                 "id_scheme": "scone.entity/1", "digest": "<sha256>", "revision": 12},
  "filters": {"status": "current", "as_of": "2026-09-11T11:00:00.000Z"},
  "entities": [{"id": "ent:…", "key": "alice chen", "label": "Alice Chen",
                "names": [{"text": "Alice Chen", "count": 2}], "kind": "person",
                "kind_status": "inferred", "kind_basis": [1, 7], "flags": [], "claims": 3}],
  "relations": [{"id": "rel:…", "subject_id": "ent:…", "predicate": "works_at", "object_id": "ent:…",
                 "fact_ids": [1], "support": {"facts": 1, "active": 1, "closed": 0, "proposed": 0,
                 "excluded": 0, "quoted": 1, "unquoted": 0, "unsourced": 0, "stated": 0,
                 "extracted": 1, "inferred": 0},
                 "first_valid_from": "…", "last_valid_until": null}],
  "attributes": [{"id": "att:…", "entity_id": "ent:…", "predicate": "joined_on", "value": "May 2021",
                  "literal_kind": "date", "fact_ids": [2], "support": {"…": 0}}],
  "coverage": {"facts_read": 9, "facts_counted": 9, "facts_limit": 50000, "entities_total": 6, "entities_shown": 6,
               "relations_total": 4, "relations_shown": 4, "attributes_total": 3,
               "attributes_shown": 3, "truncated": false, "reasons": []}
}
```

`support` counts the facts by status, grounding (`quoted`: a source quote,
checked against the source when the claim was recorded and not re-read by
this route; `unquoted`: a source with no quote; `unsourced`: stated with no
source) and origin. Inspect a relation by reading its `fact_ids` through
the fact routes.

`coverage.reasons` names every limit that applied:

- `entity_limit` or `attribute_limit`: the view's budget.
- `fact_limit`: more than 50,000 facts, so only the newest were projected.
- `store_read_cap_reached`: the store may have cut its read short
  (Elasticsearch stops at 10,000 rows).

When `truncated` is true, what is drawn is a sample, not the graph.

Ids are hashes of the space and the item, so they are stable across
requests and stores. A client should drop cached selections when
`projection.version`, `classifier`, `kinds` or `id_scheme` changes. The
`digest` changes whenever anything in the projection does.

`groupings=true` adds a separate `groupings` block, marked
`"basis": "computed"`, that is analysis and never facts:

- `membership` maps each shown entity to its community;
- `communities` gives each community's id, label, size and shown members;
- `importance` gives each shown entity's degree, PageRank, betweenness and
  participation across communities;
- `coverage` says what the analysis itself covered: entities analysed,
  any cap it hit, and `betweenness`, which is `exact` or `sampled:64`
  with `betweenness_estimated` true.

It is computed from the same counted facts as the view (see the report
below). Three limits stay separate, because a client has to show each one
differently. The view's `coverage` is paging: what this response left out.
The groupings' `coverage` is the analysis: what it was capped at, and
whether betweenness is an estimate. A read cap (`fact_limit`,
`store_read_cap_reached`) appears in the view's reasons. A view can show
every entity while its betweenness is still estimated.

#### `GET /v1/graph/report`

A whole-space analysis, `format=json` (default) or `format=markdown`, for
the same `status` and `as_of`:

- **Communities**: found by modularity optimisation over recorded
  relations, weighted by the facts behind each pair, and split into
  connected parts. Each is named after its most central members, with
  cohesion, kinds and predicates.
- **Central entities**: by PageRank, with degree, fact weight,
  betweenness (exact up to 500 entities, from 64 evenly spaced sources
  beyond) and participation across communities.
- **Bridging entities**: those whose links spread across communities.
- **Surprising connections**: relations between communities, with the
  rarest link between two communities first. Each carries its fact ids
  and the reason.
- **Questions worth asking**: built from real names, each citing its
  entities, relations and facts. They cover how two communities connect,
  what a bridging entity does, what kind a conflicted entity is, and what
  an entity known only by its values connects to.

Two parameters tune the analysis:

- `resolution` (default 1, above 0 and at most 10) sets how fine the
  communities are, as modularity with a resolution. Above 1 favours
  smaller communities, below 1 larger ones. The knowledge view's
  groupings take the same parameter.
- `exclude_hubs` (a degree percentile, 50 to 100) leaves entities whose
  number of neighbours is above that percentile out of the central
  entities, and lists them under `hubs_excluded` instead. A hub that
  everything links to otherwise leads every ranking. Excluded hubs stay
  in their communities.

The report echoes both under `analysis`, and `analysis.coverage` carries
`resolution`.

Results are deterministic: the same facts give the same report. The report
states the projection digest and analysis version it came from, and
`coverage` lists every limit that applied. `analysis.coverage` has the
same shape as the groupings' coverage. When betweenness is estimated, the
Markdown table heads the column "Betweenness (estimated)" and the coverage
section says from how many sources.

Every name, predicate, value and question in the Markdown form comes from
the ledger, so each is escaped. A stored `Alice | Injected` stays in one
table cell, and a stored `<img>` tag shows as text, not an image. Line
breaks inside a name become spaces. The JSON form keeps the raw values. A 50,000-fact space takes about
two and a half seconds. Advertised as `graph.report`.

A caveat from the ledger itself: a subject and predicate hold one value at
a time, so a newer `alice knows cho` closes `alice knows ben`. `current`
therefore shows only the latest object of each predicate, and `history`
shows them all.

#### `GET /v1/graph/schema`

What the space's graph is made of: its vocabulary, not its contents.
Read it before asking the graph anything, to know what it could be
asked. It is LlamaIndex's schema introspection, taken from the ledger
as it is rather than declared ahead of the facts. Advertised as
`graph.schema`.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `status`, `as_of` | `current`, now | Which facts count, as for the view |
| `limit` | 200 (1–1000) | Most predicates listed, most used first |
| `max_bytes` | 64,000 (1,024–1,000,000) | Most bytes the listed predicates may take as JSON |

The answer has four parts:

- `kinds`: each entity kind with its `status` and how many entities
  have it. The status says how the kind is known, for example
  `inferred`. Entities with no kind are counted under `null`, split into
  `unknown` (nothing hinted at a kind) and `conflict` (the hints
  disagree). At either end of a join, a contested entity is shown as
  `contested`.
- `predicates`: each predicate with its fact, relation and attribute
  counts. `joins` lists the kind pairs it connects, such as `person` to
  `organisation`, with counts. `values` lists the value kinds it takes,
  such as a `person`'s `quantity`.
- `totals`: the whole view's counts.
- `predicates_total`, `truncated` and `truncated_by`: how many predicates
  there are, whether the list is partial, and why. `truncated_by` lists
  `read` when the ledger read was capped, so predicates it never saw are
  missing from the total. It lists `limit` or `max_bytes` when those cut
  the list.

A predicate longer than 200 characters is shown clipped, with
`clipped`, its full `length` and `term_sha256`. That way two long
predicates that start alike can still be told apart, and one huge
predicate can't make the answer huge.

Like every read, it carries `projection`, `filters`, `complete` and
`coverage`. A kind is only as known as the entity's `kind_status` says.
The MCP tool `memory_graph_schema` gives the same as lines for a model:
`kind:` lines, then one `predicate:` line per predicate, with its
shapes, such as `(person) -> (organisation) x2`.

#### `GET /v1/entities`

The same entities as a ranked list. It takes `status`, `as_of`, `limit`
(default 100, 1–1000) and an optional `q`, which matches keys and recorded
spellings case-insensitively. It returns `api.entity_routes.EntityList`,
where `coverage` counts the matches.

#### `GET /v1/entities/resolve`, `GET /v1/entities/{id}`, `GET /v1/graph/path`

- `resolve?name=` returns the entity a name or id means, by the first tier
  that matches:
  - `id`: the id itself;
  - `key`: the same key, with case and spacing folded;
  - `variant`: the same spelling once compatibility forms are unified,
    opening quotes and brackets are stripped from a word's start, closing
    punctuation and a possessive from its end, and a leading article or
    title is dropped. So `Dr. Alice Chen` finds `alice chen`, and
    `ACME, Inc` finds `acme inc`. Symbols that make a name are kept, so
    `C#` never finds `c++`, `/tmp/a/b` never finds `/tmp/a-b`, and
    neither `.env` nor `../config` loses its leading dots. This is for
    lookup only: identity stays with the key;
  - `prefix`: a key that begins with the name at a word boundary, so
    `alice` finds `alice chen` but `ali` finds nothing;
  - `tokens`: a key holding every word of the name.

  It returns `resolved` with one candidate, `ambiguous` (it never
  guesses), or `not_found`. An ambiguous name lists the first `limit`
  candidates by key (default 20, up to 200), with `candidates_total` and
  `truncated` saying whether any were cut. A 409 from `path` carries the
  same fields.
- `/v1/entities/{id}` is an entity page: outgoing and incoming relations
  grouped by predicate, the entity's values, and every fact behind them.
  Each fact is re-read from the store, and its quote is checked against the
  retained source now, giving one of:
  - `quote_verified`;
  - `quote_not_found`;
  - `quote_source_missing`;
  - `quote_source_mismatch`: the store returned a source whose space or
    id is not the one the fact names, so the quote is not checked against
    it;
  - `source_unquoted`;
  - `stated`.

  At most `limit` relations are shown per direction, and `coverage` counts
  the rest. An unknown id is a 404.

  The page is built from one read of the ledger and then re-reads its
  facts. A write that lands between the two (an exclusion, say) would
  leave the relations counting a fact that the re-read shows excluded. The
  page compares the space's revision before its read and after its
  re-reads, and reads again when they differ. `consistent` is true when
  they matched. It is false only if the space changed through all three
  attempts, and then `coverage.reasons` holds
  `ledger_changed_during_read`.
- `path?from=&to=` connects two entities, named or by id. It returns up to
  `limit` distinct shortest routes within `max_hops`.
  - Relations are walked in either direction, and each hop says `forward`
    or `reverse` and lists its facts.
  - An entity with more than `hub_degree` neighbours is never passed
    through; it can only be an endpoint. The response lists such hubs.
  - `status` is `found`, `none_within_limit` (reachable but farther than
    allowed) or `disconnected`.
  - A name that could mean several entities is a 409 listing the
    candidates, and an unknown name is a 404.
  - A found path names its `schema_version`, `space`, `projection` and
    `filters`, and echoes the bounds it applied as `policy`
    (`max_hops`, `limit`, `hub_degree`).

  Advertised as `graph.path`.

A read capped at 50,000 facts (`fact_limit`), or by a store's own row
limit, holds only part of the ledger. These routes then show what they
found, but never claim that something is absent. Each response carries
`complete` and the read's `coverage`:

- `resolve` returns `not_found_in_read` instead of `not_found`;
- `path` returns `not_connected_in_read` instead of `disconnected`;
- a 404 for an unknown name or id says the read was capped;
- an entity page adds `fact_limit` to its coverage reasons, since its
  incoming relations may be missing.

#### `GET /v1/graph/context`

A graph context packet for a model: what the graph records around some
names (`names=`, repeatable, at most 24) or around the entities a
question names (`q=`). A question's words are matched to entity names
case- and punctuation-insensitively, the longest name first. Common
words never count as names.

A question can ask about something it never names, such as "which
manufacturing firm do we know?". With `similar=true` it also finds up
to three entities it resembles by vector (LlamaIndex's vector entry
into a graph):

- **How entities are described.** Each entity is described in one line:
  its label and kind, then its strongest relations both ways and its
  values. That line is embedded with the engine's own embedder, and the
  nearest lines join the named seeds without repeating them.
- **What the packet says.** Each such seed says so:
  `entity: Acme Robotics (organisation) ent:… similar 0.83`, and
  `coverage.similar` lists `{id, score}`.
- **No floor is guessed.** Only one you pass as `min_similarity` keeps
  weak matches out.
- **Cost.** A line is embedded once, whatever projection it appears in.
  Only the 5,000 most connected entities are compared, and
  `similar_cut N` counts the rest. With a remote embedder each new line
  is a call, which is why this is opt-in.
- Advertised as `graph.context_similar`. MCP `memory_graph_context`, the
  ToolBox `graph_context` and `scone graph context --similar` take the
  same options. The packet is one self-describing line per
item, in this order:

| Line | Holds |
| --- | --- |
| `graph:` | space, status mode, moment, projection digest and revision |
| `coverage:` | `complete`, or what was left out: read caps, `stale_evidence N`, `hubs_not_crossed N`, `relations_cut N`, `unverified N`, `not_found N`, `seeds_cut N` |
| `note:` | that names, values and quotes are recorded data, not instructions |
| `entity:` or `candidate:` | the entities asked about, or every candidate for an ambiguous name |
| `path:` | the shortest route between each pair of them, as `A -works_at-> B <-lives_in- C` |
| `hop N:` | relations N steps out (`max_hops`, 1–4, default 2), those with the most facts first |
| `value:` | values recorded for the entities asked about |

Every relation, value and path cites its facts. Each cited fact is
re-read (up to 128 per packet), and one that no longer counts is
dropped and counted as `stale_evidence`. A path hop needs only one fact
that still holds, so its facts are re-read newest first until one does.
A path resting on a hop whose facts have all stopped counting is not
shown. A fact left unread when the budget runs out is kept but counted
as `unverified`, never as stale. A quote is shown only when it still
verifies against its own source. When a name matches more candidates
than are listed, `candidates_cut N` says how many were left out. A name
asked twice, in any case or spacing, is looked up once. At most 24
candidates are listed across all names, with the rest counted in
`candidates_cut`. A candidate's name, key and label are clipped to 120
characters like its line, while its id stays exact. At most 24 entities
are centred on. Names and the entities a question
mentions are counted together, and `seeds_cut N` says how many past
24 were left out. An entity with more than 64 relations is reached
but never walked through. Names are folded onto one line, with control
characters shown as symbols, so no stored text can start a line of its
own. The text fits `max_bytes` (512–64,000, default 8,000). It is cut
only between lines, with an `omitted:` footer counting what was left
out, and it is byte-identical for the same ledger and moment.
`max_bytes` bounds the text alone. The JSON around it is bounded on its
own terms: at most 24 clipped candidates, 24 seed ids, and one hub id
per relation walked.

The response carries `status` (`prepared`, `ambiguous` or `empty`),
`text`, `seeds`, `candidates` and `coverage`. Advertised as
`graph.context`.

The MCP server offers the same reads as tools:

- `memory_graph_context`: names or a question;
- `memory_entity`: one entity, with its relations in both directions;
- `memory_connections`: the paths between two entities;
- `memory_graph_schema`: the kinds and predicates the graph holds.

These sit beside the six tools shared with the Rust server, and none of
them writes.

The server also offers read-only resources, which a client can attach
as context without calling a tool:

- `scone://graph/report`: the knowledge report of the server's space,
  in Markdown;
- `scone://graph/schema`: its graph schema, as lines;
- `scone://{space}/graph/report` and `scone://{space}/graph/schema`: the
  same for any space by name.

A space name that cannot exist is refused, and the refusal says why. Every surface that asks the graph by name, whether HTTP,
MCP, the ToolBox or the CLI, shares one set of bounds: 24 names of 1 to
200 characters, and a question of 1 to 2,000. The graph tools take
whole numbers only, so `true` or `"5"` is refused, not read as 1 or 5.

A path answer (`memory_connections`, the ToolBox's `connect_entities`,
`scone graph path`) says "not connected" only when nothing was left
out. Otherwise the `no path:` line says why:

- `none within N hops` when a path might be longer than asked for;
- `none without crossing a hub` when a hub of more than `hub_degree`
  relations (200, as for `/v1/graph/path`) was not crossed;
- `not connected in the facts read` after a capped or torn read;
- `both names are the same entity`, rather than a path citing no facts.

#### `GET /v1/graph/sources`

One source followed through (`episode=`, an episode id):

- **sections:** from its Markdown headings, with byte spans. Plain text
  has none.
- **chunks:** the stored byte spans, each with the section it starts in.
  At most `max_chunks` are listed (default 64).
- **claims:** the facts citing the episode, at most `max_claims`
  (default 200). Each carries its quote's first exact UTF-8 byte span,
  the number of times the quote occurs, the section and chunks the span
  falls in, its grounding (`quote_verified`, `quote_not_found` or
  `source_unquoted`) and the entities it names.
- **entities:** those the claims name, each listing its claims.
- **mentions:** names of known entities found in the text, reported
  apart from claims, because a name appearing is not the source asserting
  anything about it. A lowercase single word is never taken as a name,
  and an entity a claim already names is not repeated.

Spans are byte offsets into the unchanged content, like chunk spans, so
`content.encode()[start:end]` is exactly the quote even after non-ASCII
text. `episode.content_sha256` is the SHA-256 of that exact UTF-8
content, so a client can check that the bytes it shows are the bytes the
spans point into. The episode, chunks, claims and projection are read
between two matching revisions, and the view reads again if the space
moved. `consistent` says whether a still read was reached. Caps are
reported as `chunk_limit` and `claim_limit`, and the projection read's
own caps (`fact_limit`, `store_read_cap_reached`) appear in the same
`coverage.reasons`. A store that
cannot list a source's claims says `claims_unavailable`. A missing
episode is a 404 and a forgotten one a 410, as for the episode itself.
Advertised as `graph.sources`.

#### `GET /v1/graph/timeline`

One entity's facts in valid time (`entity=`, a name or id). Every fact
the entity takes part in, as subject, as object or through a value, is
an item. Items sit in lanes: `subject:<predicate>`, `value:<predicate>`
and `object:<predicate>`. Within a lane they are ordered by `valid_from`
and then id, never by when they were written, so a backfilled fact lands
where it belongs.

- **Relations:** supersession (`superseded_by`) and stored links between
  the entity's facts (`extends`, `supports`, `contradicts`,
  `derived_from`).
- **Items:** each is re-read, with its status, validity, exclusion,
  origin, source episode and grounding (a quote only when it verifies).
- **Marker:** `holds_at_as_of` marks what held at `as_of`, which
  defaults to now. Excluded facts are shown with `excluded: true` and
  never marked.
- **Limit:** `limit` (default 200, up to 500) keeps the newest items, and
  `coverage` counts the rest (`item_limit`). A link read cut short is
  reported as `links_cut`.
- **Fence:** if the space changes while the timeline is read, it is read
  again, and `consistent` says whether a still read was reached.

A name that could mean several entities is a 409 listing the
candidates, and an unknown one is a 404. Advertised as `graph.timeline`.

#### `GET /v1/graph/export`

The view's whole graph as a file for another tool. It takes `status` and
`as_of` like the knowledge view, and `format`:

| `format` | File | For |
| --- | --- | --- |
| `json` (default) | node-link JSON: `nodes`, `links`, `graph` | NetworkX, d3, custom tools |
| `graphml` | GraphML XML | Gephi, yEd, NetworkX |
| `gexf` | dynamic GEXF 1.2: every node and edge carries the valid time it holds for | Gephi's timeline |
| `cypher` | one idempotent `MERGE` per line | Neo4j, Memgraph |
| `csv` | zip of `entities.csv`, `relations.csv`, `attributes.csv`, `about.json` | spreadsheets, bulk loaders |
| `jsonld` | JSON-LD linked data | RDF tooling |
| `obsidian` | zip of one Markdown note per entity, wiki-linked, plus `index.md` | Obsidian and other note tools |
| `wiki` | zip of `index.md`, one article per topic and one per entity, in plain Markdown links | agents reading instead of the raw ledger |
| `mermaid` | a Mermaid flowchart of the 60 most connected entities and the relations between them | GitHub, Markdown viewers, docs |

Every relation and every value carries the ids of the facts behind it,
in every format. Every file records its projection digest and an `about`
block: the filters, and what the read counted and left out. The response
headers name the same scope, so a client can bind any format's bytes to
the view it shows without opening a zip or parsing XML:

- `X-Scone-Space`;
- `X-Scone-Projection-Digest` and `X-Scone-Projection-Revision`;
- `X-Scone-Status` and `X-Scone-As-Of`;
- `X-Scone-Truncated`, true when the read was capped, so a partial
  export says so in the response as well as in the file.

How each format places values and escapes its own syntax:

- **GraphML** nodes have a `type`, `entity` or `value`. A value hangs off
  its entity by an edge whose `link` is `value`, carrying its predicate
  and facts; relations are edges whose `link` is `relation`. Every key id
  is distinct. The file is written by an XML serializer, with each
  carriage return written as `&#13;`, since a parser folds a raw one
  into a line feed. XML 1.0 cannot hold some characters the ledger
  accepts (U+0001, U+FFFE, a lone surrogate). Text shows a control as its
  Control Pictures symbol (U+0001 as ␁) and anything else as U+FFFD, and
  that element gains an `exact` field holding its original values as JSON.
- **GEXF** is a dynamic graph (`mode="dynamic"`, `timeformat="dateTime"`).
  Each node and edge carries one `spell` per stretch its facts held, so
  a relation that lapsed and resumed is absent in between, not drawn
  across the gap. Every end is exclusive, as `valid_until` is, so it is
  written as GEXF's `endopen`, which holds the instant itself and never
  appears beside an `end`. A stretch still holding has no end. On Gephi's
  timeline, a person who changed employer keeps their node, and the old
  employer's edge is gone at the moment the new one appears. Nodes, values and
  escaping are laid out as in GraphML, and the file's description holds
  the projection digest and `about`.
- **Wiki** is for an agent to read instead of the raw ledger, starting
  from `index.md`.
  - The index lists the topics, which are the report's communities, the
    20 most connected entities, and any entity in no topic.
  - Each topic article lists its members with their kinds, the relations
    inside it, and those leading out, each naming the topic at its other
    end.
  - Each entity article lists its kind, its other spellings, its topic,
    its relations both ways and its values.
  - Every statement cites its facts. Links are plain relative Markdown,
    and every page is reachable from the index. Stored text is escaped,
    parentheses included, so a stored `[x](url)` cannot pass for a link
    even to a crawler that reads links with a pattern.
  - A section longer than 200 lines continues on numbered pages beside
    its article, each linking the next. Every topic is listed from the
    index, and every entity from its topic or from "In no topic", so
    no page is ever out of reach.
- **Mermaid** draws at most 60 entities, the most connected, and up to
  300 of the best-supported relations between them. That is well inside
  Mermaid's own defaults of 500 edges and 50,000 characters. The whole
  chart, header included, is measured as the browser counts it, in
  UTF-16 units, where an emoji is two. Edges give way, then entities,
  until it fits in 45,000.
  Each edge carries its predicate and up to three of its facts. Labels
  are clipped to 80 characters. The first line, a comment, says which
  view the chart draws, what the read left out, what the chart left
  out, and that values are not drawn.
  - Node ids are the chart's own (`n1`, `n2`, ...).
  - A name is a quoted label, in which `"`, `#`, `<`, `>`, `&`, `` ` ``
    and `|` are written as Mermaid entity codes. No stored text can
    close a label or add an edge.
- **Cypher** writes `(:Entity)`, `(:Value)`, `[:RELATES]` and
  `[:HAS_VALUE]`, with the predicate a property, never query syntax.
  Strings escape quotes and backslashes, and write controls, line
  separators and lone surrogates as `\u` escapes, which Cypher decodes
  back to the same text.
- **CSV** prefixes with `'` any cell that a spreadsheet would read as a
  formula (`=`, `+`, `-`, `@`). A NUL or a lone surrogate, which CSV
  readers cannot take, is shown as ␀ or U+FFFD; `about.json` and the JSON
  formats keep it exactly.
- **JSON-LD** has two layers. Each entity node states its relations and
  values plainly, for any RDF tool. Each relation and value is also a
  `Claim` node with its `subject`, `predicate`, `object` or `value`, and
  `facts`. So two people who know Bob are two claims, each with its own
  facts, and the facts are never gathered onto Bob. Every predicate is a
  percent-encoded `p:` term, lone surrogates included, so a stored
  predicate named `label`, `key` or `@id` cannot overwrite the node's own
  fields. The digest and `about` sit on an `Export` node, and the
  document holds only `@context` and `@graph`, so every statement lands
  in the default graph. It expands under a JSON-LD 1.1 processor (checked
  with PyLD 3.3).
- **Obsidian** notes escape names as the Markdown report does. Note file
  names drop characters that file systems or wiki links misread, and a
  Windows device name (`con`, `lpt1`, even `con.txt`) gains a leading
  `_`. Each name is checked against every name already given, folded the
  way a case- and normalisation-insensitive disk folds it. On a clash it
  takes more of the entity's id, then a number, so every entity keeps its
  own note and every `[[link]]` opens one.

Entity ids are defined for any text the ledger holds, lone surrogates
included. Valid text gets the same id it always had.

The bytes are deterministic, zip timestamps included, so the same
projection always exports the same file. Advertised as `graph.export`.

### From the command line

`scone graph` reads the same projection as the routes, on the configured
store:

| Command | Prints |
| --- | --- |
| `scone graph report [--markdown] [--resolution R]` | the report, as JSON or Markdown |
| `scone graph path SOURCE TARGET [--max-hops N]` | the shortest paths, one `path:` line each |
| `scone graph context [NAMES…] [--question Q] [--max-bytes N]` | the graph context packet |
| `scone graph entity NAME` | one entity's relations both ways and its values |
| `scone graph timeline NAME [--as-of T]` | the timeline, as JSON |
| `scone graph walk NAMES… [--direction in\|out\|both] [--hops N] [--limit N]` | the entities reached from the names, each with its hop, as JSON |
| `scone graph schema [--limit N] [--max-bytes N]` | the kinds and predicates the graph holds, as JSON |
| `scone graph export --format F [--out FILE]` | the export; the zip formats need `--out` |

Every command takes `--space`, and reads the clock once, so what it
prints and the instant it says it read at always agree. `--json` prints
`path`, `context` and `entity` as the JSON that `/v1/graph/context`
returns: the packet text with its status, seeds, candidates, coverage
and filters.

A name that is ambiguous or unknown exits with status 1, after printing
its candidates or the reason. For `timeline`, that is the body the HTTP
404 or 409 answer carries, including whether the read was whole: after a
capped read, a name that was not found may still exist.

An option the HTTP route would refuse is refused here too, with exit
status 2 and the option named:

- more than 24 names, or a name outside 1 to 200 characters;
- `--question` outside 1 to 2,000 characters;
- `--max-hops` outside 1 to 4;
- `--limit` outside 1 to 1,000, or a schema `--max-bytes` outside 1,024
  to 1,000,000;
- `--resolution` outside (0, 10], or not a number;
- `--max-bytes` outside 512 to 64,000;
- an `--as-of` that is not an RFC 3339 timestamp, or whose UTC
  instant falls before year 1 or after year 9999.

### Restatements and the order facts arrive in

Asserting a claim that already holds returns the fact that holds
(`restated`). When the restatement starts on a later day than that fact,
the day is kept as an affirmation. It carries the assertion's
confidence, origin, source and quote, and the space's revision moves
once; saying the same again moves nothing. Readers still see one fact.

The affirmation matters when a backfill arrives afterwards with another
object and cuts the fact short before that day. Told Acme from 2020,
Acme again from 2023, then Globex from 2021:

- the ledger now holds Acme from 2020 to 2021, Globex from 2021 to
  2023, and Acme again from 2023;
- the resumed Acme ends where the first one ended and keeps any
  exclusion;
- its evidence is the restatement's;
- later affirmations move to it.

The same history told in any order leaves the same partition of time;
a property test checks random histories in random orders. An approved
proposal that restates what holds is kept as an affirmation too.

All five document stores keep affirmations (`fact_affirmations`), and
archives carry them. SQLite adds the table without changing the shared
schema version, so older readers ignore it. On SQLite, a placement's
writes commit together or not at all. The other stores write them in
turn, as before. The Rust core does not keep affirmations yet.

### How the ledger is read

Each graph request reads the space's ledger. It never reads it on a
recall path.

- **Paged**, when the store can page its ledger. All five document
  stores can: in-memory, SQLite, MongoDB, PostgreSQL and Elasticsearch.
  `page_facts` returns the newest facts first, in every status, below a
  cursor, at most 1,000 at a time, each page on the store's (space, id)
  index. A short page is not the end; only an empty page is. The read
  stops one row past the 50,000-fact cap, so a capped read never touches
  older facts.
- **Unpaged** otherwise: one whole-ledger list, then the newest 50,000.
  A store without a pager that stops a list at a fixed row count
  (Elasticsearch stops at 10,000) is reported as
  `store_read_cap_reached`. With its pager, Elasticsearch is read whole:
  each page is one bounded search, well inside that window.
- **Refused pages:** a page with another space's rows, ids out of order,
  rows at or past the cursor, or too many rows is refused. The read falls
  back to one list and reports `pager_rejected`.
- **Fenced:** the space's revision is read before and after. A write
  during the read makes it read again. If the space is still changing on
  the second attempt, the view reports `ledger_changed_during_read`.

`coverage.read_mode` says which way a view was read: `paged` or
`unpaged`.

### How projections are kept between requests

The engine holds each space's ledger read and the views built from it
(`engine.entities`):

- **Same revision:** a view is returned after one revision check, with
  no other store call.
- **New revision:** if the store keeps a ledger stamp (in-memory and
  SQLite do), the stamp is compared first. The stamp is a count of fact
  row writes, which triggers keep in SQLite, so every writer bumps it. If
  it has not moved (an episode was stored, say), the views are restamped
  with the new revision and nothing is read. Otherwise the ledger is read
  again. If every fact is still as it was, the views are kept and
  restamped; if not, they are rebuilt when next asked for.
- **Moments:** `current` and `history` count facts by their `valid_from`
  and `valid_until`. A view built for a moment is exact until the next
  such boundary, so time passing costs a rebuild but no read.
- **Bounds:** the facts held across spaces stay under 200,000, and the
  least recently used spaces go first. A read that never held still
  (`ledger_changed_during_read`) is never kept. A deleted space, and a
  closed engine, are forgotten.

Requests that arrive together for one space share a single read, and
one view asked for at once is built once. A view over more than 2,000
facts is projected in a worker thread, so the server keeps answering
other requests while it builds.

A graph request waits at most 5 seconds for a projection. Past that, the
build continues in the background, and the request gets a 503 with
`Retry-After: 1` and `{"code": "projection_building"}`. The next request
finds the view ready.

- **Bounded:** at most 8 build jobs (ledger reads and projections) run
  at once. A request that needs a new job past that gets the same 503
  immediately and starts nothing. A library call without a budget waits
  for a slot instead. A request whose view is built, or is being built,
  takes no slot.
- **Workers:** large views are projected on two worker threads per
  engine.
- **Closing** the engine admits nothing more, cancels every request in
  flight, and returns only once no worker thread is still projecting.
  Nothing built afterwards reaches the cache.
- **Failures:** a store that fails (its own read timing out, say) gives
  its own error, never a 503.

With 20,000 facts on SQLite, a view costs:

| When | Time |
| --- | --- |
| First read | 1.0 s |
| Same revision | 0.1 ms |
| Revision moved, facts unchanged, with a ledger stamp | 0.1 ms |
| Revision moved, facts unchanged, without one | 0.2 s |

### Limits of this first version

- Identity is key identity only: no merges of different spellings yet.
- Resolution uses key identity only; merges of different spellings come with identity decisions.

## Which embedder wrote the vectors

A cosine means something only between vectors that one embedder made under
one set of settings, and a vector's width cannot tell two embedders apart.
SQLite and in-memory vector indexes record their writer: the embedder id
plus whether contextual prefixes were embedded. The hash embedder's id also
names the tokenizer version and Python's Unicode tables, because its vectors
are hashed tokens.

The record is kept true. Every vector write checks it in the same
transaction (SQLite takes its write lock first). A write by a different
embedder turns the record to `mixed` instead of leaving another writer's
name over vectors it did not make. A rebuild marks the index `rebuilding`
before its first write, and records its writer only if that marker is still
there at the end. The engine reads the record before every recall and every
semantic duplicate search, not only when it opens, so an engine that
another process rebuilt underneath stops trusting its vectors at once.

| State | Meaning | Vector lane |
| --- | --- | --- |
| `verified` | The index records this engine's writer | on |
| `rebuilt` | This engine re-embedded every stored chunk and recorded itself | on |
| `declared` | An operator vouched for vectors stored before writers were recorded | on |
| `mismatch` | Another writer is recorded | off |
| `unknown` | Some vectors have no recorded writer | off |
| `mixed` | Vectors from more than one writer | off |
| `interrupted` | A rebuild began and never finished | off |
| `unverifiable` | The index cannot record a writer | on, as before |

With the lane off, recall still answers from the lexical lane and names the
reason in `degraded` (for example `vectors: embedder mixed: …`), and
semantic duplicate detection refuses rather than comparing. The hash
embedder is local, free and deterministic, so a store it cannot trust is
re-embedded on open, like a derived index. Any other embedder may be slow or
paid, so it is rebuilt only on request:

```bash
scone vectors            # state, recorded writer, this engine's writer
scone vectors --reembed  # re-embed every stored chunk, drop orphan vectors, record the writer
scone vectors --adopt    # vouch for vectors no one recorded (recorded as declared)
```

The same operations are `MemoryEngine.reembed_vectors()`,
`MemoryEngine.adopt_vector_identity()` and `MemoryEngine.check_vectors()`.
An interrupted rebuild is simply run again. `--adopt` applies only to
`unknown` vectors, and only if any vectors written since then came from this
same embedder; a recorded different writer, a mix or an unfinished rebuild
can only be rebuilt. If another embedder writes during a rebuild, the
rebuild fails with `VectorWriterChanged` and the record stays `mixed`.

Other vector indexes (Qdrant, Chroma, LanceDB, Milvus, PostgreSQL, Redis,
Elasticsearch, OpenSearch, ElastiCache and bridged LangChain stores) do not
yet record a writer. They report `unverifiable` and keep their previous
behaviour, so switching embedders on them still requires emptying or
rebuilding the index yourself.

## Indexed fact recall

SQLite accelerates lexical fact lookup with a derived token index. Query token
overlap, confidence/ID ordering, historical validity, and source filters retain
the ledger scan's semantics. Facts outside the requested source scope cannot
consume the result limit. The engine checks returned facts against current ledger
records; an unavailable or invalid index falls back to scanning and reports
`fact_index: unavailable` in recall degradation. Other document stores retain
their existing scan unless they implement the optional `IndexedFactSearch` port.

SQL triggers record fact edits made by older clients as well as this library.
The next lookup refreshes pending terms for that space. Existing databases incur
an initial backfill; missing or incompatible derived objects are rebuilt on open.
The index does not change the ledger schema version. Historical-chain retrieval
still scans, and common query terms can still require sorting many matching
postings. This index does not accelerate vector retrieval or model generation.

Measure exact result parity and warm lookup latency on disposable synthetic
ledgers, with initial indexing reported separately:

```sh
python -m scone_memory.testing.fact_search_benchmark \
  --sizes 1000 10000 50000 --repeats 5 --output /path/to/new-report.json
```

The diagnostic includes sparse terms, common terms, and a source filter that
rejects most higher-ranked matches. It reports full ledger rows materialized and
fact point reads; it does not claim constant-time lookup or generation accuracy.
No model, network connection, or application database is used.
