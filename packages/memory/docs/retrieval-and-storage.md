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

## Abstaining, by a floor that was measured

A similarity is a number one embedder produces under one set of settings.
It is not a probability, and a floor that abstains well for one embedder
on one corpus says nothing about another, so this framework ships no
floor and guesses none. It measures one instead:

```bash
scone calibrate bench-data/longmemeval_s.json --sample 40 --out abstention.json
export SCONE_ABSTENTION_POLICY=./abstention.json
```

- **What is measured.** Each question is asked of its own memory, and one
  question whose answer is not in that memory is asked of it too, so both
  sides are measured on the same corpus with the same embedder. Each
  candidate floor is scored by how many unanswerable questions it would
  catch and how many answerable ones it would withhold.
- **Which floor is taken.** The highest one whose share of withheld
  answers is inside `--target-false-abstain` (0.05 by default). When no
  floor is that cheap, none is written and the command says so.
- **What the policy holds:** the floor, the embedder id and width it was
  measured with, what it caught and withheld, the target, the dataset and
  when. `scone status` prints it, and `/v1/status` returns it.
- **It is refused, not reused.** An engine whose embedder or width differs
  from the policy's refuses to start, because one embedder's similarities
  say nothing about another's. A policy file that cannot be read stops the
  engine too, rather than being ignored.
- **What it changes.** A recall whose top similarity is below the floor
  comes back with `low_confidence` true. Nothing is hidden and no answer
  is rewritten: the reader decides what to do with it.
- **What it does not mean.** The rates are what the floor cost on that
  corpus, not a probability for the next question. Re-measure when the
  embedder, its settings or the corpus changes.

### What it measured here

On a 40-question sample of `bench-data/longmemeval_s.json`, with the hash
embedder and one unanswerable question per answerable one:

| Floor | Catches (of unanswerable) | Withholds (of answerable) |
| --- | --- | --- |
| 0.30 | 65% | 35% |
| 0.35 | 83% | 60% |
| 0.40 | 93% | 85% |
| 0.45 | 100% | 95% |

No floor is within the default budget of 5% withheld, so `scone
calibrate` writes nothing and says so. That is the result, not a failure
of the command: the hash embedder's cosine does not separate what this
corpus can answer from what it cannot, and a floor chosen anyway would
withhold a third of the answers to catch two thirds of the gaps. An
embedder whose similarities separate them better would be measured the
same way, which is the point of measuring rather than shipping a
constant.

## Questions about dates, answered by computation

Much of what people ask memory is arithmetic over dates: how long between
two things, how long ago one was, which came first, what order they were
in. Asking a model to do that means asking it to find the dates inside
prose and subtract them itself, which is why that class is the weakest in
every published measurement of this kind of system. Retrieval is good at
finding which passage an event phrase means, and software is exact at
subtracting dates, so the work is split along that line.

```bash
scone when "How many weeks passed between the time I sold baked goods and the time I ran the bake-off?"
```

`GET /v1/answers/temporal?q=…` (capability `answers.temporal`), the MCP
tool `memory_temporal_answer`, the ToolBox tool `temporal_answer` and
`scone when` all give the same answer.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `q` | — | The question |
| `now` | now | The moment to answer from |
| `limit` | 5 (1–50) | Passages read for each event named |
| `max_bytes` | 4,000 (512–64,000) | Byte budget for the text |
| `as_of` | now | Which moment's memory is read |

- **What it reads.** "How many days/weeks/months between A and B", "how
  many days after A did B", "from A to B", "how many weeks ago did I X",
  "how long ago", "how many months has it been since X", "which came
  first/last, A or B", and an order question that lists its events.
- **A question about one claim's own time** ("how long did Alice work at
  Acme", "when did the Lisbon office open") is answered from the ledger
  rather than from passages, because the ledger keeps when each claim
  held. The claim holding most of the question's words wins and must hold
  at least half; a claim holding nearly as much with another valid time
  leaves it undecided. A claim that still holds is counted up to the
  moment asked and says so, and the answer cites the facts behind it.
- **A question about a day** ("what did I do five days ago", "who did I
  meet last Tuesday") is answered from that day instead: the days it
  names bound the search rather than rank it, so nothing from another day
  answers it, and `status` is `recalled`. It needs one reading of one day
  and a question word; "the Wednesday two months ago" holds two readings
  and is left alone, and "how many books did I read last year" asks for a
  count rather than for the day.
- **What it refuses**, leaving the question to ordinary recall: anything
  it cannot read confidently, since a computed answer arrives stated as
  fact. That includes ages ("how old was I when …", which needs a birth
  date rather than two events), a category whose members the question
  does not name ("the order of the six museums I visited"), and events
  named in one word ("between Rome and Paris"), which are things rather
  than events.
- **How an event is grounded.** The passage holding most of the event
  phrase's words wins, the better-ranked one on a tie, and it must hold
  at least half of them. The event's day is the day that passage records.
- **When it does not answer.** `status` says why: `not_temporal` (not a
  question this reads), `ungrounded` (an event is not in memory), or
  `ambiguous`. A day is undecided when a passage from another day holds
  nearly as much of the phrase (within a tenth of it), and both days are
  shown; and when one passage holds more than one of the events, since
  the day it records is its own and the distance between them would be an
  artefact of that.
- **What comes back.** Each event cites its episode, chunk, day and a
  verbatim excerpt; the answer line gives the unit asked for and the
  exact days; and a working line shows the subtraction, so the answer can
  be checked instead of believed.
- Dates named in a question ("in May 2023", "three weeks ago", "last
  month", "the past two months") are read into windows of whole days by
  the same module, against the moment asked.

### What it scores

```bash
scone bench-temporal bench-data/temporal-40.json
```

Each question gets its own memory built from its own dated sessions, so
no question is answered from another's. The scorer calls no model: a
number answer is right when the expected answer holds that number, in
the unit asked for or in days, counting a day either way (the files
themselves say "9 days ago. 10 days including the last day is also
acceptable"); a chosen event is right when the expected answer names it
rather than the one refused; an order is right when the expected answer
puts the same events in the same order.

On the 40 temporal questions of `bench-data/temporal-40.json`:

| | Questions |
| --- | --- |
| Computed | 20 |
| — right | 16 (80% of what it computed) |
| — wrong | 4 |
| Answered from the day named | 6 |
| — returning a passage the expected answer rests on | 5 |
| Refused: not a question it reads | 8 |
| Refused: nothing in memory for the event or the day | 4 |
| Refused: an event's day undecided | 2 |

Restricting ordinary recall to the dates a question names was measured on
the same 40 questions and is not worth doing: the session the expected
answer rests on reaches the top five for 36 of 40 either way, and the two
sets differ, because an event is often told on a day other than the one
the question names. The days bound the search only for a question about a
day, where the day is what is being asked for.

The four wrong ones are grounding, not arithmetic: the phrase matched a
later passage recalling the event rather than the one recording it
("the Hindu festival of Holi", told again three weeks after the day).
Choosing the earliest passage instead was measured and is worse (16
right and 6 wrong), and so is widening what counts as undecided past a
tenth (14 right, 4 wrong). What the planner refuses to read is counted
apart, because leaving a question to ordinary recall is not the same as
answering it wrongly.

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
  `usage_since`. `coverage.usage` says how many were read
  (`recalls_read`), when the oldest was made (`oldest`), and whether
  more were left (`truncated`).
- It is a window, never all time. The event log keeps what its
  retention keeps, and `usage.retention` says what that is when the log
  says (`max_events`, `max_age_days`). A count of 0 means none of the
  recalls read returned it.
- A fact a recall returned as history (`history=true`) counts like one
  it returned as current, so the history map shows it too. Recalls
  recorded before recall events kept their history are counted in
  `history_unrecorded`.
- Nothing is guessed from an event that cannot be read. One written under
  another payload version is counted in `unsupported`; one whose fact
  ids are not whole numbers (`true` is not fact 1, nor is `1.9`) is
  counted in `malformed`. Neither is counted as a recall.
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
- `usage=true` (and `usage_since`) adds `recall_usage`: the ten entities
  the recalls read returned most, and the central entities none of them
  returned. The Markdown adds "What recall uses" and claims only the
  window it read: "Central, and returned by none of them … Earlier
  recalls, or ones the log no longer keeps, may have returned them."
  With no event log, or no recall kept, it says usage is unknown and
  names nothing as unreached. It also says what the log keeps and which
  events it could not count.
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

Each predicate says its `cardinality`: `one` value at a time, or `many`
held side by side where the engine was configured for it. The lines mark
only the many-valued, since one at a time is what a predicate does
unless someone says otherwise.

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

#### `GET /v1/entities/duplicates`

Entities that may be one thing under two names, suggested with why. The
graph joins names by the one identity rule alone (case and spacing
aside), because deciding that "Dr. Alice Chen" and "alice chen" are one
person is a decision with evidence behind it. This finds the pairs worth
that decision. Nothing is merged.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `limit` | 50 (1–500) | Most pairs suggested; more are counted as `pairs_cut N` |
| `min_score` | 0.5 (0–1) | Suggest only pairs at least this likely |
| `max_bytes` | 8,000 (512–64,000) | Byte budget for the text |
| `status`, `as_of` | `current`, now | Which facts count, and when |

- **Why a pair is suggested**, each reason named on the pair:
  - the same name once titles and punctuation, or spacing, are set aside
    ("Studio54" and "Studio 54"), which scores 1;
  - the names share words, or have words one letter apart, which catches
    a misspelling such as "Welington" or "Acme Compnay";
  - one name is the initials of the other ("IBM");
  - neighbours in common, cited by their facts, which raise the likelihood
    by up to 0.15 but never make one: two people at one firm in one city
    are two people.
- **Names are read word by word.**
  - Words are scored like a set: how many the names share, of all they
    hold. A misspelt word counts for how alike its letters are, so a
    misspelling of a long word stands on its own, while one letter in a
    short name needs shared neighbours to reach 0.5.
  - A misspelling is one edit (a letter changed, added or dropped, or two
    neighbouring letters swapped) in words of four to 32 letters. One
    letter makes another word of a short one, so Bob is not Rob; and a
    word longer than 32 letters must be spelt alike, because spelling out a
    word's misspellings costs its length squared.
  - Two names that each keep a word the other has no spelling of name two
    things, however much else they share: "University of Lisbon" and
    "University of Porto", "John Smith" and "Jane Smith". One name may
    still sit within the other ("Acme" and "Acme Robotics").
- **What keeps a pair out.** Two entities of different known kinds are never
  suggested, nor two whose names hold different numbers, read as their runs
  of digits in order ("Room 101" and "Room 102" are two rooms; "version
  1.23" is not "version 12.3"). Two related to each other are halved, since
  a thing rarely points at itself under another name, and the relation is
  named.
- **Every pair that can score is compared, and few others.**
  - Of two names alike by their words, one has all its words matched in
    the other. So each name is filed under the spellings of all its words
    (the word, and for words of four letters or more each spelling one
    letter shorter, which two words one edit apart share).
  - Each name then searches under one of its words, the one whose
    spellings the fewest names hold, and keeps only the names that hold a
    spelling of every one of its words.
  - Names that fold the same, and matching initials, are filed together.
  - A test compares every pair outright over generated names mixing
    misspellings, initials, spacing, small words and numbers, and finds
    exactly the same pairs.
  - A spelling held by more than 200 names is too common to search under,
    and is counted as `blocks_skipped N`.
  - At most 250,000 spellings are filed for one answer, counted before any
    is made. Names past that are filed under their words alone, so their
    misspellings are not looked for, and are counted as `spellings_cut N
    names`.
  - At most 1,000,000 pairs are looked at and 50,000 compared, the cheapest
    searches first; the rest are counted as `candidates_cut N blocks`.
  - On 10,000 generated names built from twelve syllables, as alike as
    names get, an answer takes under a second and says what it cut. On
    9,000 names of random words it compares 34,000 pairs, cuts nothing,
    and takes about half a second.
  - `coverage.compared` says how many pairs were compared.
- **Evidence is read again before it is shown.** Each shown pair's cited
  facts are re-read, at most 256 for one answer. A neighbour in common speaks
  for a pair only through facts found to still count. A relation between the
  two still counts against the pair when it was not read again, and is named
  "not read again" rather than cited. Facts left unread are counted as
  `rereads_cut N`, and the answer is fenced by the space's revision.
- Each pair puts the likelier canonical name first: the more connected,
  then the earlier.
- On the graph benchmark's fixture it suggests exactly its one non-case
  alias ("dr. alice chen") and nothing else.
- Advertised as `entities.duplicates`. MCP `memory_entity_duplicates`, the
  ToolBox `find_duplicates` and `scone graph duplicates` take the same
  bounds.

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
  Vectors are kept by the embedder's id and dimension, so one model name
  at two sizes is embedded at each. Only the 5,000 most connected
  entities are compared, and `similar_cut N` counts the rest. With a
  remote embedder each new line is a call, which is why this is opt-in.
- **An unusable answer is said, not kept.** The embedder's answer is
  checked whole before any of it is kept or scored: one vector per text,
  each of the declared dimension, every value a finite number. When it
  fails, or the embedder raises, the packet keeps its named seeds and
  says `similar_unavailable (…)`, in words of ours, never the
  provider's.
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

#### `GET /v1/graph/match`

A structured question over the graph: triple patterns joined by shared
variables (LlamaIndex's text-to-Cypher and Cypher templates, done safely).
"Who works at an organisation based in Lisbon?" is two patterns:

```
GET /v1/graph/match?where=[{"subject":"?who","predicate":"works_at","object":"?org"},
                           {"subject":"?org","predicate":"based_in","object":"Lisbon"}]
```

| Parameter | Default | Meaning |
| --- | --- | --- |
| `where` | required | A JSON array of 1 to 6 patterns, each `{subject, predicate, object}`; at most 10,000 characters |
| `returns` | every variable | The variables to answer with (repeatable) |
| `limit` | 20 (1–100) | Most rows answered |
| `status`, `as_of` | `current`, now | Which facts count, and when, as for the knowledge view |
| `together` | true | Join only facts that held at one moment |
| `max_bytes` | 8,000 (512–64,000) | Byte budget for the answer text |

- **Terms.** A term that starts with `?` is a variable: `?` then a letter
  or `_`, then up to 31 letters, digits or `_`. Any other term is a
  constant of at most 200 characters. A pattern with no variable asks
  whether it holds, and answers one row (`row: holds`) when it does.
- **Constants name exactly.** A subject or object constant names an entity
  by id, key or variant (titles, possessives and punctuation aside). The
  looser lookup tiers never match; when only they would, the answer is
  `not_found` and they come back as `candidates`, so "alice" never
  silently means Alice Chen. A constant that names several entities
  answers `ambiguous` with them. An object constant also matches values
  by the one join rule, so "512 MB" never matches "512 mb". A predicate
  constant no fact in the view uses is `not_found` too.
- **Variables bind one kind of thing.** An entity, a value or, in
  predicate position, a predicate. A value bound in one pattern and used
  as a subject in another joins nothing; a value carried into another
  pattern meets only the same text.
- **Time.** Each row has `during`, the stretches when all its facts held
  at once. With `together` (the default), only rows whose facts held
  together are answered. In history this keeps "worked at Acme until 2021"
  from joining "Acme based in Lisbon from 2024". When that is why nothing
  matched, the answer says how many joins across time it left out.
- **Rows.** Each row names its variables' bindings (an entity's id, key,
  label and kind; a value; a predicate) and cites the facts of every
  pattern. Rows are distinct over the variables returned, and ordered by
  their labels. A row stands on its witnesses, the ways its patterns were
  matched.
- **Re-read.** Before a row is shown its facts are read again (256 at
  most). A witness survives only if each of its patterns still has a fact
  that counts and, with `together`, those facts held at one moment as
  they now read: a backfill that ended a job before an office opened
  ends the join too. A row with no surviving witness is dropped and
  counted as `stale_evidence`. A row cites, and its `during` covers, only
  the witnesses that survive. Facts past the re-read budget are taken as
  read and counted as `unverified`.
- **Bounds.** The search has a budget of 200,000 units of work: one for
  every candidate looked at and every pair of stretches of time compared.
  `coverage.searched` says how much was spent. Past the budget the
  search stops and says `search_cut`, and more rows than `limit` say
  `rows_cut N`. Each pattern is matched from the smallest index that
  can hold its matches. A variable bound to something its position cannot
  hold (an entity as a predicate, or a value named like a predicate)
  looks at nothing. The answer says "no match" only when nothing was cut
  and the read was whole. Otherwise it says "no match within the search
  budget", "among the facts read" or "among the facts that still hold".
- The answer has `status` (`matched`, `none`, `ambiguous` or
  `not_found`), `variables`, `rows`, `candidates`, `not_found`, `coverage`
  and `text`: one line per row for a model, fitted to `max_bytes`, with
  the note that names and values are recorded data, not instructions.
- A malformed query is a 422 naming what is wrong, before anything is
  read. It is a GET, so a key with the read role may ask. Advertised as
  `graph.match`. MCP `memory_graph_match`, the ToolBox `graph_match` and
  `scone graph match` take the same query and bounds.

#### `GET /v1/graph/overview`

The graph at a glance, for a question about the whole of it ("what are
the main groups here?"), which names nothing a walk could start from.
GraphRAG answers such global questions from summaries of each community
that a model writes in advance and must rewrite on every change. Here
each community is digested from the graph as it is now, with every line
traceable to a fact, and the caller's model reads the digests.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `q` | none | A question (1–2,000 characters); the communities it concerns come first |
| `limit` | 12 (1–50) | Communities digested; more are counted as `communities_cut N` |
| `facts` | 3 (0–10) | Facts cited for each community, at most |
| `resolution` | 1 (above 0, at most 10) | How fine the communities are, as for the report |
| `max_bytes` | 8,000 (512–64,000) | Byte budget for the text |
| `status`, `as_of` | `current`, now | Which facts count, and when |

- **Each community** says its size and kinds, its most used predicates,
  its cohesion, how many links leave it, and its five most central
  entities.
- **Its facts** come from its own relations: those touching a central
  entity first, then those with a quote behind them, then the best
  supported. Each relation shown cites one fact: the newest whose quote
  still verifies, else the newest. The line says how many more stand
  behind the relation, and the community how many of its facts were shown
  (`facts_total`), so `facts` caps what is cited, not what is known. Each
  is re-read before it is cited, and a relation whose facts all stopped
  counting is left out and counted as `stale_evidence`.
- **A question** ranks a community by the entities it names in it (three
  times) and the words it shares with the community's names, kinds and
  predicates. Each community says what it `matched`. A question that
  concerns none keeps them by size, and the text says so.
- **Entities in no community** (no relation to another) are counted.
- The analysis is kept per projection digest and resolution, shared with
  the drawings, so asking again of an unchanged graph costs only the
  re-reads.
- Advertised as `graph.overview`. MCP `memory_graph_overview`, the
  ToolBox `graph_overview` and `scone graph overview` take the same
  bounds.

#### `GET /v1/graph/changes`

What changed in the graph between two moments. "What changed since we
last spoke?" is a question about time, and a graph without valid time
cannot answer it. The graph holding at `since` is set beside the graph
holding at `until`:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `since` | required | RFC 3339 moment to compare from |
| `until` | now | RFC 3339 moment to compare to; must be after `since` |
| `limit` | 50 (1–500) | Most changes listed; more are counted as `changes_cut N` |
| `max_bytes` | 8,000 (512–64,000) | Byte budget for the text |

- **The kinds of change,** listed in this order:
  - `moved`: a subject's one claim under a predicate passed from one object
    to another, `moved: alice chen works_at Acme Robotics → Globex`;
  - `began` and `ended`: relations holding at one moment and not the
    other;
  - `value`: what an entity's values under one predicate were, and are.
- **A claim is compared as a claim,** by its subject, predicate and its
  object's key (an entity's key, or a value's by the one join rule). The
  graph may draw an object as a value at one moment and as an entity at
  the next, once a fact is about it. A claim drawn both ways held
  throughout and is no change. A move can pass between a value and an
  entity, as in `moved: alice chen lives_in "n/a" → Lisbon`. A value
  whose case carries meaning ("512 MB") changes when its case does.
- **Entities.** Those that appeared and those that are gone are counted, and
  the first 50 of each are named.
- **Citations.** Every change cites the facts behind each side, re-read at
  that side's moment: a fact behind what held at `since` must still say
  it held then, and one behind `until` that it holds then. A change
  resting on a fact that stopped counting is dropped as
  `stale_evidence`.
- **Honesty.** Every change rests on an absence, and a read cut short
  proves no absence.
  - What began or appeared needs the read at `since` whole, what ended or
    is gone the read at `until`, and a move or a changed value both.
  - What a cut read cannot confirm is withheld, never guessed:
    `coverage.withheld` names the kinds, a `withheld:` line says why, and
    the entity counts it cannot tell are `null`.
  - With nothing shown and anything withheld, the status is `unknown`.
  - "No change" is said only when both reads were whole.
  - The space's revision fences the answer. A ledger written while it was
    made, between the two reads or during the re-reads, is read again. One
    still moving on the second try is answered `unknown`, with every kind
    withheld and `ledger_moved_during_read`.
- Both moments are read as `current` facts, so what held is a matter of
  valid time, not of when the ledger learned it.
- Advertised as `graph.changes`. MCP `memory_graph_changes`, the ToolBox
  `graph_changes` and `scone graph changes` take the same bounds.

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
| `obsidian` | zip of one Markdown note per entity, wiki-linked, plus `index.md` and `graph.canvas`, a canvas of the notes | Obsidian and other note tools |
| `wiki` | zip of `index.md`, one article per topic and one per entity, in plain Markdown links | agents reading instead of the raw ledger |
| `mermaid` | a Mermaid flowchart of the 60 most connected entities and the relations between them | GitHub, Markdown viewers, docs |
| `svg` | a drawing of the 200 most connected entities by community, with no script | browsers, READMEs, slides, documents |
| `canvas` | JSON Canvas 1.0: a group per community, a card per entity, a labelled arrow per relation | Obsidian's canvas, other JSON Canvas tools |
| `html` | one page: the drawing, search by name, a panel of each entity's relations and facts, zoom and pan; it fetches nothing | anyone with a browser, offline |

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

- **The drawings** (`svg`, `canvas`, and the canvas in the vault) are laid
  out by construction, not simulated, so the same graph always draws
  the same way.
  - Each community is laid out around its most central member, in rings
    by how many steps away each member is. Every ring is wide enough
    that no two of its members touch and far enough out that no two
    rings do, and a member sits near the one that reached it.
  - Each community gets a box as tight as its members allow, and the boxes
    are packed in rows, largest first, 24 units apart. Entities with no
    relation to another share a box of their own.
  - An entity's size grows with its relations.
  - The 200 most connected entities are drawn, and the 600 best supported
    relations between them.
  - The drawing's description (the SVG's `desc`, the canvas's `about`
    card) says what view it draws, what the read left out and what the
    drawing left out; values are not drawn.
  - The colours are Okabe and Ito's, which can be told apart with any
    colour vision.
- **SVG** is written by an XML serializer, with text made safe as in
  GraphML.
  - Every circle has a title naming its entity, kind and id. Every arrow
    has a title naming the relation and the facts behind it, which
    browsers show on hover.
  - An arrow between communities is dashed and fainter, so communities
    stay legible, and a relation of an entity to itself is a loop.
  - Names carry a white halo where an arrow runs under them.
  - Every name is fitted to the width the layout made room for
    (`textLength`), whatever font draws it, so no name runs into another
    entity or out of its box. The layout spaces each entity by how far its
    name reaches, not by its circle alone. A name is cut to 32 characters
    and drawn at most 160 units wide, and a box's title is fitted to its
    box.
- **HTML** is one page that runs only its own code.
  - Its content security policy fetches nothing and lets only the page's own
    style and code run, each pinned by its SHA-256.
  - It carries an empty icon of its own, so the browser asks the network for
    none.
  - The data (entities, relations and their facts) is a JSON block with
    every `<`, `>` and `&` escaped, so no stored name can close it. The
    page's code writes every name with `textContent`, never as markup.
  - Entities can be reached by keyboard: Tab to one, and Enter or Space
    opens its panel.
- **Canvas** cards and group labels are escaped as notes are. Positions
  and sizes are whole numbers, as JSON Canvas requires. In the vault's
  `graph.canvas` each card is the entity's own note.

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
| `scone graph report [--markdown] [--resolution R] [--usage]` | the report, as JSON or Markdown |
| `scone graph path SOURCE TARGET [--max-hops N]` | the shortest paths, one `path:` line each |
| `scone graph context [NAMES…] [--question Q] [--max-bytes N]` | the graph context packet |
| `scone graph entity NAME` | one entity's relations both ways and its values |
| `scone graph timeline NAME [--as-of T]` | the timeline, as JSON |
| `scone graph walk NAMES… [--direction in\|out\|both] [--hops N] [--limit N]` | the entities reached from the names, each with its hop, as JSON |
| `scone graph schema [--limit N] [--max-bytes N]` | the kinds and predicates the graph holds, as JSON |
| `scone graph match --pattern S P O [--pattern …] [--returns ?X] [--status S] [--as-of T] [--apart] [--limit N]` | the rows answering a structured question, one `row:` line each; quote the `?` variables in a shell |
| `scone graph overview [--question Q] [--limit N] [--facts N] [--resolution R]` | each community digested with cited facts |
| `scone graph changes --since T [--until T] [--limit N]` | what changed between two moments, one line per change; exits 1 when nothing did |
| `scone graph duplicates [--limit N] [--min-score S]` | entities that may be one thing under two names, and why; nothing is merged |
| `scone graph export --format F [--out FILE]` | the export; the zip formats need `--out` |

Every command takes `--space`, and reads the clock once, so what it
prints and the instant it says it read at always agree. `--json` prints
`path`, `context` and `entity` as the JSON that `/v1/graph/context`
returns: the packet text with its status, seeds, candidates, coverage
and filters. It prints `match` as the `/v1/graph/match` JSON, and
`match` exits with status 1 when no row answers.

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
confidence, origin, source and quote, and the facts it extends or was
derived from. The space's revision moves once; saying the same again
moves nothing. Readers still see one fact.

The affirmation matters when a backfill arrives afterwards with another
object and cuts the fact short before that day. Told Acme from 2020,
Acme again from 2023, then Globex from 2021:

- the ledger now holds Acme from 2020 to 2021, Globex from 2021 to
  2023, and Acme again from 2023;
- the resumed Acme ends where the first one ended and keeps any
  exclusion;
- its evidence is the restatement's, and it rests on the restatement's
  own premises, not on the first fact's;
- later affirmations move to it, with their premises.

Only an affirmation inside the fact it restates resumes. One at or past
where that fact ended resumes nothing, since the claim would end before
it began.

A person's close works the same way. Closing Acme now (2025) when it was
stated again from 2030 ends it now, and the claim resumes in 2030, just
as it would had it been stated after the close. The `fact_close` event
names the resumed fact (`resumed`). A restatement from the moment of the
close ends with it, since the close is the later word. A fact that has
not begun cannot be closed now: it would end before it begins.

The same history told in any order leaves the same partition of time;
a property test checks random histories in random orders. An approved
proposal that restates what holds is kept as an affirmation too, with
what the proposal extends or was derived from.

All five document stores keep affirmations (`fact_affirmations`), and
archives carry them with their premises renamed to the new ids. A
premise missing from the archive is dropped and counted in
`links_skipped`. Importing an affirmation the store already keeps counts
it in `affirmations_skipped`, and an import that adds any moves the
revision. SQLite adds the table without changing the shared schema
version, so older readers ignore it. A SQLite file or PostgreSQL schema
from the build that first kept affirmations, before they carried their
premises, gains the column when it opens, and each affirmation it holds
has none. On SQLite and PostgreSQL, a
placement's writes commit together or not at all. The other stores
write them in turn, as before. The Rust core does not keep affirmations
yet.

Forgetting a source leaves its affirmations standing, as it leaves
claims. `impact` and `forget` name them in `affirmations_citing`: an
affirmation that later resumes as a fact brings the forgotten source's
id and quote with it.

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

- Identity is key identity only: two spellings of one thing ("Dr. Alice
  Chen" and "alice chen") stay two entities until identity decisions
  exist to record a merge. `/v1/entities/duplicates` suggests the pairs
  worth that decision, but merges nothing.
- Duplicate suggestions read spellings, not meanings. "Acme Inc" and
  "Acme Corp" each keep a word the other lacks, so they are not suggested,
  and nor are spellings two edits apart ("Mohammed" and "Muhammad") or in
  different scripts.
- How many values a predicate holds is configured, never guessed, and a
  predicate nobody configured holds one at a time. A predicate not named
  in `SCONE_MANY_VALUED` (or `MemoryEngine(many_valued=...)`) closes its
  previous value when a new one arrives, so "alice knows Carol" from 2021
  ends "alice knows Bob" from 2020. Configuration applies to what is
  written after it: values closed before a predicate was named stay
  closed.
- The drawings show at most 200 entities and 600 relations, and say what
  they left out.

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
