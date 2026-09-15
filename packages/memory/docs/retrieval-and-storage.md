# Recall semantics, storage and recovery

Interrupted whole-space erasure uses [durable deletion recovery](space-deletion-recovery.md).

[Package overview](../README.md) · [Architecture](../ARCHITECTURE.md)

Examples below run from `packages/memory/` unless a section names another working directory.

## What recall returns

Items carry `score` (rank within this query; the top item is always 1.0)
and `similarity` (cosine from the vector lane, when that lane saw the
chunk). Two lanes, vector and lexical, are fused by reciprocal rank with a
small recency term, capped at two chunks per episode. The recency term is
`SCONE_RECENCY_WEIGHT` at age zero (default 0.005, small against a fused rank
score, so it breaks near-ties toward newer memory and nothing else), halved
every `SCONE_RECENCY_HALF_LIFE_DAYS` (default 30); a memory of a support queue
can weight it up, and zero turns it off. The same knobs are
`MemoryEngine(recency_weight=…, recency_half_life_days=…)`. A lane that fails is
named in `degraded` and the other lane still answers. `context_reduction`
is the share of the space's bytes that were left behind.

A recall narrowed by `conditions`, `kind`, `source_prefix`, `since` or
`until` carries `narrowing` (`null` when nothing narrowed): per lane,
whether the request was applied `in_store` (every row the lane holds was
eligible) or `postfiltered` (the lane returned its best window and the
request then removed what did not fit), how deep each lane looked
(`text_window`, `vector_window`) and returned, how many candidates the
filter removed (`postfiltered_out`), and `window_exhausted` -- the filter
removed candidates while a post-filtered lane's window was full, so a
memory that fits may lie deeper than the recall looked. An empty answer
with that flag set is not "there is none". The in-process vector indexes
evaluate conditions themselves; kind, source and date bounds live on the
episode, which no vector row carries, so for those the vector lane is
post-filtered from a window twenty-five times wider than an unnarrowed
recall's, the same allowance the text lane has when its store cannot
narrow. The CLI prints a note when the flag is set; the recall evidence
event carries the same fields under `narrow`.

`top_similarity` is the best cosine the vector lane saw for the query.
With `SCONE_SIMILARITY_FLOOR` set (a cosine, e.g. `0.45`), a recall whose
best hit falls below it, or that finds nothing, carries
`low_confidence: true` so the reader can decline to answer from weak
evidence; without a floor the field is `null` and nothing is judged. The
floor has no default because the right value depends on the embedder:
`scone-memory bench` prints, for each candidate floor, how many
no-evidence questions it would catch and how many answerable ones it
would wrongly withhold, and that sweep is where a floor comes from.

`lanes=text` or `lanes=vector` (CLI `--lanes`) runs one lane alone; both
run by default. A lane not asked for is not run at all: a text-only
recall makes no embedding call, which suits an exact code or identifier
no embedding knows, and comparing the two lanes on one question. It is
not reported as degraded, because nothing failed. The answer's `lanes`
names the lanes that answered, so with both asked and one failed it
names the other, and the recall event records the same list. A vector
lane that did not run judges no confidence: `top_similarity` and
`low_confidence` are `null`. `lanes` names only these two; the entity
lane is asked for with `graph_boost` and the context lane with
`SCONE_CONTEXT_LANE`, whatever `lanes` says, and whether either ran shows
on the items (`lanes.entity`, `lanes.context`) and in `degraded`, not in
`lanes`. A note in `degraded` about a lane that still answered, such as
a text lane whose lexical index is behind, does not take it out of
`lanes`. When the only lane asked for fails, recall
fails rather than returning an empty answer that reads as nothing found.
Advertised as `recall.lanes`.

`require` and `exclude` (repeat each for more; CLI `--require`,
`--exclude`) are phrases a returned passage must all hold, or must hold
none of. A phrase matches as whole words in order, whatever the case and
the punctuation between them: `slew ring` matches `SLEW-RING`, and `art`
does not match `party`. In scripts written without spaces a phrase
matches inside a run. The reference's keyword filter drops nodes after
retrieval, so a filter that drops three of five returns two and says
nothing. Here the phrases are checked across every fused candidate before
the per-episode cap and the limit, so a passage ranked below the limit
that holds the phrase takes the place of one that does not. `phrases` in
the answer (and the recall event) says how many candidates were checked,
how many each rule dropped, and `short: true` when fewer passages came
back than the limit after the phrases dropped some while a lane filled
its candidate window, because passages beyond that window were never
checked. Each lane is judged against its own window: a narrowed recall
that post-filters the vector lane gives it a deeper one than the text
lane's. When no lane filled its window every passage was a candidate,
and a short answer is only a small space. Facts are not filtered.
A phrase both required and excluded, one with no word to match, more than
20 phrases or one over 200 characters is refused. Advertised as
`recall.phrases`.

`diversity`, a weight from 0 to 1 (CLI `--diversity`), keeps near-copies
of one passage from taking several places. Fusion ranks by relevance
alone, so three restatements of a note can fill three of five places.
With `diversity` the places are filled one at a time, each by the
candidate whose relevance (its fused score over the best one's) times
`1 - weight`, less `weight` times its greatest cosine to a passage
already placed, is highest: maximal marginal relevance, over the fused
candidates rather than one vector lane. `0` keeps the relevance order.
The likeness is read from the index's stored vectors where the index can
give them back (in memory and SQLite), and otherwise every candidate is
embedded once so all likenesses are on one scale; `diversity.vectors`
says which. `diversity.replaced` counts the first `limit` places, before
the per-episode cap, that relevance alone would have filled differently.
Only the first 200 candidates are compared and only twice the limit's
places are filled this way; the rest keep relevance order and are
counted. It runs after any phrases and is refused beside an active
reranker, which would set the order again. Unmeasured. Advertised as
`recall.diversity`.

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

### What the ledger says about a passage

A passage is evidence for a claim, and the ledger may have retired that
claim since: replaced it (`superseded_by`), or closed it without a
replacement. Fusion cannot see this. `demote_restated` reorders passages
that **lexically** restate one another, and for a replacement that
differs at the end it is perfect; a replacement that differs
mid-sentence, or is worded afresh, shares nothing it can see, and the
retired passage led in every such case measured. So recall reads the
ledger once more, about the passages it is returning.

For each returned episode it reads the claims that episode stated -- one
indexed read per episode, proportional to the result and never to the
ledger -- and asks of each: had it ended by the boundary asked about
(`as_of`, else now)? A passage with such a claim carries
`superseded: true`. If the claim's successor was stated by another
returned episode, the retired passage is placed after that episode's
passages; the passages involved fill their own positions, and every
other passage keeps its place. The pairing is by id: a retired claim
follows the claim that replaced it, never a coexisting value of a
many-valued predicate, and never a replacement the reader did not
receive. A passage may replace one claim and be replaced under another,
and is ordered accordingly; where two passages each replace a claim the
other made, the reader's order stands for that pair.

The ledger is honoured as written. A claim a person **excluded** is
suppressed from recall: it neither marks nor moves anything, and an
excluded successor cannot pull its passage ahead -- though the interval
it left behind stands, so the claim it replaced is still retired. A
claim that had not begun by the boundary is not retired; ask about March
and March's statement leads, unmarked. Nothing is ever dropped, since
the reader may be asking about the past.

What could not be read is said in `degraded`: a store that cannot read
facts by episode (`supersession: store cannot read facts by episode; …`),
and an episode that stated more than the read cap of 2000 claims, of
which only the first 2000 are seen (`supersession: episode N stated more
than 2000 facts; …`). `MemoryEngine(demote_superseded=False)` turns the
rule off, and then neither the mark nor the reordering is applied.

## A profile: who the space is about

A profile answers "who is this about" without being asked a question:
the claims that hold, then the space's recent activity. `GET /v1/profile`
(capability `profile.read`), `scone profile` and the ToolBox tool
`read_profile` give it.

- **What leads it** is what the space says most often, then most
  recently. A claim restated three times leads one stated once, because
  restating a claim is the space saying it matters; ties go to the newer.
  Each item keeps the evidence it rests on, so a profile item can be
  traced to the episode and quote behind it.
- **What counts** is configured, never guessed: `SCONE_PROFILE_PREDICATES`
  names the only predicates a profile is made of, and
  `SCONE_PROFILE_WITHOUT` names the ones it never shows. Unset, every
  predicate counts. Names are read as the ledger reads predicates.
- **Bounded and said.** At most 20,000 facts are read, and restatements
  are counted for the newest 200 of what the policy keeps; `coverage`
  says how many facts were read, what the read left out, the policy in
  force, how many candidates were considered, and how often each shown
  claim was restated.
- **One revision's profile.** A profile is three reads — the claims, how
  often each was said again, what happened lately — so a ledger that moves
  between them could answer with a claim from before a write beside a
  count from after it. The revision is taken before and after, a profile
  made across a write is read again, and a ledger that will not hold still
  for two attempts is answered with `ledger_moved_during_read` among the
  coverage reasons rather than passed off as settled. `coverage.revision`
  says which revision the answer is of.
- Closed, excluded and proposed claims are never profiled, as before.

## What people said about a passage

```bash
scone lessons --window-days 90 --half-life-days 30 --min-corroboration 2
scone recall "when was the crane survey booked" --lessons
```

A person can mark a returned passage useful or not (`POST /v1/feedback`),
and that judgement is kept as an event. `lessons` reads those events back.
For each passage, the latest judgement of each recall counts. Each one
weighs 1, positive when useful and negative when not, and the weight halves
every `half_life_days`. Each passage gets a state:

- `preferred` needs at least `min_corroboration` useful judgements and none against, because one person's word is not a preference;
- `dead_end` means judged and never useful;
- `contested` means judged both ways;
- `tentative` means useful but not yet corroborated.

Each lesson says whether its passage can still be read (`present` or `gone`).
The read covers `window_days` and at most `max_events` judgements; past that
the oldest are left out and `events_cut` says so. Judgements are counted
per space, since no per-person identity is recorded yet.

`recall(lessons=True)` (`lessons=true` on `/v1/recall`, `--lessons`) puts
each passage's lesson beside it, and `GET /v1/lessons` lists them all.
Lessons never change the order: nothing has measured that ranking by them
answers better, so they are information beside the score, not part of it.
A recall asked for lessons also carries `lessons_read`: the window,
half-life and corroboration used, and how many judgements were read and
whether that read was cut. A recall not asked answers exactly as before,
with no `lessons` or `lessons_read` field. `--lessons` is refused with
`--merge`: a merged passage joins chunks that were judged separately, and
one lesson cannot stand for them.

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

## One question, and the machinery that suits it

```bash
scone answer "How long ago did I move to Lisbon?"
# route: temporal — it asks about dates and the ledger grounds it (computed)
# answer: 121 days
```

Growing a computed temporal answer, an entity graph and two retrieval
lanes left callers with a problem they did not have when there was only
recall: knowing which to ask. The leading frameworks answer that with a
model writing a plan, which is expensive and inscrutable when it is
wrong. Here the rule is written down, tried in order, and **every answer
says which way it went and why**:

1. **A question about dates** goes to the one that computes — but only if
   the ledger can actually ground it. "Looks temporal" is not enough: a
   question whose events are not there is better served by the passages,
   and the answer says *"it reads as a question about dates, but the
   ledger could not ground it"* rather than leaving a caller unable to
   tell a gap in the ledger from a gap in the rule.
2. **A question about something the graph knows by name** is answered
   from the claims, naming the entities it recognised.
3. **Anything else** is an ordinary search.

`route=` insists on one instead, and the answer says it was asked for
rather than chosen: a rule that cannot be overridden is a rule somebody
will work around, and then the framework learns nothing from being
wrong. `GET /v1/answer`, `scone answer`. Nothing here calls a model
unless the fourth route is asked for by name:

4. **`route=synthesize`**, which the rule never chooses. A broad question
   ("what is known about the launch?") is not answered by five passages
   and a score; this reads up to `limit` passages (fifty at most) and has
   a model write a few sentences about them, each naming the passage it
   came from and a quote from that passage **that the framework found
   there**. A sentence without such a quote is not shown, and is counted.
   Passages are packed into rounds by bytes, one model call each, so an
   early passage cannot shape what a later one is allowed to say and a
   model that fails costs only its round; when more than one round left
   notes, one more call may fold them into a summary whose sentences cite
   notes by id. `detail` counts what was read, unread, refused as too
   large for a round, and dropped as unquoted or uncited, says whether the
   sentences were cut, and carries `verified_accuracy: false` on every
   record because only the quotes were checked, not the sentences. It
   needs a model (`SCONE_CHAT_URL` and `SCONE_CHAT_MODEL`); without one
   the route is refused. `scone answer --route synthesize --limit 30`.

An ordinary answer shows each passage to its first 200 characters, and
says so: `shown` carries `per_item_chars`, `items_cut` and
`chars_omitted`, and when anything was cut the text ends with one line
naming how many of how many were shortened and that `detail` holds them
whole. `scone answer --whole`, `GET /v1/answer?whole=true` and
`answer_question(..., max_item_chars=0)` show them whole instead.

### What a question says about where to look

```bash
scone recall "what did we decide about the launch in last week's notes?" --infer
curl "…/v1/recall?q=the+retry+policy+under+docs/agents/&infer=true"
```

A question often carries its own scope: a date ("last week", "in March"), a
kind of memory ("in my notes", "from the files", "in our conversations"),
tags ("tagged urgent", "#finance"), a place ("under docs/agents/", "in
README.md"). The reference framework has a model infer metadata filters
from the question; here the words that name a scope are read by rule, each
recorded with the words that said it, and — with `--infer` or
`infer=true` — searched with where no filter was given by hand: the
question's date becomes `since`/`until` (the readings `dates.py` already
makes for temporal questions), its kind `kind`, its tags `tags`, its place
`source_prefix`. A filter the caller set is never replaced, and a caller's
window, whichever end they set, is theirs whole; a tag filter is "all of",
so a question's tag is never added to tags the caller set. The answer
carries `inferred`: every reading, which were applied, and which were
`withheld` with the reason, so a wrong reading is visible rather than a
silent narrowing. A tag is the one reading whose mistake empties an answer
outright (`#include` has the shape of a tag), so a tag the space holds no
memory under is withheld and said, and one it holds is applied in its
stored spelling; a hashtag begins a word, so the `#install` of a URL is not
one. The words stay in the question the
lanes search, since a word that names a scope can still name what the
passage says. A kind is read only after a word that places the question in
it ("in my notes"), a tag begins with a letter (`#12` is an issue, not a
tag), and nothing is read without `--infer`.

### Summary trees for long documents

```bash
scone summarize 42 --fan-in 6          # store the tree as notes beside the document
scone summarize 42 --dry-run --json    # write it and show it, store nothing
```

A long document answers a broad question badly in chunks: "what does this
report conclude?" is spread over forty passages, and the five that score
best are five fragments. The leading frameworks build a tree of summaries
at ingestion (RAPTOR: cluster, summarize, repeat) and index the summaries
beside the leaves, so a broad question finds a summary and a narrow one
finds a chunk; the summaries are the model's word. `summarize` builds that
tree with the synthesizer above, so every sentence rests on a quote the
framework found in the level below: level zero is the document's chunks in
their order, each level above groups `--fan-in` adjacent nodes and writes
one node from them, and a citation resolves downward to the chunk quotes it
rests on. Adjacency is the grouping, not a clustering — a document's own
order is a structure nobody has to guess at. A group the model wrote
nothing about leaves no node and is counted; a lone remainder is carried
up, not summarized from itself; a level nothing was written at stops the
tree and says so; `--max-levels` (five at most) can leave the top level
unjoined, and the record says so.

The nodes are stored as notes the framework wrote, with the document's
source and `#summary/<level>/<index>` as theirs (the index is the group's
position at its level, so a group the model left empty leaves a gap rather
than renaming what follows) and metadata saying what they are:
`summary_of` (the episode), `summary_level`, `summary_index`,
`summary_fan_in`, `summary_chunks` with `summary_first_chunk` and
`summary_last_chunk`, `summary_model`, `summary_content_hash` of the
document at the time, and `summary_detail`, the attachment the note links
that holds the full account — every chunk under it, what it was written
from, every sentence with its citations and quotes — since that account is
longer than a metadata value may be. The ordinary lanes then retrieve a
summary beside the chunks and a reader can see what a passage is. `GET
/v1/episodes/{id}/summaries` lists a document's summaries top level first;
`POST /v1/episodes/{id}/summaries` builds and stores the tree with the
synthesis model, and without one it is refused. Building a tree again for
the same episode replaces the tree that was there, whatever its shape (a
different `fan_in`, a group the model left empty this time), so a document
never holds two trees. A document that changes is a new episode, with no
summaries until its own are built; the old episode's tree stays with it.
Forgetting a document does not yet forget its summaries; they carry
`summary_of`, so they can be found and forgotten by it. Not yet measured:
coverage on broad questions with and without stored summaries waits for a
run with the local model.

### Checking the rule instead of asserting it

A routing rule written down is only better than one a model invents if
somebody checks it, so the check is part of the framework rather than a
scratch script:

```bash
scone bench-route bench-data/temporal-40.json
# routing: 40 question(s) of …: temporal 12, graph 0, recall 28; of 12 computed,
# 9 agree with the file. This does not say whether a question sent to search
# would have been answered better another way: the file has one answer, not one
# per route.
```

Each question gets its own memory, so the configured store is neither
read nor written, and the computed answers are scored against the file's
own answers — read from the file, because the bench loader keeps only the
fields retrieval is scored on.

The temporal route answers two ways and the report keeps them apart. It
either **computes** an answer or hands back the day's **passages**, and
only the first has arithmetic to compare with the file; a recalled answer
in the `computed` denominator made the ratio report the computation as
wrong when nothing had been computed. Of the computed ones, `agree`,
`disagree` and "a shape this cannot judge" are three separate counts,
because the scorer returns *cannot tell* for answer shapes it does not
handle and adding those to the wrong ones tells a reader neither.

What this **cannot** say is the more interesting half: whether a question
the rule sent to search would have been answered better by computing it.
That needs a known answer for every question under every route, which
these files do not have. The report says so in its own output rather than
reading as though the rule had been vindicated.

## A question that asks two things, searched as two

One query over "What did I decide about billing, and who was at the
meeting?" returns one blend of passages, and the half whose words are
commoner in the corpus tends to take every slot. The leading frameworks
split such a question with a model writing sub-questions — a paid call
per question, and nothing a person can read when it comes out wrong.
Here the rule is written down and every decomposition says what it did:

```bash
scone recall "What did I decide about billing, and who was at the meeting?" --parts --limit 2
# it asks 2 parts, searched on their own and merged so each has its turn; no
# similarity floor is configured, so a part returning passages is not evidence
# that part was answered
# [What did I decide about billing] We reverted the billing change after ...
# [who was at the meeting?] At the Thursday meeting were Priya, Tomas ...
```

`GET /v1/recall/parts` (capability `recall.parts`). Nothing is
paraphrased: a part is a verbatim span of the question and carries its own
offsets, so a receipt can quote exactly what was searched.

It takes the same narrowing as an ordinary search — `as_of`, `tags`,
`where`, `conditions`, `kind`, `source_prefix`, `since`, `until` — and the
response **echoes the space it authenticated for and the filters it
actually applied**. A filter a caller passed and the server quietly
dropped is worse than one it refused: the page shows an answer that looks
narrowed and is not.

`history` is **refused rather than ignored**. It returns the closed chain
behind the facts one query matched, and merged across a question's parts
that has no defined meaning; inventing one silently would be the same
fault as dropping a filter. Ask `/v1/recall` with `history` for the whole
question instead.

### What it measured, which is nothing

```bash
scone bench-parts bench-data/longmemeval_s.json --k 10
# parts: 13 of 500 question(s) ... split at k=10. On those: any-evidence 12
# whole vs 12 parted; all-evidence 12 whole vs 12 parted (+0 question(s)).
```

**Splitting changed no retrieval on the one dataset available**, at k=5
and k=10 alike. So it is an opt-in flag and a separate route, not the
default search: it costs a search per part and buys no measured gain.
This is the third retrieval idea measured and left off by default this
week, and all three point the same way — with the hash embedder the
lexical lane carries retrieval.

What the number does **not** say is that the idea is worthless. The rule
split 13 of 500 questions, and LongMemEval's questions are single-focus
by construction: the benchmark cannot test a two-part question because it
barely contains any. A corpus of genuinely multi-part questions would be
needed to answer it, and arranging one by hand would measure the
arrangement.

What it does buy, measurement aside, is the receipt: which part placed
each passage, which parts found nothing, and — when no floor is
configured — the plain statement that finding passages is **not** evidence
a part was answered. There is deliberately no single confidence number
for a multi-part answer, because the floor was measured per query and one
number over the lot would be a number a reader could mistake for a
judgment about the whole question.

### The rule, and why it is shy

The risk is one-sided. A question wrongly left whole retrieves what it
would have anyway; a question wrongly split is searched as two queries
that mean nothing, and the answer is worse than before. So a `?` always
ends a part, while a `;` or an "and" splits only on evidence:

| The question | What happens | Why |
| --- | --- | --- |
| `How many engineers do I lead now? How many did I lead before?` | splits | two sentences, each its own question |
| `What did I decide about billing, and who was at the meeting?` | splits | comma before "and", and the right side opens with "who" |
| `Where do I work and where does my sister work?` | splits | the right side opens with its own interrogative |
| `How many hours of jogging and yoga did I do last week?` | whole | no comma, no interrogative — "and" is joining a list |
| `How many days passed between the day I cancelled ... and the day I did ...` | whole | "between … and …" *is* the question |
| `... the 'To Adapt or Not to Adapt? Real-Time Adaptation' submission?` | whole | the `?` is inside a quoted title |

### The languages we claim, and what they actually gave

Being on a supported list is a claim, and four of ours were not true. Each
was found by running the extractor over five lines of the language rather
than by reading the list:

| language | what it gave | what it gives now |
| --- | --- | --- |
| Rust | **no inheritance at all**, and `impl Shelf` counted as a second definition of Shelf | `impl Store for Shelf` → `Shelf mixes_in Store`; an inherent `impl` is neither an edge nor a definition |
| Go | **no definition for `type Shelf struct`** — Go types were invisible | `type … struct` and `type … interface` define |
| Kotlin | `inherits Base()` — the call kept, so `Base()` and `Base` were two entities | `inherits Base`, and the constructor call is what tells a superclass from an interface |
| Scala | **nothing**: `extends Base(3) with Store` defeated the clause pattern on both the parens and the `with` | both bases, with `with` read as a clause |

A graph holding both `Base()` and `Base` cannot answer a question about
either, and a Rust graph without `impl … for …` is missing the language's
most important relation. Fixing what the list already promised was worth
more than adding a twentieth language: tree-sitter would bring ~40, and
that is a dependency and a grammar per language rather than a patch.

The last three rows of the earlier table were not foreseen. An earlier version required only
an asking word on each side, and on real LongMemEval questions it split
"how many hours of jogging and yoga did I do last week" at the "and",
because "did" satisfied the test; it also split a paper's title at the
question mark inside it. Running the rule over 500 real questions found
all three, which is the argument for a rule you can run over a corpus and
read the output of.

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

### A codebase is already a graph

With `code_graph=True`, remembering a source file also records what the
file says about itself, as ordinary claims:

```
app/planner.py        defines  app/planner.py:plan
app/planner.py:Engine defines  app/planner.py:Engine.recall
app/planner.py        imports  json
app/planner.py:plan   calls    app/planner.py:tidy
```

No model is called and nothing leaves the machine: it is Python's own
parser, so it is exact where the parser is exact and silent everywhere
else — a file that does not parse says nothing rather than guessing.
Every claim is quoted from the line it was read on, cited to the episode
the file was stored as, and marked `extracted` rather than `stated`,
because nobody said it: it was read.

```bash
scone map src/scone_memory --graph
# map: 278 file(s) read, 10049 claim(s), 10 had nothing to say

scone graph match --pattern "?who" calls "retrieval/temporal.py:_spelled"
# row: ?who = retrieval/temporal.py:_ledger [fact 7690; quote verified:
#      "if len(spells) == 1 else f\"answer: in {_spelled(len(spells))} spells\""]
```

`scone map <dir> --watch` keeps mapping: after the pass it reads the tree
again every `--every` seconds (5 by default) and records what changed — a
changed file replaces its memory and, with `--graph`, its claims; a file
that is gone is forgotten (`removed` in the receipt, its claims closed
the way `sync` closes them); a quiet tree costs a read and a hash per file
and writes nothing — until interrupted or `--rounds` passes are done. Each
pass prints its own receipt under a timestamp (under `--json`, a
`{"pass": n, "at": …}` line before each receipt, so the output stays a JSON
stream `jq` can read), so what the graph reflects at any moment is what the
last pass said. A file that did not change is not recorded again, but what
it declares still counts: a call in a changed file binds to a declaration
in an unchanged one. This is the reference graph
tool's watch mode without a file-system event library: polling, bounded,
and honest about each pass.

A repository has a better signal than the clock. `scone-memory hooks
install` (from anywhere inside the repository, or `--root DIR`) writes a
runner script into the repository's hooks directory, wherever
`core.hooksPath` puts it, and a guarded block into `post-commit`,
`post-checkout` and `post-merge` that calls it; each call maps the tree
with `--graph` (`--no-graph` to store the files alone) in the background,
so a commit is not made to wait, and appends the JSON receipt to
`scone-map.log` in the repository's git directory (`.git/scone-map.log`;
a linked worktree's own git directory, so two worktrees do not share a
log). A checkout of files (the branch flag 0) maps nothing. `hooks
status` says which hooks carry the block (`installed`, `absent`, or
`other` for a hook of the person's own), the interpreter the runner
names and whether it still exists, the settings it carries, and how the
log ends: the last receipt when the last run finished, or the log's last
line when it did not, so a failed run is never hidden behind the success
before it. `hooks uninstall` takes the blocks out and leaves whatever
else the hook files held, removing a file only when nothing but its
shebang is left. A hook file that exists is appended to, never replaced;
a second install replaces its own block where it stands, so what the
person put after it stays after it; a block with a start and no end is
refused rather than guessed at, and a marker is a whole line, so a
comment that mentions one is not a block. The interpreter's path is
written in full, so a commit from an editor with no shell environment
still finds it, and `PYTHONPATH` goes with it when the install ran with
one; only the settings that name a store kind or a local path
(`SCONE_DOCUMENTS`, `SCONE_VECTORS`, `SCONE_EVENTS`, `SCONE_SQLITE_PATH`)
are written into the runner, and a connection URL or a key never is: the
runner reads those from the environment it runs in, or from a file the
person names with `--env-file`, which the runner sources. `SCONE_HOOK_WAIT=1` makes the runner wait for the map. A
file-system event watch is not built: the hooks cover the moments a
repository's tree moves, and `--watch` covers an editor's saves.

`map` walks a directory, remembers every source file under the path it
was read from, and with `--graph` records what each says. A map is of the
tree as it is now: a file is held under the identity `sync` uses for it,
so mapping again after an edit updates the file's memory rather than
adding a second, the receipt counts it as `updated`, and with `--graph`
the claims the new version no longer makes are closed, naming the file
(`claims_closed`). What `map` stored, `sync` recognises as its own, and
the other way round, and both follow a relative import the same way: only
to a file the walk actually read -- the ones walked now and the ones the
marker already holds -- never guessed at, through one resolver
(`code_resolution.file_resolver`) that a batch of files remembered together
also uses among themselves. An answer
carries the line it rests on, **re-read from the file** before it is
shown: a graph of a codebase goes stale the moment somebody edits it, and
a citation that was not checked is the thing least worth trusting.

**Languages other than Python get what can be read and not what would
have to be inferred.** For the brace family the declarations come from
the same scanner that cuts those files into chunks, and imports are read
from the lines that write them (`from "./x"`, a bare `import "x"`,
`require("x")`, a Go import block, a Rust `use`). **No call is claimed**:
resolving a call means knowing what a name refers to, which needs a
parser this does not have, and an edge nobody can check is worse than no
edge. Python gets calls because Python's own parser gives them.

A relative import is followed only to a file the map actually read.
Resolution belongs to the walk, because that is what knows which files
exist; a file on its own cannot tell where its package root is, so on its
own it says nothing about `from .code import x` rather than guessing. What it read
and what it did not is said: files already here, files with nothing to
say, files it could not read, files left unread past the limit, and files
read only as far as the byte budget. A map that quietly skipped half a
repository is worse than no map.

**It will not guess.** A call to something the file cannot see — another
module's function, a method on a value whose type nobody stated — is
left out rather than pointed at a name that might mean anything. A graph
with edges nobody can check is worse than a smaller graph. What it does
resolve: a bare name that is one of the file's own declarations,
`self.method` and `cls.method` inside the class that defines it, and a
call through a class the file declares, `Shelf.keep()` or
`Shelf.Label.print()`, when everything before the last name is a class
declared there. A class brought in by an import is not followed that way:
a file cannot tell an imported class from an imported object, so
`from x import Y` then `Y.m()` stays unbound. Over this package's own
source that added 15 call edges to 14,425.

A function written inside another is its own caller. `defines` names it
`outer.inner`, and its calls are now recorded under that name and only
under it: before, they were made by `path:inner`, an entity nothing
defined, and credited to `outer` too, while `outer`'s own call to
`inner()` went unbound. A bare name is looked up from the innermost
enclosing function outwards, skipping class bodies as Python does, and
`self` inside a class written in a class is that class. Over this
package: 1,072 of 9,837 distinct call edges started at a function nothing
defined, and none do now; 1,619 edges went and 1,287 came, for 9,505.

**Documents are in the graph too.** `map` reads `.md`, `.rst` and
`.txt` files beside the code, and a document's links become claims the
way a file's imports do: a link to a file the walk read (`[text](./x.md)`,
`[[Title]]`, a reStructuredText `:doc:` or `<target>`_, a path in
backticks such as `` `src/pkg/engine.py` ``) is `references` from the
document to that file, one per pair however often the page links it, and
a decision record or standard it names (`ADR-12`, `RFC 7231`) is `cites`,
the same node the code's comments cite (a record does not cite itself).
A link that leads outside the tree is counted, not claimed. A link to a
file the walk did not read is unresolved and `map`'s receipt names it
(`unresolved_links`), never guessed at; a bare name or wikilink title
that two files would answer binds to neither and is named apart
(`ambiguous_links`); a link that says where it is (`./x.md`, `../x.md`)
is followed only there. Links inside fenced code blocks are examples and
are skipped; a badge inside a link leaves the link it wraps. A document
past the line bound, a line past 4,000 characters, or claims past the
cap are not read further, and the result says which bound bit. Because
`references` is one of the predicates `graph affected` follows, "what
rests on this module" answers with its callers **and the pages that
describe it** — the ones that go stale when it changes — and through
them the pages that link those. A tree mapped before this reader gains
its documents' claims as their files change, or on a map with the
documents touched: an unchanged file is not read again.

Because they are claims, everything else already works on them. "What
calls this?" is `graph match` over `calls`; the path from one function to
another is `graph path`; and a vocabulary that says `defines` is the
other side of `defined_in`, or that `imports` carries through, makes the
graph answer more without any of it being written down twice. It is off
unless asked for, because it writes to the ledger and a space's owner
decides what goes in theirs.

### How long a claim held

"How long did Alice work at Acme?" is answered from the claim's own valid
time, and a claim that held twice is two spells, not one long one: the
answer counts the spells and says how many there were, because a claim
made of two stretches with a gap between them did not hold during the
gap. `value` carries `days`, `months`, `years`, `holds`, `spells` and
`periods` (every stretch, half-open); the working names each stretch. A
claim that still holds is counted up to the moment asked and says so.

## Withholding what a caller must not receive

Secrets are scrubbed on the way **in**, but only for the agent feed.
Memory arrives by many other doors — `remember`, `sync`, document import
— and none of them scrub, so a space accumulates whatever was put into
it. A framework whose business is remembering what people said should be
able to withhold on the way out.

```bash
scone recall "write to ana about the deploy" --withhold email,secret
# 2 match(es) of email, secret withheld from this answer; this is a net of
# patterns, not a guarantee -- nothing withheld is **not a finding** that there
# is nothing of these kinds in the text, and the memory still holds whatever it
# held
# 0.81  2026-09-12  #4  Write to [withheld: email] about [withheld: secret].
```

`GET /v1/recall?withhold=email,secret` (capability `recall.withhold`).
Kinds: `email`, `phone`, `ip`, `card`, `secret`, and the name kinds
`person`, `organisation`, `place`.

The name kinds withhold the names the space's graph holds for entities of
that kind, matched whole on word boundaries, without regard to case or
spacing: `Acme Robotics` is withheld in `ACME  Robotics`, not in
`Acmeville`, and a longer name goes before a shorter one inside it, of any
kind (`Paris Hilton` before the place `Paris`). A name in a script written
without spaces needs no word boundary, so `田中太郎` is found in
`田中太郎さんが来た`. The address patterns run before the names, so a name
inside an address never breaks the address. Names
shorter than three characters are too easily a word and are not used; the
report counts them in `names_skipped` and the names it did use per kind in
`names_known`. A name the graph does not hold is not found. Someone never
recorded as an entity, or recorded without a kind, is not withheld, and
the report says so. The graph is read before the search, at the moment the
recall asks about (`as_of`), since a past passage names who counted then; a
capped read of it is said in the report; and a graph still being built
refuses the request rather than withholding nothing. Every kind now also
scans a fact's `closed_reason`, which holds the prose a person gave when
closing it.

Four things it does deliberately:

- **The caller names the kinds.** Withholding something nobody asked to
  withhold damages an answer to protect nothing, and an unknown kind is
  refused rather than ignored.
- **A number is checked, not just matched.** A run of sixteen digits is
  an order reference far more often than a card, so the card kind runs a
  Luhn check. A false positive here costs the reader the answer.
- **The report is not a safety claim.** "Nothing withheld" means the
  patterns matched nothing, *not* that there is nothing to find, and
  every report says so in those words. A caller who reads it the second
  way is worse off than one who was told nothing.
- **Nothing is deleted.** This is what one answer hands back; the memory
  still holds what it held, which is why the field is `withheld` and not
  `removed`. A passage longer than the scan bound has its tail reported
  as unexamined rather than silently passed.

### What it covers, and what it refuses to be asked

The first version of this scanned `text` and nothing else. The same
address came back as the item's `source`, in its `tags`, in a `metadata`
value and as the `object` of a fact — while the report said one match and
nothing unscanned, which reads as "this answer was covered". Scrubbing
the prose and handing the address back in the next field is not
withholding, it is moving it.

Every text-bearing field of an item is scanned (`text`, `source`, `tags`,
`metadata` values) and every one of a fact (`subject`, `predicate`,
`object`, `quote` — the quote is an exact substring of the episode, so it
carries whatever the episode carried). Metadata *keys* are not scanned,
and do not need to be: a key is validated to `[a-z][a-z0-9_]{0,31}` at
every door into a space, so no key can hold any of these patterns.

The report names them, because `unscanned: 0` on its own is a claim about
coverage that cannot be checked:

```json
"withheld": {"count": 5, "by_kind": {"email": 5}, "kinds_applied": ["email"],
             "unscanned": 0, "surfaces": ["text", "source", "tags", "metadata", "facts"]}
```

**The expansions are refused rather than half-covered.** `evidence_graph`,
`graph_analysis`, `structural_context`, `multi_hop` and `graph_boost` each
build their own structure, and withholding does not reach inside them.
Asking for one of them together with `withhold` is a `422` naming which,
because the alternative is a report that covers the items and reads as
covering the answer. Covering them is open work; until it is done the
limit is a refusal and not a silence.

**The policy is checked before the search runs.** An unknown kind was
previously refused *after* the whole recall had happened and been logged
— work spent, and an event recorded, for an answer nobody receives.

## Cutting a document where it already divides itself

The chunker prefers a paragraph break, then a sentence end, then any
whitespace, then a hard cut inside a word. It knows nothing about
headings, numbered clauses or tables. So `Article 7.2` can end one chunk
while the clause it names begins the next, and a table's rows can arrive
without the header row that says what their columns mean — and the clause
number and the column names are usually the query terms.

The leading document pipeline answers this with a chunker per document
type: one for books, one for laws, one for papers, one for resumes. That
needs someone to declare what kind of document this is before it is read.
This reads the structure the document already carries.

```python
MemoryEngine(store, index, embedder, structure_aware=True)
```

That is the engine's rule for every record. One record can choose for
itself: `remember(..., chunking="structure")`, `scone remember --chunking
structure`, `"chunking": "structure"` on `POST /v1/episodes` and in each
batch record (`length`, `code`, `structure`, `semantic` or `unit`; unset keeps
the rule). The receipt says which way was actually used -- `code` for a
code source unless the record said otherwise -- and, for structure, the
chunker's own counts (`at_boundary`, `by_size`, `over_target`, `capped`).

An imported Word, OpenDocument or HTML file is stored as its paragraphs'
text, where a heading is a line like any other. Its reader keeps each
heading's level beside the text (`heading_level`), and a structure cut of
that file reads those headings back from the manifest kept with the
episode: it cuts at them whatever the line says, and the receipt carries
`document_headings`, the number of the file's own headings it read. A
record that is not an imported file carries no such count. A manifest
that does not match the episode's text is refused rather than ignored.

`unit` cuts an imported file one chunk per unit its reader named: a PDF
page, a slide (with its notes), a table or sheet row, a spreadsheet
cell's row (and a legacy `.xls` row), a JSON Lines record, an image or video frame or an audio segment.
Consecutive paragraphs in no unit, such as a Word document's body text
between two tables, are one `text` unit. A unit longer than the target
is split exactly as the length cut would split it, and the receipt says
so: `units`, `split_units`, `by_size` (chunks those splits added) and
`kinds`, the units by kind. It is asked for per file: `chunking` on
`POST /v1/documents` and `POST /v1/documents/pdf`, or on
`ingest_document`, `store_document` and `ingest_pdf`. A record whose
reader named no units, including anything that is not an imported file,
is refused before its episode is stored. The units come from the
manifest kept with the episode, so recovery cuts the same way. A file
imported again with a different `chunking` is the same episode, so the
first cut stands.
The choice is kept on the episode's metadata under `chunking`, so a
recovery after an interruption cuts the way the record asked; `code` on
a source whose name does not say its language, a mode not on the list,
or a metadata key that already says otherwise is refused before anything
is stored.

Built on `ingestion/structure.py`, which already finds headings, fenced
code and pipe tables and is already used by retrieval and source
inspection. Only what that parser deliberately leaves out is new:
setext headings, numbered and lettered clauses, `Q:`/`A:` pairs.

Chinese and Japanese numbering is a clause too, with or without a space
after it: `第三条`, `第2章`, `一、`, `（二）`, `1、`. A numeral alone is not a
clause, so `第一次` ("the first time") and `一九八四年` stay prose.

Both chunkers also cut Chinese and Japanese prose where it divides itself.
Its sentences end at a full-width stop (`。！？`) with nothing after it,
and before this every chunk of such text was cut at exactly the byte
target, inside a word. A full-width stop now counts as a sentence end,
with any closing quote or bracket after it kept in the sentence. With no
stop in reach, the later of a space and a full-width pause (`，、；：`) is
the cut. Text with no full-width punctuation chunks byte for byte as it
did, and a test holds the spans taken before the change.

Measured over this repository's own 23 documents, 437,141 bytes, at
commit 90ea0ce:

| | default | structure-aware |
| --- | --- | --- |
| chunks | 829 | 867 (+4.6%) |
| tables split across chunks | **10 of 34** | **0** |
| headings left as the last line of a chunk | **105** | **0** |

That measures the defect, not recall. Recall on our benchmark corpus
would have been zero and meaningless: it is chat sessions, which have no
headings, clauses or tables at all. Whether a reader answers better from
these chunks is unmeasured, and the cost of 4.6% more chunks — more
embeddings at ingestion, more candidates per query — is real. The sha
matters because the corpus is this directory: editing these docs changes
the numbers slightly, and the script that produces them lives beside the
write-up in `bench-runs/structure-chunking-2026-09-12/`.

Four rules, each with a test:

- **A document with no structure chunks byte-identically to today.** A
  test compares the two span lists directly, because cut positions decide
  what chunks exist and stored offsets are part of the shared
  specification.
- **A unit longer than the target is still split**, and the receipt
  separates chunks beginning at a boundary from chunks beginning where
  the target fell. "Structure-aware" must not read as "every chunk is a
  section".
- **A table is never cut, and travels with the heading above it.** A
  table over the target is counted in `over_target` rather than quietly
  returned: a caller sizing a context window needs that more than an
  assurance that nothing exceeds the target.
- **Structure that is not there is not invented.** `1984 was a year` is
  prose, not clause 1984. A `#` or a numbered step inside a fence is an
  example, not a heading. The `---` closing YAML front matter is not a
  heading underline. Each was a false positive found by reading the rule,
  and each has its own test.

**It is off by default**, because chunk boundaries decide what chunks
exist for every future ingest into a space, which is the caller's
decision and not ours to make for them.

One thing it got wrong and one thing the measurement caught are recorded
in `bench-runs/structure-chunking-2026-09-12/results.md`: prose between a
table and the next heading was dropped from every chunk — silently
unretrievable — and a heading above a table was separated from it. The
invariant test that should have caught the first asserted exactly the
right property and passed, because its fixture never contained the
junction.

## Embedding a chunk with the headings above it

A chunk cut from a long document loses what the document said it was
about: "within 30 days" under "## Refund policy" in "# Chapter 4" is,
once cut, only "within 30 days". With `heading_context=True`
(`SCONE_HEADING_CONTEXT=1`) each chunk's embedding input starts with the
path of headings above it, outermost first, and a code chunk's starts
with its file and the declarations it sits inside. For an imported file,
the headings its reader marked count as well, each running to the next
of the same or a higher level; recovery reads the same ones, so a
recovered chunk is embedded as an uninterrupted one would have been.

- **Only what is embedded changes.** Stored text is untouched, so recall
  still returns the exact excerpt.
- **The path is bounded.** At most `MAX_HEADING_CONTEXT_BYTES`; a longer
  path keeps its innermost headings, and the receipt counts the chunks it
  was cut for.
- **It is part of the vector writer's identity** (`heading-path-v2` since
  imported files' own headings joined the path). Vectors embedded with
  headings and vectors embedded without them never answer one search
  together: turning the setting on or off for an existing store reads as
  a mismatch until the vectors are rebuilt.
- **Off by default.** On the hash embedder, queries made of heading
  titles found their chunk about twice as often and queries made of the
  chunk's own words barely moved; the learned-embedder run has not been
  done yet. Both halves and their limits are in
  `bench-runs/heading-context-2026-09-13/results.md`.

### Keeping that context inside the embedder's window

A model embeds at most so many tokens and drops the rest without saying
so: the local BGE models read 512. What goes in front of a chunk -- the
heading line, a table's header row, the source and date -- comes first,
so a long enough prefix cuts off the end of the chunk it explains.
`embedding_budget=True` (`SCONE_EMBEDDING_BUDGET=1`) shortens the context
until the whole input fits:

1. the outermost headings, one at a time;
2. then the whole heading line;
3. then the table context.

The chunk itself is never cut. A chunk too long on its own is embedded
without context and counted in `body_over`. The receipt's
`embedding_context.budget` gives the window (`tokens`), how each chunk
came out (`fits`, `shortened`, `dropped`, `body_over`) and the `method`
that counted.

- **Counted by the model's own tokenizer where it has one.** The local
  embedder counts with an untruncated copy of its tokenizer
  (`tokenizer-v1`).
- **Estimated otherwise** (`estimate-v1`): a word counts one token per
  four letters of each camel-case part, a digit run one per two, each
  mark one, each character of a script written without spaces one, and
  two for the markers a model adds. On this project's 1,173 documentation
  chunks it never counted fewer tokens than BGE's tokenizer, and 1.36 times
  as many at the median. That margin has a cost on long chunks. At a
  2,000-character target with heading lines, the tokenizer found 332 of 371
  inputs fit, 9 shortened, 10 dropped and 20 bodies over 512. The estimate
  called 252 bodies over. Read `body_over` under `estimate-v1` as "might
  be over".
- **It bites rarely at the default target.** At 700 characters, no chunk of
  those documents came near 512 tokens with its heading line. It matters
  for long chunks and wide table headers.
- **It needs an embedder that declares its window** (`max_input_tokens`).
  The local BGE models do; one that does not is refused, rather than
  budgeted against a number nobody stated.
- **It is part of the vector writer's identity** (`;budget=tokenizer-v1`
  or `;budget=estimate-v1`), so turning it on over an existing store reads
  as a mismatch until the vectors are rebuilt. Off by default.

## A recalled body, with the signature and imports that make it readable

A chunk of code already says which declaration it came from —
`Engine.forget` — and that was where the answer stopped. It did not say
the declaration's **signature**, so a caller saw a body without its
parameters, and it did not say what the file **imported**, so a name in
the body could not be traced to where it came from. For "how is this
done here", a body without its signature and its imports is a fragment.

```bash
scone recall "write the paper to the shelf" --code-context
#   #4 inside Shelf.keep (line 17)
#       def keep(
#           self,
#           paper: str,
#           *,
#           tag: str = "unsorted",
#       ) -> Path:
#   #4 line 3: from __future__ import annotations
#   #4 line 5: import json
#   #4 line 6: from pathlib import Path
```

The reference that has this prepends the context **into** the chunk text.
Ours does not: invariant I1 says `content[span.start:span.end]` is the
source unchanged, and a chunk that has grown a header is no longer a
quotation of the file. So the context sits beside the chunk, quoted from
the source with the line numbers it came from, and every line of it can
be checked against the file.

Three rules, each with a test:

- **Nothing is guessed from content.** A file called `notes.md` holding a
  code block is prose that quotes code, and gets no code context — the
  language comes from the stored name, as everywhere else.
- **A source confirmed gone is dropped, not answered.** The rule merging
  and windowing already follow.
- **A list that stopped says so.** A file bringing in more names than the
  bound reports `more_imports`, because a count of what was listed must
  never read as a count of what the file imports. The episode budget
  counts reads that failed, for the same reason.

The signature runs from the declaring keyword to the end of its parameter
list. Python takes it from `ast` — the header ends where the body
begins — and the brace family scans a copy of the source with string
literals and comments blanked in place, so every line and column still
indexes the real file and the quote comes from the unblanked text.

**An earlier version of this counted brackets on the raw line and this
page argued that was sufficient**, on the grounds that a colon inside a
default argument is inside brackets. True, and beside the point: a
bracket inside a *string* is inside nothing, so
`def f(value="("):` never reached depth zero at its own colon and
`def f(value=")"):` drove the count negative. Bracket depth cannot
establish a lexical boundary without knowing what is code.

A header longer than twelve lines is quoted to there, and says so —
`clipped` on the holder, `shortened` on the receipt, and a note beside
the name in the terminal. A bound that bit in silence is the fault this
framework keeps making.

## Where the question's words are in a passage

A recalled passage says which lanes found it and how it ranked. It did
not say where the matching words are, so a page drawing the passage had
to tokenise it again, and got it wrong wherever its tokeniser differed
from the lexical lane's: that lane folds case, normalises width, keeps a
possessive with its word and reads scripts written without spaces as
characters and pairs.

```bash
scone recall "crane repainted" --highlight
# <score>  <date>  #<episode>  The harbour Crane was repainted in May.
#       matched: Crane, repainted
scone --json recall "crane repainted" --highlight   # adds "highlights"
curl -H "authorization: Bearer $KEY" "$URL/v1/recall?q=crane+repainted&highlight=true"
```

`highlights` sits beside `items`, one entry per item in the same order,
each with the question's `terms`, the `spans` (`start`, `end`, `term`)
and `total`. The rules, each with a test:

- **The lexical lane's own tokeniser decides.** The passage is read word
  by word and a word is marked when its tokens include a query term, so
  `CRANE` and `Alves's` are marked and a stopword in the question marks
  nothing. No stemming is invented here: the lane does not stem, so
  `repaint` does not mark `repainted`.
- **Offsets are code points of the text as returned**, and the spans are
  computed last, after windowing, merging, withholding and code context,
  because a span is only true of the text it was measured on.
- **In a script without spaces** the span is where the term's characters
  stand inside the run, and the overlapping characters and pairs of one
  query term merge into one region.
- **The bound says it bit.** A passage keeps at most `MAX_HIGHLIGHTS`
  spans; `total` counts all of them and `truncated` says some were not
  listed.

## One passage instead of three fragments of it

Small chunks match precisely and read badly. Three neighbouring fragments
of one paragraph are three citations to the same thought, and between them
they crowd the rest of the answer out of the limit.

The leading frameworks fix this by indexing a hierarchy at ingestion — a
parent node holding children — and merging children back into the parent
at retrieval. That means choosing the hierarchy before anyone has asked a
question, and re-indexing to change it.

```bash
scone recall "crane survey rust jib slew" --merge --limit 5
# 1 passage(s) joined from 3 chunk(s)
# joined 3 chunk(s) into #12: 11, 12, 13 (84% of it retrieved)
```

Over HTTP, `GET /v1/recall?q=...&merge=true`, with the record in
`merged`. Advertised as `recall.merge`.

**No hierarchy and no re-index**, because every chunk already carries the
byte span it came from: neighbours from one episode are merged by reading
the span that contains them. The shape of a merge is therefore decided by
what was actually retrieved, not by a decision taken at ingestion.

The rules it keeps:

- **A merged passage says what went into it.** `from_chunks` names every
  chunk absorbed, because a citation nobody can check is worse than three
  that can.
- **It keeps the best score of its parts, never their sum.** A sum would
  make a merged passage outrank everything by arithmetic rather than by
  relevance.
- **It is a passage, not a document.** An episode's fragments are joined
  in clusters, taken in the order they sit, each spanning at most
  `max_merged` bytes (4,000). A fragment too far from the rest is left
  alone and the report says so; silently returning most of a document to
  answer a question about a sentence would be worse than not merging. One
  far fragment does not stop the fragments beside each other from
  joining, and every cluster in an episode is merged from one read of it.
- **It says how much of itself was retrieved.** A merge reads the text
  between fragments too, so two short hits far apart would make a passage
  mostly nobody retrieved. `shares` gives, for every merged passage, the
  part of its bytes that retrieved chunks cover, overlaps counted once,
  counting the spans the chunks were retrieved at rather than any window
  around them.
  `merge_min_share` (`--merge-min-share`) leaves a sparser merge as
  fragments, counted in `too_sparse`. The reference merges children into
  a parent only when enough of them were retrieved; this is that rule in
  the bytes a reader gets. It defaults to 0, no floor, and is unmeasured.
- **A budget that bit says so.** At most 50 episodes are read per call;
  the rest stand unjoined, counted in `not_read`. A passage whose span
  reads back blank from an episode that was read (its text changed under
  the chunks) is left as fragments and counted in `blank`, apart from an
  episode that could not be read.
- **Withholding scans what a merge reads.** Merging runs after any window
  and before withholding, so an address in the text between two
  fragments is withheld like one inside them. The command line refused
  `--withhold` beside `--merge` while it merged after withholding; it now
  merges first and allows both. Merging does not combine with
  `compress`: a merged passage is reported under one chunk, so
  compression would not keep the others it holds.
- **The caller's ranking survives.** Neighbours are found per episode and
  emitted in the order they arrived, a merged passage taking the place of
  its best fragment. Walking a ranked list by episode and appending group
  by group would rearrange it, which is a change nobody asked for and
  nothing reports.
- **A deleted source is not an unreadable one.** If the episode is
  confirmed absent its text is gone, so the fragments quoting it are
  dropped rather than served from text this space no longer holds. A
  store that merely would not answer is a different fact: those fragments
  stand, and the reason says the merge *failed* rather than that there
  was nothing to merge.

Opt-in, because it is not yet measured. It changes the shape of an answer
for certain; whether it changes what is *found* is a question for the
bench, and until that number exists this does not become the default.

## The passage around a precise hit

Merging needs two hits in one episode. The commoner case is one: a
sentence matches exactly, and the answer is in the sentence after it. A
window returns each hit with the episode's own text around it, read at
retrieval from the byte span every chunk carries.

```bash
scone recall "rust jib slew grease" --window 200                         # 200 bytes either side
scone recall "rust jib slew grease" --window 1 --window-unit sentences   # one whole sentence either side
```

Over HTTP, `GET /v1/recall?window=1&window_unit=sentences`; the response's
`widened` receipt says what was done.

The leading framework builds its sentence window at ingestion: one
sentence per node, the neighbours stored beside it, and a re-index to
change the size. Here the unit and the count are the caller's, per
request, and nothing is re-embedded.

- **A window is quoted, never assembled.** Its text is the episode's
  bytes between two offsets, both on character boundaries.
- **Counted in bytes**, both edges land wherever the count does.
- **Counted in sentences**, the hit first grows to the whole sentences it
  touches, then by `window` whole sentences either side (at most 20), so
  both edges sit on sentence boundaries. With `window=0` it only
  completes the sentences it touches. A hit that is only the space
  between sentences takes the sentence after it. The window always holds
  the whole hit.
  - A sentence ends at `.`, `!` or `?`, with any closing quote or
    bracket, followed by space or the end. It does not end at an initial
    (`J. Anderson`), a title (`Dr. Okafor`) or a stop followed by a
    lower-case word (`i.e. before`), the rules the semantic chunker cuts
    by. It also ends at a CJK full stop (`。！？`), which needs no space
    after it, and at a blank line, so a heading or list item with no stop
    is a sentence of its own.
- **A window cut short says so.** `clipped` counts windows that met the
  start or end of the episode. `capped` counts sentence windows stopped at
  the 100,000-byte reach inside an over-long sentence. `aligned` counts
  byte windows moved off a partial character. The `why` line says each in
  words.
- **A source confirmed gone is dropped, not served**, and one that could
  not be read stands as it was and is counted.

Advertised as `recall.window` and `recall.sentence_window`.

## Cutting a window back to what bears on the question

A window of sentences either side of a hit holds the answer more often
than the hit alone, and it holds a good deal besides. `compress` keeps
the sentences the passage was retrieved for, always, and at most a share
of the sentences the window added around them:

```
GET /v1/recall?q=what+was+wrong+with+the+crane+jib&window=3&window_unit=sentences&compress=0.5
scone recall "what was wrong with the crane jib" --window 3 --window-unit sentences --compress 0.5
```

The reference's sentence optimizer embeds every sentence of a node and
drops the ones least like the question, so a node found by a sentence
that happens to score low can lose that sentence. Here the retrieved
span is never scored, only what widening added around it.

- **Two scorers.** `terms`, the default, needs no model. A sentence
  scores the weight of the question's words it names, each word weighted
  by how few of the passage's sentences name it, so a word the whole
  passage repeats counts for less than one it names once. Words that only
  ask, such as how, why, or the many of a "how many", do not count: on
  this documentation they kept every sentence saying "how many". A
  sentence naming none of the rest is never kept. `embedding` scores a sentence by its cosine
  to the question under the space's embedder, every sentence in one call,
  at most 400 per recall.
- **`compress` is a ceiling.** 0.5 of five sentences keeps at most two,
  never rounded up; 0 keeps only what was retrieved. Between sentences of
  equal score, the one nearer the hit is kept.
- **Kept text is quoted in runs, never spliced.** Adjacent kept sentences
  are one run of the episode's own bytes, listed in `compressed.runs` as
  byte offsets per chunk. Runs that are not adjacent are joined by ` … `,
  so the text never reads as one quote. The item's `start` and `end`
  bound the first and last run.
- **What was not cut says so.** A passage with no retrieved span inside
  it is left whole and counted in `unpinned`; a passage past the
  embedding budget is left whole and counted in `unscored`; a question
  naming no word the terms scorer can weigh cuts nothing, and `why` says
  each in words. `bytes_before` and `bytes_after` count what was saved.
- **Refused without a window of sentences**, where there is nothing it
  may cut, and beside a merge, whose passage is reported under one chunk
  so the other retrieved chunks inside it would not be kept; on the
  command line also beside `--parts`, which answers without it.
  It runs after widening and before withholding, so what withholding
  scans is what is returned.

Neither scorer is measured. Which share keeps the answer while saving the
most is an answer-bench question, and until that number exists `compress`
stays opt-in and its record says `measured: false`. Advertised as
`recall.compress`.

## Where each sentence of an answer came from

An answer composed from recalled passages reads as sourced whether it is
or not. The leading framework asks the model to cite numbered sources as
it writes, and nothing checks the numbers afterwards. Attribution takes an
answer already written, by a model or a person, and the stored chunks it
was composed from, and aligns each sentence to them without a model.

```bash
scone attribute --answer "Priya moved the launch to March because the audit ran late. The board was not told." \
  --chunk 12 --chunk 14
# quoted chunk:12: Priya moved the launch to March because the audit ran late.
# unattributed: The board was not told.
```

Over HTTP, `POST /v1/answers/attribute` with `{"answer": ..., "chunk_ids": [...]}`.
The chunks are read from the space, never taken from the caller, so an
answer cannot be attributed to text the space does not hold. An id the
space does not hold is named in `chunks_missing`, and one it holds whose
text is blank, with nothing to attribute to, in `chunks_empty`.

Each sentence gets one status:

- **quoted**: it shares a run of at least 5 consecutive words with a
  passage, compared case-folded, with the punctuation between words
  ignored. The record gives the run's span in the passage and its text.
- **overlapping**: no such run, but at least 60% of its content words
  (the lexical lane's tokens, stopwords left out) are in one passage.
- **unattributed**: neither, for every passage given.
- **too_short**: fewer than two content words.

A quote beats an overlap and a longer run beats a shorter one; then more
of the sentence's words wins, then the passage given first. Numbers in a
sentence that its passage does not hold, such as `14th` against a passage
saying `twelfth`, are named on the sentence.

What it is not:

- **Word overlap is not support.** A sentence can quote a passage and
  still misstate it, and a faithful paraphrase can come out unattributed.
  The record says `verified_accuracy: false`.
- **Both rules are unmeasured**, and the record says so (`rules.measured:
  false`).
- **A run is counted between words separated by space or punctuation**,
  so text in scripts written without spaces can overlap but is rarely
  quoted.

An answer over 20,000 characters or more than 50 passages is refused, not
cut. Advertised as `answers.attribution`.

## What a codebase says about itself beyond who calls whom

Call edges are not a code graph. Two questions people actually ask are
answered by neither `calls` nor `imports`, and both are readable without
a model:

**Which types are which.** A class hierarchy is how anyone navigates a
codebase, and nothing in a call graph says a word about it. `inherits`
edges now come out of the same pass:

```
pkg/shelf.py:Paper  inherits  pkg/shelf.py:Shelf      # a base this file defines
pkg/shelf.py:Shelf  inherits  pkg.base.Store          # a base from an import
web/shelf.ts:Shelf  inherits  Store                   # extends
web/shelf.ts:Shelf  mixes_in  Face                    # implements
```

**What a signature names.** A call edge says what a function runs;
nothing said what it takes or returns, so "what uses `Receipt`?" found only
its constructors. A Python function's parameters (including `*args` and
`**kwargs`), its return and its annotated locals, and a class's annotated
attributes, now give `uses_type` edges, through generics (`list[Receipt]`)
and quoted forward references (`"Shelf.Label"`) alike:

```
pkg/shelf.py:put          uses_type  pkg/shelf.py:Shelf     # a class this file declares
pkg/shelf.py:put          uses_type  pkg.receipts.Receipt   # a name imported from the project
pkg/shelf.py:later.inner  uses_type  pkg.models.Paper       # a nested function, named as defines names it
```

A name in a class body means that class's member first. The standard library
and built-ins are left out as noise (`str`, `Optional`, `datetime.date`), and
so is a local function named in an annotation, since a function is not a
type. A name taken out of a module (`from pkg import models`) and then used
as `models.Paper` is not followed, for the same reason calls are not: a
file cannot tell a module from an object. The edges are not part of a blast
radius (`affected` walks `calls`, `imports`, `inherits` and `mixes_in`).
Over this package's 408 files they add 4,139 edges beside 14,446 call
edges, in the same pass.

**Extending a class and satisfying an interface are different relations,
and collapsing them loses the question people ask.** "What is a Shelf?"
has one answer; "what can be used as a Face?" has many, and a graph with a
single `inherits` edge cannot tell them apart. So `implements`, Scala's
`with` and Rust's `impl Trait for Type` are `mixes_in`, while `extends`
and Python's base list are `inherits`. Rust has no class inheritance at
all, so **no Rust edge is ever `inherits`** — a claim the extractor used to
make on every `impl … for …` line.

A colon clause says less than a keyword does, and how much less depends on
the language, so the rule is language-aware rather than uniform:

| form | read as | why |
| --- | --- | --- |
| `extends Base` | `inherits` | the keyword says so |
| `implements Face`, `with Store`, `impl Store for Shelf` | `mixes_in` | the keyword says so |
| Kotlin `: Base(), Store` | `inherits Base`, `mixes_in Store` | Kotlin constructs its superclass and never constructs an interface, so the parens are the language's own answer |
| C++, C#, Python `: Base` / `(Base)` | `inherits` | C++ has no interfaces; reading a base list there as a mixin would be a new wrong claim |

The last row is a limitation stated rather than hidden: a C# `: IFace` is
an interface and is recorded as inheritance, because nothing on the line
distinguishes it from a base class and inheritance is the more common
case. Kotlin is the only language where a colon list carries the answer.

A base the file can see is named by its path, like any other declaration.
One that arrived through an import whose module resolves to a file is
named there. Anything else is recorded **as the source wrote it** — the
same rule imports already follow, because the name is what the file said
even when its home is unknown. A base that is not a plain name (a
subscripted generic, a call) is left out rather than guessed at.

**Why the code is the way it is.** The rationale is in the comments and
the decision it came from is in an ADR or an RFC, and both were
previously invisible:

```
pkg/shelf.py:Shelf.open  notes  the index is rebuilt on open because a
                                half-written index is worse than none
pkg/shelf.py:Shelf.open  cites  ADR-0007
pkg/shelf.py:Paper       flags  the paper shelf cannot hold two of the
                                same thing yet
```

Three things this gets right that one predicate would not:

- **A rationale and a known problem are different claims.** "Why does
  this exist" and "what is wrong with it" are different questions;
  `notes` (`WHY:`, `NOTE:`, `RATIONALE:`) and `flags` (`TODO:`, `FIXME:`,
  `HACK:`, `XXX:`) answer them separately, and one predicate for both
  would answer neither.
- **A citation is a node, not a string.** `ADR-0007`, `ADR 7` and `adr#7`
  normalise to one name, so every declaration that cites a decision
  record is reachable from it — which is the point of putting it in a
  graph rather than in a grep.
- **Rationale belongs to the thing it explains.** A note is attached to
  the innermost declaration whose lines contain it, not to the file that
  happens to hold it, using the same declaration spans recall cites.

Only tagged comments become claims: an untagged line is a remark, not a
statement about the code.

**Never from a string literal.** A string holding `# WHY: …`, `ADR-0007`
or `class X extends Y` is data, and reading it would have the graph
assert something the source never said — the one thing this must not do.
Python comments therefore come from `tokenize`, which knows a comment
from a string that looks like one, and citations additionally from
docstrings, because a docstring is documentation while an arbitrary
string is not. The brace languages have no tokenizer here, so their
source is scanned with string contents blanked in place, which keeps
every line and column where it was.

An earlier version read the raw source with a regex and fabricated
claims from quoted text. It also joined `extends Base implements Face`
into one invented target, resolved `Store as Shelf` to the local
nickname, and let `import json, csv` claim both names came from the last
module. Those were found by review, on hand-built counterexamples rather
than on a corpus — a graph whose claim is that it does not guess has to
be tested on the shapes that tempt it into guessing.

### What a project says it depends on

A manifest is where a project writes down what it needs, and until now
none was read: the graph knew every `import requests` and nothing about
which project declares requests, at what version, or only for its tests.
`pyproject.toml` (PEP 621, PEP 735 dependency groups, Poetry),
`requirements*.txt`, `Pipfile`, `package.json`, `Cargo.toml`, `go.mod`,
`pom.xml`, `build.gradle` and `build.gradle.kts`, `Gemfile`,
`composer.json` and a .NET project file (`.csproj`, `.fsproj`, `.vbproj`)
are read wherever they sit, by `scone map --graph` and by any remembered
file with one of those names as its source, into claims like a source
file's -- quoted from the line, cited to the file, extracted rather than
stated:

```
packages/memory/pyproject.toml  defines        scone-memory
scone-memory                    depends_on     requests        "requests>=2.31",
scone-memory                    depends_on     uvicorn         "uvicorn[standard]==0.30.1",
scone-memory                    develops_with  pytest          test = ["pytest>=8", "pytest-asyncio"]
```

Two predicates, because they answer two questions: `depends_on` is what
the project needs to run, optional extras included; `develops_with` is
what it needs to build, test or document itself -- build requirements,
dependency groups, dev dependencies. A project does not run on pytest.

The object is the package's bare name, spelled as its index spells it (a
Python name lowercased with runs of `-_.` as one `-`, a crate with `_` as
`-`, an npm, Composer or NuGet name lowercased, a Maven or Gradle
artifact as `group:artifact`, a gem as named), and the extras, version
and marker stay in the quote. Each format's own way of saying "for
tests" is read: Maven's `<scope>test</scope>` and build plugins,
Gradle's `test*`, `annotationProcessor`, `kapt`, `ksp` and `classpath`
configurations and its `plugins { id }` block, a Gemfile's
`group :test, :development` blocks and inline `group:`, Composer's
`require-dev`, a .NET `PrivateAssets="all"` reference, a Pipfile's
`[dev-packages]`. What the reader cannot place it leaves out: a Gradle
`project(':lib')` or map-style dependency, a Composer platform
requirement (`php`, `ext-json`), a .NET `ProjectReference`. So a package five manifests name is one thing in the graph,
and each manifest's line says what it asked for. The subject is the
project's declared name where it has one, else the manifest's path. A Go
`// indirect` requirement is left out: it is what a dependency needs, not
what the module declares. A renamed Cargo dependency is the crate it names,
not the alias. A Maven coordinate spelt with `${project.groupId}` or
`${project.artifactId}` is read as Maven reads it, and a plugin without a
group is Maven's own; any other placeholder is left out rather than
guessed. Left unread, by design: Maven's `<dependencyManagement>`,
`<pluginManagement>` and `<profiles>`, a Gradle `platform(...)` or
version-catalog reference, a Gemfile's `gemspec`, and
`Directory.Build.props`/`Directory.Packages.props`.

Dependency names are not bound to import names. They differ often
enough (`beautifulsoup4` and `bs4`, `Pillow` and `PIL`) that binding them
would guess, and the graph does not.

A package a manifest names is an entity, as a code symbol is, so the
questions the graph answers about code reach it: `scone graph affected
requests` lists the projects whose manifests declare it, through
`depends_on`, beside the files that import it, and a test dependency's
blast radius runs through `develops_with`.

### What a project hands its agents

An MCP configuration is a manifest of a different kind: where a project
writes down the tool servers its agents talk to. Until now none was
read, so the graph knew a project's packages and imports and nothing
about the servers, what they run on, or which variables must be set
before one starts. `.mcp.json` (Claude Code), `claude_desktop_config.json`
(Claude Desktop), `mcp.json` (Cursor, Windsurf, VS Code under
`.vscode/`), `mcp_servers.json`, `mcp_config.json`,
`cline_mcp_settings.json`, `.gemini/settings.json` and Codex's
`.codex/config.toml` are read wherever they sit, by `scone map --graph`,
by `sync`, and by a remembered file with one of those names as its
source. The `map` and `sync` walks pass dot-named entries by, as they
always have, with these files and the four tool directories that hold
one (`.vscode`, `.cursor`, `.gemini`, `.codex`) as the stated exception;
from those directories only the configuration is read, so
`.vscode/settings.json` stays unread as before. A configuration makes
claims like a manifest's:

```
app/.mcp.json             defines       app/.mcp.json:git       "git"
app/.mcp.json:git         runs_with     uvx                     uvx
app/.mcp.json:git         depends_on    mcp-server-git          mcp_server_git
app/.mcp.json:git         requires_env  $GIT_TOKEN              "GIT_TOKEN"
app/.mcp.json:remote      connects_to   https://mcp.example.com https://mcp.example.com
```

- `defines`: the file defines each server it configures, named by the
  file and the server's key, so two files that both configure a
  `filesystem` stay two things. A server is no declaration a call could
  reach: like a manifest's project name, it is left out of call
  resolution.
- `runs_with`: the executable a local server starts with, by its base
  name (`npx`, `uvx`, `docker`, `node`), so "everything that runs
  through docker" is one question. A command that is itself a reference
  (`${TOOL_HOME}/bin/server`) names no executable.
- `depends_on`: the package the server runs, when the executable says
  which index it comes from -- the first positional argument of `npx`,
  `bunx` or `pnpx` is an npm package (`--package` names it instead), of
  `uvx` or `pipx` a PyPI distribution (`--from` and `--with` name
  distributions too) -- spelled as its index spells it, the same object
  a `package.json` or `pyproject.toml` names, so a server and a manifest
  that name one package meet at one entity. A path, a URL, a VCS spec
  and a reference are not packages. A docker image is left out: the
  `run` line's flags cannot be told from the image without knowing every
  flag, and a guess would name the wrong thing.
- `requires_env`: the environment variables a server needs, by name and
  never by value: the keys of its `env` map; a `${NAME}`,
  `${NAME:-default}` or `${env:NAME}` reference in its command,
  arguments, URL, headers or env values; a `-e NAME` or `--env NAME` a
  docker run passes through from the host (`-e NAME=value` sets a value
  and needs nothing); and Codex's `bearer_token_env_var` and
  `env_http_headers`. The object is `$NAME`, as a shell writes it. A
  reference in lower or mixed case (`${workspaceFolder}`,
  `${input:token}`) is a tool's own variable, not the environment's,
  unless it says `env:`. One variable a server names twice is one claim.
- `connects_to`: the origin (scheme, host and port, lowercased) of a
  remote server's URL, when the host is written out rather than
  referenced.

A value is never quoted. What an `env` map holds is what a configuration
keeps secret, an argument or header may carry one, and a URL's query
can; so every claim quotes the token that grounds it -- the server's key,
the command, the package as written, the variable's name, the URL's
origin -- and its byte span covers that token alone. The three new
predicates are many-valued, code-shaped, followed by `graph affected`
(a change to the variable or the executable reaches the servers that
rest on it) and name entities the graph may only know by name, as
`imports` does; the object of `runs_with` and `connects_to` is a product,
as `depends_on`'s is.

### Schemas as things code rests on

A code graph that knows a project's files and packages still stopped at
the database: a table is what half the functions read and write, and a
migration that drops a column reaches every one of them. A `.sql` (or
`.ddl`) file is now read as the schema it writes down — the reference
graph introspects a live database; a repository holds the schema as text,
a diff changes it, and nobody has to connect to anything. `scone map` and
`remember()` with such a source record, each claim quoted from its line:
the file `defines` each table and view (`db/schema.sql:orders`); a table
`defines` each of its columns (`db/schema.sql:orders.customer_id`); a table
with a foreign key — inline `REFERENCES`, a `FOREIGN KEY` constraint, or an
`ALTER TABLE … ADD CONSTRAINT` — `depends_on` the table it references; a
view `depends_on` the tables it selects from. Comments are not read, quotes
and brackets around a name are not part of it, and a schema-qualified name
is kept as written. `depends_on` is the predicate a manifest's dependencies
already use, so `scone graph affected db/schema.sql:customers` lists the
tables and views that rest on `customers`, nearest first, beside the code
that imports a module. Nothing binds code to a table: a query is a string,
and a string that names a table is a guess this graph does not make.

### What a diff reaches

```bash
git diff main...HEAD | scone graph impact --root .
scone graph impact change.diff --root . --json
```

`graph affected` answers for one symbol; a pull request touches lines in
many files, and the question people bring to it is the same one asked of
the diff as a whole. `graph impact` reads a unified diff (what `git diff`
writes, quoted paths included), re-reads each changed file at `--root` —
the tree **after** the change, the diff's `b` side, whose line numbers the
hunks give — with the declaration reader that cuts files into chunks, and
names what every hunk touches at two
levels, because the graph holds edges at two levels: the declaration under
the changed lines (`pkg/store.py:Shelf.keep`), for calls the graph bound to
it, and the file, spelt as its path and, for Python, as the module an
import names (`pkg.store`). A change outside every declaration — an import
line, a constant — touches the file alone; a file the root no longer holds
is taken as removed and asked about as a file. Each thing touched is listed
with what the graph was asked and what it said (`found`, `nothing`,
`unknown` for a file this graph was never given), and everything that rests
on the change follows once, nearest first, with the fewest hops from
anything touched and the name it was reached through. Bounds on files
examined, things asked about and dependants listed are each disclosed when
they bite; so is a bound that bit inside one of the graph's own answers (a
list it cut, a depth it stopped at), a name longer than the graph takes, a
file the diff names outside the root, and a root that is not a directory;
and an empty answer means nothing *in this graph* rests on the
change. The module spelling is a second question asked, not a link
asserted: where a file and the module that names it are joined only at
ingestion (`scone map` supplies the resolver; `remember()` does not), the
answer says which spelling found what.

### Claims read from files hold side by side

A ledger predicate holds one value at a time unless configured
otherwise, and that is right for what people state: "lives in Lisbon"
retires "lives in Austin". It was wrong for what the readers extract. A
module with three imports held one and closed two as superseded, a file
with two functions defined the second, and every graph built on the
ledger kept the last claim of each kind and called the rest history.

The predicates the framework extracts -- `defines`, `imports`,
`imports_when_called`, `imports_for_types`, `calls`, `inherits`,
`mixes_in`, `uses_type`, `notes`, `flags`, `cites`, `depends_on`,
`develops_with` -- are many-valued by their nature, declared so in the
core (`scone_memory.core.extracted.MANY_VALUED`), and no configuration
takes one out of that set. `SCONE_MANY_VALUED` still adds predicates a
person names; `GET /v1/graph/schema` marks both kinds as `many`.

### A claim read from a file holds while the file says it

`replace` and `sync` store a changed file as an update: the old episode
is forgotten, the new one stored, its claims read. Forget's contract
leaves claims standing, rightly -- a person's memory of a fact survives
deleting its source -- but for what a reader extracted that meant the
ledger held what the file used to say beside what it says now: a module
that dropped an import still imported it, a function that was removed
was still defined.

So on replacement, the extracted claims the old episode grounded that the
new content did not restate are closed, reason `no longer stated by
<path>`, event kind `source_changed`; a restated claim -- the same
subject, predicate and object read out of the new content -- is one fact,
still holding. When a sync asked to `remove` forgets a file that is gone,
every extracted claim it grounded is closed, reason `<path> was removed`,
kind `source_removed`. Neither is counted as a manual closure. What a
person stated about the episode is left alone, a plain `forget` still
touches no claim, and nothing happens with the code graph off.

The receipts say what was done. `Replaced.claims_closed` counts the
closures, `None` when the store cannot read claims by episode (nothing is
closed on a guess); `claims_unread` is true when the old episode grounded
more claims than one read returns, so some were not examined and may
still stand. A sync receipt carries `claims_closed` and `claims_unread`
only when there is something to say, so a sync without the code graph
reads exactly as it did. The durable directory-sync service does not read
claims and is untouched.

## Code: cut where the declarations are

A source file stored under a name that says which language it is in
(`.py`, and the brace family: `.ts`, `.js`, `.go`, `.rs`, `.java`, `.c`,
`.swift` and their neighbours) is cut at its declarations rather than
every `chunk_target` characters. A function is a chunk when it fits; a
longer one is cut at its own lines, preferring a nested declaration and
then a blank line, and whatever is left at the end goes back to the piece
it was cut from; small neighbours share a chunk. Nothing is rewritten:
`content.encode()[start:end].decode()` is still the source, exactly.

The name decides, never the content: a note that quotes code is prose.
`code_aware=False` puts the ordinary chunker back.

Python is parsed with `ast`, so its spans are exact — the `def` line with
its decorators and the comment lines written directly above it, through
the last line of the body. Brace languages are read by a header line and
a brace count, which is the guess a parser would not have to make: a
brace inside a template literal that spans lines can be counted wrongly,
and the result is a chunk boundary in the wrong place, never a changed
byte. Line endings are counted as Python counts them, a lone carriage
return included. Nesting is read 32 deep; what Python's own parser
refuses is cut the ordinary way.

Every recalled item now carries where it came from, worked out from the
episode rather than stored, so a chunk written before any of this answers
the same way:

- `start` and `end` — the chunk's own UTF-8 byte span of its episode.
- `first_line` and `last_line` — the lines that span covers.
- `declaration` — the declaration that holds all of it, qualified by
  everything that holds it (`Engine.forget`), or `null` when the chunk is
  module-level code or crosses more than one declaration.

### What that is worth, measured

```bash
scone bench-code src/scone_memory --k 5 --asked name
scone bench-code src/scone_memory --k 5 --asked name --by-length   # to compare
```

It stores every source file under a directory and asks one question per
documented function: either the
docstring's first paragraph as written, or "what <the function's name, in
words> does". A hit is a returned chunk holding that function's own `def`
line. On this package's own source — 273 files, 737 functions, hash
embedder, k=5:

| asked | cut | own definition in top 5 | own file | a whole declaration |
| --- | --- | --- | --- | --- |
| docstring | declarations | 691 (94%) | 717 (97%) | 292 (40%) |
| docstring | by length | 697 (95%) | 718 (97%) | 259 (35%) |
| name | declarations | 321 (44%) | 406 (55%) | 126 (17%) |
| name | by length | 325 (44%) | 410 (56%) | 124 (17%) |

Read it honestly. **Cutting at declarations does not find more.** What it
changes is what comes back: a whole function rather than the end of one
and the start of the next, which is what makes a citation quotable. The
first pair of rows is close to an exact-match measurement — the docstring
is in the chunk, verbatim — so what it really says is that chunking a
file does not bury a function either way. The second pair is the
interesting one: asked in a person's words, two in five questions do not
return the function's own definition. That is an embedder question, not a
chunker one, and it is where the next gain is.

### Judging an answer with a local model

Retrieval metrics say whether the right passages came back; `bench.evaluators`
says what a local judge (any `ChatModel`) makes of the answer written from
them, our way: the judge is handed the exact texts and asked for a small JSON
verdict, the verdict is parsed strictly, and what could not be parsed, what
the judge failed to produce, or what exceeded the input bound comes back as
**unverified** — never a pass, never a fail. `faithfulness` (every claim in
the answer, and whether a context supports it), `answer_relevancy`,
`context_relevancy` and `correctness` (against a reference, on a five-point
scale) are the four the reference framework also asks. Two more:

- `pairwise(judge, question=…, answer_a=…, answer_b=…, reference=None)` asks
  which of two answers is better, **in both orders**. A judge that prefers
  whatever it read first gives two different verdicts; that disagreement is
  reported as a tie with the reason, not resolved by a coin. The score is 1
  when A wins, 0 when B wins, 0.5 for a tie; two calls per judgment.
- `semantic_similarity(embedder, answer=…, reference=…, passing_similarity=0.8)`
  is the cosine between the two embeddings, clamped to [0, 1], with no judge
  called. It says nothing about truth — two fluent wrong answers can sit
  close together — and what it says depends on the embedder, whose id is on
  the reason.

A judgment is the judge's opinion, not proof; the bench reports it beside the
metrics that need no judge, and says which is which.

### Questions your own corpus answers

```bash
SCONE_CHAT_URL=http://127.0.0.1:11434/v1 SCONE_CHAT_MODEL=gemma4-e4b-ctx8k \
  scone bench-questions docs/ --set docs-questions.json --write
scone bench-questions docs/ --set docs-questions.json --k 1,5,10
```

LongMemEval-S measures conversations and `bench-code` measures functions;
nothing measured retrieval on a corpus of your own. `bench-questions --write`
stores every text file (`.md`, `.txt`, `.rst`) and PDF under a directory in
its own in-process store, shows a sample of the chunks to the configured
local model (at most `--max-chunks`, sampled by `--seed`; the set says how
many chunks there were), and asks it for `--per-chunk` questions each with
the sentence that answers it. A question is kept only when that sentence is
in the chunk the model was shown, word for word (whitespace aside); one
whose quote is invented is dropped and counted, as is a reply that is not
the JSON asked for, a call that failed, and a chunk too long to show. Every
kept question can therefore be checked by anyone holding the text.

Measuring needs no model and no chunk ids: the same root is stored again,
every question is asked, and a returned passage that holds the quote is a
hit — `quote in top k`, MRR, and `source in top k` for questions whose
chunk came from a file. Because the truth is a quote and not a chunk id,
one set measures the corpus stored with a different chunk size, store or
embedder, as long as the text is the same; that is what makes it a bench
for ingestion changes (a PDF read in a different order, a chunker cut at
different places) and not only for retrieval settings. The set is a JSON
file with a version, so a saved set is refused by a reader that does not
know its shape.

## Choosing settings by measuring them

```bash
scone tune bench-data/longmemeval_s.json --sample 30 --k 5 --candidates 100,200 --contextual
```

Every retrieval knob is a choice somebody could argue about, and none is
right everywhere. `tune` runs the ordinary bench once per setting, over
the same stratified sample and the same questions, and prints what each
found. It varies three things and holds everything else at whatever the
environment says, so what it measures is the difference between the
settings and nothing else: `SCONE_RECALL_CANDIDATES`,
`SCONE_DEMOTE_RESTATED`, `SCONE_CONTEXTUAL_EMBEDDINGS`.

### The context lane: found by what it is under

A chunk under the heading "Refunds" in a document titled "Billing
rules" need not say either word, and a query about billing refunds then
misses it in the text lane, which finds the words a passage has and
only those. With `SCONE_CONTEXT_LANE=1` (`MemoryEngine(...,
context_lane=True)`) each chunk's context is derived at ingestion with
no model — the headings enclosing it, the document's title (its top
heading, or a short first line that does not read as a sentence) and
the words of the source's name — keeping only what the chunk itself
lacks, and indexed **beside** its text, never in it. Stored text and
offsets do not change. At recall the same query the text lane got is
searched over that index as a third lane and fused by rank at twice the
weight of the others; `lanes.context` on each item says where the lane
placed it.

The weight is measured, not guessed. Rank fusion is flat, so a lane
that finds what the others cannot needs weight to be heard at all: on
the `under-v1` benchmark (`testing.context_lane_benchmark`: twenty
passages under a heading whose words they never say, eight distractors
each repeating the question's words) the passage reached the top five
in 0 of 20 cases without the lane, and with it in 0 at weight 0.5, 1 at
1.0 and 13 at 2.0 — where the distractor also lost first place in 13
cases. The entity lane made the same choice for the same reason. The
cost is on the record too: a passage under the words can now come
before one that merely says them, which is what the benchmark's
question wants and a literal search would not. A document's frequent
words were tried as context and left out: spread over every chunk they
make the lane fire on mentions rather than on structure.

The words are bounded — at most 32 per chunk, headings first, then
title, then source words, the rest counted as omitted — and the lane is
honest about where it is not: the SQLite and in-memory stores keep the
index, and an engine with the lane on over a store that does not
reports `context lane: not kept by …` in `degraded` rather than
pretending the lane ran.

### Fusing by distribution

`--fusion distribution` (`fusion="distribution"` on `recall`, the same name
over HTTP) is a third way to add the lanes, beside rank fusion and
relative-score fusion. Relative-score fusion scales each lane by its two
extremes, so one outlier at the top of a lane pushes every other candidate
toward zero. Distribution fusion scales each lane by its own mean and
spread: the mean less three standard deviations is 0, the mean plus three
is 1, and a score outside that range is clipped. An outlier then counts as
an outlier, and the candidates near the lane's mean keep their credit
instead of being pushed toward zero by it. (Neither scaling cares about a
lane's units; both place a score by its lane's own shape.) What follows: a
lane's top candidate no longer gets full credit for being top -- the top of
a two-candidate lane sits at two thirds -- so a lane that returns few
candidates speaks more quietly than one that returns many, and the lanes'
weights apply on top of that; the clip itself bites only on lanes of eleven
or more, since fewer candidates cannot put one past three deviations. The
conventions of relative fusion hold: a lane whose scores do not spread gives
every candidate full credit, and a lane that reports no score contributes
its order. The recall event records which fusion ran. It is a choice, not a
default, until a measurement on the benches says which mode should be.

### The vector lane's voice in fusion

Reciprocal rank fusion gives every lane the same voice. That is right
when both lanes know something the other does not, and wrong when one
is a weak echo of the other: an embedder whose vectors are hashed
tokens ranks by word overlap, badly, and its confident wrong picks can
outvote the text lane's right ones. Measured on LongMemEval-S with that
embedder, the text lane alone was ahead of the fused ranking.
`SCONE_VECTOR_WEIGHT` (`MemoryEngine(..., vector_weight=)`) is the
vector lane's weight against the text lane's 1.0 — a number above 0 and
at most 4 — and every recall event records `fusion_weights`, so a
ranking can always be read back to the voices that made it. Unset, it
follows the embedder: 1.0 for any embedder but hashed tokens, and 0.01
for hashed tokens. Change it only on a number.

The hashed default is measured against LlamaIndex's best retrieval
(its BM25 retriever fused with its vector retriever) on LongMemEval-S,
both sides on the same hashed vectors, sessions folded from passages
([results](../benchmarks/northstar-defaults-2026-09-14.results.md)):

| hashed vector weight | frozen 50: R@5 / all@5 / R@15 / MRR | 100 other items: R@5 / all@5 / R@15 / MRR |
|---|---|---|
| 1.0 | 0.84 / 0.72 / 0.92 / 0.759 | 0.93 / 0.74 / 0.98 / 0.789 |
| 0.25 (the default before) | 0.88 / 0.72 / 0.98 / 0.807 | 0.95 / 0.79 / 0.99 / 0.852 |
| **0.01** | **0.90 / 0.76 / 1.00 / 0.838** | **0.96 / 0.84 / 0.99 / 0.885** |
| LlamaIndex BM25 + vector | 0.88 / 0.74 / 0.98 / 0.831 | 0.91 / 0.71 / 0.99 / 0.810 |

Every voice the hashed vector lane had in the order cost the ranking.
At 0.01 the fused ranking scored exactly as the text lane alone did on
both samples, and its top five sessions were the text lane's in 149 of
the 150 items. The
lane still runs at that weight: it gives the confidence signal
(`top_similarity`, the similarity floor), it answers alone when the text
lane fails, and a passage only it found can still come back.
`SCONE_VECTOR_WEIGHT=0.25` restores the previous
default. Relative-score fusion and larger chunks also moved the numbers
on these samples; neither is a default, and the results file says why.

The weight is a voice against the text lane, so it applies only when the
text lane brings passages. When it brings none (`lanes=["vector"]`, a text
lane that failed, or one that found nothing), the vector lane ranks at a
full voice of 1.0, and that recall's `fusion_weights` says so. The recency
term is sized against a full voice: at a hundredth, the vector lane's
first and second places differ by what about half an hour of age is
worth to that term (at 1.0, about two days), so newer passages would
come back first instead of closer ones.

### Both stores agree on every script

Two stores that answer the same query differently are a bug a reader
cannot see. The in-memory lane cuts an unspaced run — a Japanese or
Thai phrase — into character grams and finds a part of it; SQLite's
built-in tokenizer kept the run as one token and could not. The SQLite
text lane now ranks our own tokens: a derived table holds each chunk's
terms exactly as the lexical tokenizer makes them, diacritics folded as
the in-memory lane folds them, searched through an FTS5 shadow that
splits only on the spaces between them. It is versioned by the
tokenizer and the Unicode data it ran under and rebuilt whole when
either changes — lazily, so an old database opens at once and each
space pays as it is read — and triggers keep it current on write and
delete. A parity test pins that Japanese, Chinese, Thai, Korean and
accented Latin queries find their passage through the text lane in
both stores.

### A word's family by prefix

"bills", "billing" and "billed" are one word to a reader and three to the
text lane. The usual answer is a stemmer over the index, and it is the
wrong one here: it rewrites what every store holds, ties the tokenizer's
version to a set of language rules, and puts a guess ("policies" and
"police" as one) where a reader cannot see it. `SCONE_LEXICAL_STEMS=1`
(`MemoryEngine(..., lexical_stems=True)`) does less: a query term that
ends in a known English suffix also searches as a prefix of its stem —
`bill*` for "billing", `invoic*` for "invoices" — so the family is found,
the index is untouched, and the result's `prefixes` says which prefixes
were added and whether the store could take them (`applied`; the SQLite
and in-memory stores can, and a family counts as one term in the score,
not several). A language the rules do not know, a short word, a number,
is left exactly as it was. The rules keep a stem of at least three
letters after a strong suffix and four after a plural or a final "e", do
not strip a plural after "s", "u" or "i", and reduce a doubled consonant
except where English keeps it. Measured on LongMemEval-S before it was
a flag; the numbers are on the pull request that added it.

### Synonyms the caller wrote down

The lexical lane finds the words a passage has, and only those. A
passage that says "automobile" is invisible to a query about a "car"
unless somebody wrote down that in this corpus the two are one word.
`SCONE_SYNONYMS=./synonyms.txt` names that list — one group per line,
terms separated by commas, `#` for comments — and `MemoryEngine(...,
synonyms=Synonyms(groups))` gives it in code. A query term that is in a
group adds the group's other members to the **text lane's** query; the
vector lane's query stays as written, because an embedder already knows
what it knows about the two words and padding its input with a list
would move the vector in ways nobody measured. The lanes are fused by
rank, so a passage found only through an added word competes on rank,
never on a score the addition inflated.

No model proposes a synonym here, and nothing is guessed: matching uses
the lane's own tokenizer, so case, possessives and stopwords are treated
exactly as the index treats them, a phrase matches as a phrase, and the
result's `expansion` says which terms matched and which words were
added (`matched`, `added`, `offered`, `capped`; the recall event carries
the counts). The list is bounded — 2,000 groups of up to 16 terms of up
to 64 characters, at most 12 words added to one query, the rest left
out with `capped: true` — and a list over a bound is refused, not cut.

The rule for choosing is stated rather than implied: the setting that
answered most wins; a tie goes to the quicker; and a change that only
matches the default is no change at all, so the default stands and the
answer says "nothing measured better than the defaults". Every
difference is given in questions as well as in rate, because a rate
hides how few questions are behind it — and that is not hypothetical.
On 30 questions of `longmemeval_s`, a candidate limit of 100 measured
0.900 against the defaults' 0.833 and was taken; on 60 questions of the
same file the same setting measured **below** the defaults. The
difference the first sweep took was two questions.
[The run is recorded](../../../bench-runs/retrieval-tuning-2026-09-11/results.md). What comes out
is a recommendation with its measurement attached and the environment
lines that put it in force — nothing is written anywhere, and no engine
reads a tuning file behind anyone's back. It uses its own in-process
stores per item, so the configured store is neither read nor written.

## Scoring an answer by what it means, with a threshold from its own embedder

Exact match and token F1 score a correct paraphrase as wrong: a one-word
reference restated in a sentence gets nearly nothing.
`bench.answer_similarity.answer_similarity` gives the answer's best cosine to
any of its references. A pass needs a `SimilarityThreshold`, which carries the
embedder id and width it was measured with; with any other embedder the
result keeps its score, decides nothing, and says why, because a cosine's
scale belongs to the model that made it. The reference framework passes at a
fixed 0.8 for every embedder.

`measure_threshold` takes the threshold without a judge. The scores of
answers that match their reference exactly are the passes it must allow; the
scores of answers set against another question's reference are the passes it
must refuse; the threshold is the lowest that passes at most
`target_false_pass` (5% by default) of those, with the matched pass rate
beside it. A threshold that passes no matched answer is not taken. A pass is
a score above the threshold. The measurement over the public QA rows with
the local embedder has not been run yet.

## Measuring on a BEIR dataset

```bash
scone bench-beir datasets/scifact --split test --k 1,3,10 --queries 300 --seed 1 --json
```

Every other bench here reads a dataset this project shaped. BEIR is the
ground retrieval systems are compared on: a directory holding
`corpus.jsonl`, `queries.jsonl` and `qrels/<split>.tsv`, relevance judged
in grades. `bench-beir` reads those files itself, with no BEIR package:
any split, not only `test`. A malformed judgement is refused with its
line number, and a judgement naming a document or query the files do not
hold is counted in the report, never dropped in silence. Queries nobody
judged are counted and not run.

The corpus goes into a fresh in-process memory, one episode per document
under `beir:<id>`, so the configured store is neither read nor written.
Each judged query is recalled, and the passages returned are mapped back
to documents in rank order, a document chunked many times counted once at
its best rank. The scores are graded nDCG, with gain equal to the grade
as trec_eval computes it and the ideal the judged grades in their best
order; recall and precision of documents judged relevant (grade above 0);
and reciprocal rank at the largest k. Every per-query ranking and score is
in the JSON.

Recall takes a query of at most 1,000 characters, and argument-retrieval
sets hold whole paragraphs as queries. A longer query is recalled cut at
the last space within the limit, marked `"cut": true` in its per-query
entry and counted in `queries_cut`. A query with no text retrieves
nothing, scores zero and is counted in `queries_empty`. Neither stops the
run after the corpus is stored, and both stay in the averages, so a run
over such a set says how many of its queries it asked as written.

`--queries` runs that many judged queries chosen by `--seed`.
`--max-documents` stores at most that many documents, keeping every
document judged for the queries run and filling the rest in file order.
A cut corpus has fewer distractors and scores higher, so the report says
it was cut and by how much. The embedder is the one the report names:
with the default hashing embedder the numbers measure lexical overlap,
not a semantic model.

## Trying a parked record again, on purpose

A record the extractor keeps failing on is **parked** after
`SCONE_DISTILL_MAX_ATTEMPTS` tries, and reported as failed without
another model call. That is right: a poisoned record should not burn a
call every pass forever.

But the park lives in the **running process**. Until now the only way to
try a parked record again was to restart the server — which un-parks
*everything*, including the records there was every reason to leave
alone. So there is a deliberate version:

```bash
curl -XPOST $SCONE/v1/consolidate/retry -H "$AUTH" -d '{"episodes": [412]}'
# {"cleared": 1, "unparked": 1, "unknown": 0, "asked": 1, "parked_now": 3}
```

`POST /v1/consolidate/retry` (capability `consolidation.retry`, present
only when a consolidation worker is configured). With no `episodes` it
retries every failure the running distiller holds for the space.

- **`cleared` and `unparked` are counted apart.** A record with one
  failure of five against it is not the same as one that has been given
  up on, and a single number for both would hide which happened.
- **The durable attempt count is not reset.** A retry that works still
  shows it took two goes, which is what the job item's `attempts` is for.
- **An unknown id is reported, not refused.** A record may have succeeded
  since it last failed, so `unknown` counts ids with nothing recorded
  against them rather than failing the whole call.
- **`parked_now` is what is left**, so a caller can tell "I cleared the
  one I named" from "I cleared the lot".
- **`queued` and `blocked` are counted apart from `cleared`.** A pass only
  considers episodes no claim cites yet, so an episode whose extraction
  failed *after* writing one fact is never looked at again: clearing its
  park is real and nothing follows from it. Reporting that as `cleared`
  alone would read as "queued", which is a promise this cannot keep.
- **Episode ids are strict integers.** Pydantic's default would coerce
  `true`, `"1"` and `1.0` all to the integer 1, so a caller could clear
  episode 1 without ever naming it.

There is deliberately **no `scone retry`**. The park is in the process
that holds it, and a command-line invocation is a *new* process with
nothing parked in it — a CLI retry would report success and do nothing.
`scone distill` already retries everything, for the same reason.

## Keeping a space in step with a directory

`map` remembers the files under a directory, notices when it has seen
one before, and updates one that has **changed**. What it cannot notice
is that a file is **gone**, and it never plans before writing — those are
the difference between an import you run once and a sync you run on a
schedule. The two share one identity for a file, so either can follow the
other.

```bash
scone sync ~/work/notes                          # a plan: nothing is written
# would sync /Users/me/work/notes: 34 of 34 file(s) read; 2 added, 1 updated,
# 31 unchanged; 1 file(s) are gone from disk but not removed from memory; pass
# remove to forget them
# last sync: 2026-09-11T22:04:11Z (34 added, 0 updated)

scone sync ~/work/notes --apply                  # writes the added and changed
scone sync ~/work/notes --apply --remove         # also forgets what is gone
```

An unchanged file is **not a write**: the space's revision does not move,
so a sync on a timer does not churn the store. A changed file is an
update through the engine's keyed `replace`, so a source that changed
leaves **one** memory and not two — and with `SCONE_EMBEDDING_CACHE` set,
only the chunks whose text changed reach the embedder; the receipt's
`embeddings_reused` counts the rest (see [file ingestion](file-ingestion.md#reusing-embeddings-across-updates)). A sync
that reads code reads the
project's manifests too (`pyproject.toml`, `package.json`, `Cargo.toml`,
`go.mod`, `requirements*.txt` and `requirements/*.txt`, and the rest
`map` knows), whatever their suffix, judged by their path below the
root as `map` judges them, so with the code graph on (`code_graph=True`;
the command line records claims only under `map --graph`) a scheduled
sync keeps what the project depends on as current as what it defines; a
sync of notes alone (`--suffix .md`) leaves them, and a manifest that
falls out of a narrowed sync's scope is out of scope, not missing. The
receipt's `files_found` counts them with the files the suffixes chose.

**What the tree says not to read is left unread.** A repository walked
whole is a repository with its `node_modules`, `build`, `dist`, `target`
and `vendor` in it: thousands of files nobody wrote, embedded and put in
the graph ahead of the source. So `sync` and `map` read every
`.gitignore` below the root and leave what it excludes unread, with
git's own rules (`ingestion/ignore.py`, written here rather than
borrowed): a blank line or `#` comment says nothing; `!` re-includes; a
trailing `/` matches only a directory; a pattern with a slash anywhere
but its end is anchored to its file's directory and one without matches
at any depth below it; `*` and `?` never cross a slash and `**` does;
the last matching pattern wins, and a deeper file's patterns come after
a shallower one's. A `.sconeignore` in any directory is read after the
`.gitignore` beside it and can only exclude more: what `.gitignore`
excludes stays excluded whatever it says, and a file under an excluded
directory is never re-included, as in git. The receipt says what the
rules did (`ignored`, `ignored_directories`, `ignore_files`); at most
500 files and 20,000 patterns are read, and when that bound bit the
receipt says so (`ignore_truncated`), since a run past it may have read
what the tree said not to; a pattern that cannot be read is passed over
and named (`ignore_unusable`). A memory `sync` holds for a file the rules
now exclude is not a file that is gone: it is left alone, neither read
nor removed, and counted (`ignored_memories`). `--no-ignore` reads the
tree whole. Dot-named directories and files and `__pycache__` are never
walked, rules or no rules, and a symbolic link is left alone and counted
(`links`): what it points at is outside the root. One thing that is
git's and not here: git matches case-insensitively where
`core.ignorecase` is set, as it is on a Mac's default file system; these
rules match as written.

### Deletion is opt-in, previewed, and refused when the path looks wrong

Forgetting memory because a file is missing is destructive, and a
directory can be missing for reasons that have nothing to do with
intent: an unmounted volume, a half-finished checkout, a typo. So there
are three separate guards, each with its own test:

- **Nothing is written without `--apply`.** The default is a plan.
- **Nothing is forgotten without `--remove`** *as well*. An ordinary
  sync reports what is gone and leaves it alone, because a caller who
  has not thought about deletion should not get it.
- **An empty directory is refused outright.** If the walk found no files
  and the marker holds memories, the sync stops before forgetting
  anything and says so. A repository whose every file was deleted is far
  rarer than a wrong path.

Two further guards are about honesty rather than intent, and both exist
because a file can stop appearing in the walk for reasons that have
nothing to do with the disk:

- If the walk stops at the **file cap** it has not seen the whole
  directory, so it cannot tell a file that is gone from one it never
  reached. Such a sync reports `checked_for_missing: false` and forgets
  nothing — without that, lowering `--limit` would silently delete
  memory, and `removed: 0` would read as "nothing is gone" when it means
  "we did not look".
- If this run's **`--suffix` list** no longer selects a file the marker
  holds, that file was never looked for. It is counted as `out_of_scope`
  and left alone, never as missing. Otherwise narrowing a flag between
  two runs would delete every memory the narrower run stopped asking
  about.
- If a **directory could not be read**, an unknown number of files are
  hidden behind it. `rglob` swallows a `PermissionError` and returns what
  it could reach, which is indistinguishable from a smaller directory —
  so the walk is explicit, counts what it could not open, and a run with
  any `unreadable` forgets nothing and says why.

**Only ordinary files are opened.** A named pipe matching the suffixes
would block on open until somebody wrote to it, and a sync that hangs is
worse than one that counts wrongly: nothing reports it and nothing
recovers. Pipes, sockets and devices are counted as `special` and left
shut.

Two more things the walk does not do, both because the root is the whole
scope. **A symbolic link is counted and not followed**: what it points at
is outside the root the caller named, and storing it would file content
nobody asked for under a path inside the root, so nothing in the space
would say where it came from. And a file longer than `--max-bytes` is
**read only to the limit plus one byte** — enough to know it is longer,
without reading a gigabyte to keep a kilobyte.

An earlier version got the hidden-directory rule wrong in a way worth
recording, because the report it produced was confident and false. The
rule is meant to skip a repository's own `.git`, and it was judged on the
whole path rather than on the part below the root — so syncing any root
*reached through* a dot-segment (`~/.config/notes`, `~/.claude/projects`,
the checkout this is developed in) excluded the entire tree. `files_found`
came back 0 with every file on disk, `checked_for_missing` was `true`
because `0 == 0`, and the receipt said every memory the marker held was
gone from disk and offered to forget it. The guard above refused the
deletion, which is the only reason it was not data loss.

The rule has one stated exception. An MCP configuration lives in a
dot-name by every tool's convention (`.mcp.json`, `.vscode/mcp.json`,
`.cursor/mcp.json`, `.gemini/settings.json`, `.codex/config.toml`), so
those files and those four directories are walked, and from the
directories only the configuration is read; see *What a project hands
its agents* above.

### The marker is a name, not a path

Every episode a sync writes carries its marker in metadata, so "what did
the last sync of this directory leave here" has an exact answer rather
than a guess from path prefixes. `--marker` lets a directory be moved or
renamed without losing what it stored.

The **identity** a file is stored under carries the marker too, and has
to: keyed on the relative path alone, two directories synced into one
space would share an identity for every filename they had in common —
`README.md` and `README.md` — and the second sync's `replace` would
forget the first's episode to store its own. No `--remove`, nothing in
the receipt, memory gone. The marker's length precedes it in the key, so
no marker and path can be read two ways; a separator alone could be,
since both halves are text a caller chose.

`last sync` comes from the event log, and only an **applied** sync is
recorded — a plan succeeded at nothing. A store with no event log raises
rather than answering "never", because "nothing keeps the record" and
"it has never run" are different facts.

### Not on the HTTP surface

`scone sync` reads whatever local directory it is pointed at. Exposing
that over HTTP would let an API caller choose which of the server's
directories to read, so it is a command-line operation only. The
sandboxed memory filesystem below is the HTTP-facing story for paths.

## Memory as a tree of paths

An agent that can list and read paths can explore a space without being
taught an API for every kind of thing in it. A tree is opened for one
space:

```
/episodes/12.md          one episode, exactly as it was stored
/facts/alice%20chen.md   every claim about a subject, each citing its fact
/entities/acme.md        an entity: its relations, what follows, its values
/notes/plan.md           a note an agent wrote
```

It holds nothing of its own. Every path resolves to something the engine
already has, and reading one changes nothing — which is the only reason a
filesystem is a safe shape for memory rather than a second place where
things are true.

- **The space is not in the path.** A tree is opened for one space, so
  there is no path that could name another and nothing to escape from.
  `..` is refused rather than resolved.
- **Names are encoded, not cleaned.** A subject called `a/b c` is a real
  subject; its file is `a%2Fb%20c.md`, and two different names never
  become one file.
- **A note is an episode.** Writing `/notes/plan.md` remembers an episode
  whose source is `fs:/notes/plan.md`, so it is recalled, distilled,
  forgotten and exported like anything else. Writing it again supersedes
  it and keeps what was there; nothing stored is rewritten.
- **Nothing is writable unless the owner says so**
  (`FilesystemPolicy(writable=True)`), and then only under `/notes`.
- **A write that would land on top of a newer one is refused**, not
  resolved: `write(..., if_version=...)` takes the note's own version, as
  a read reports it. A space revision would not do — it moves whenever
  anything at all is written, so it would refuse writers who conflicted
  with nobody.
- **`search`** answers a query in paths, using the engine's ordinary
  recall underneath: a passage written as a note is answered at its note
  path, and a matching claim at the page of its subject.
- **A listing says how many there are, not how many it read.** The tree
  reads a bounded number of episodes; past that a listing reports the
  real total and says in `capped` how many it looked at, so a reader who
  pages to the end is not left thinking they saw everything.
- **At the command line**: `scone fs ls /`, `scone fs cat
  /episodes/1.md`, `scone fs find "desks"`, and `scone fs write
  /notes/plan.md --writable` reading the note from stdin. Writing needs
  the flag that says so, so a mistyped path cannot write where somebody
  only meant to look.
- **Over HTTP**: `GET /v1/fs`, `GET /v1/fs/read`, `GET /v1/fs/search` and
  `POST /v1/fs/notes`, with the refusals keeping their meanings — a path
  that cannot mean anything is a 422, a write to a tree nobody made
  writable is a 403, and a write onto a note that moved is a 409 carrying
  the version it now stands at. `create_app(..., filesystem=...)` decides,
  and `/v1/capabilities` says so as `filesystem.read` and
  `filesystem.write`. A read-only key is refused by the key's own role
  before the tree is asked.
- **Every action is recorded** where the space keeps its events —
  `filesystem.list`, `.read`, `.write`, `.search` and `.refused` — each
  carrying the path under one key, so an audit reads without knowing
  which action it was. What was searched for is not recorded: an audit is
  for what was done, not for what was wondered.

## Measured and not shipped

Two cheap ideas for better recall were measured on this machine and left
out, because they did not earn their place. Both are written down so the
next person does not spend the day finding out again.

- **Restricting recall to the dates a question names.** On the 40 dated
  questions of `bench-data/temporal-40.json`, the session the expected
  answer rests on reaches the top five for 36 of 40 either way, and the
  two sets differ: an event is often told on a day other than the one the
  question names. The days bound the search only for a question about a
  day, where the day is what is being asked for.
- **Splitting a question into parts and fusing the searches.** On 40
  stratified questions of `bench-data/longmemeval_s.json`, 12 split at
  all, and fusing changed nothing at five (36 either way) while finding
  every expected session for one more question. On 40 multi-session
  questions, where it should help most, 6 split and it gained one
  question at five. A question people ask memory is usually one clause,
  and splitting it on conjunctions mostly finds the same passages twice.

What both measurements say is that the lever here is the embedder, not
the question: with a hash embedder the lexical lane is doing the work,
and cleverness around the query does not add to it.

## Taking back a review decision

A review is a person's judgement, and people are wrong sometimes. A
system that records judgements with no way to revise them teaches its
users not to judge, so each one can be taken back:

| Decision | Taken back by | Refused when |
| --- | --- | --- |
| `exclude` | `include` | — |
| `decline` | `reconsider` | the claim was never declined |
| `close` (by hand) | `reopen` | the claim holds, or another claim superseded it |

A claim another claim superseded is not reopened behind that claim's
back: both would hold at once, one saying the other is wrong. The refusal
names the claim that superseded it, so a person can decide what they
actually meant.

Nothing is erased. The decision taken back stays in the event log with
its reason, and the one that takes it back is recorded beside it, so the
history of what people decided reads in full. Both are review decisions,
so a key with the write role cannot make them and one with the review
role can. `POST /v1/facts/{id}/reconsider`, `POST /v1/facts/{id}/reopen`,
`scone reconsider`, `scone reopen`.

## Two claims that begin at the same instant

Valid time cannot separate them, so something else does: the order they
were recorded in. That is a fact about the recording, not about the
world, and a reader who cannot see the difference will read arrival
order as history.

- The closed one says **"superseded at the same instant by fact 2, which
  was recorded later"**, which reads differently from an ordinary
  supersession because it means something different: the claim never held
  for any length of time.
- `graph health` counts them as **`contested_instant`**, with the subject,
  the predicate, the instant and both claims, and points at `scone facts`.
  The ledger cannot decide which is right; a person can.
- Saying the same thing twice at one moment is agreement, not a
  collision, and is not counted. Nor is a many-valued predicate
  (`calls`, `imports`, `depends_on`, or one named in `SCONE_MANY_VALUED`):
  its values hold side by side by design, so a file that imports two
  modules on one line is not contested. Counted, they made a code graph
  read as thousands of collisions.

## Moving one space into another

```bash
scone --space old merge-space --into new --dry-run
# would move 412 episode(s) and 1,308 claim(s) from old into new
scone --space old merge-space --into new --confirm old
```

A merge is not a new kind of write: it is the archive read out of one
space and into another, so everything that makes an import honest holds —
identity is re-derived for the space it lands in, what was forgotten
there stays forgotten, and claims arrive with their history.

- **It previews.** `--dry-run` (or `"preview": true` on
  `POST /v1/spaces/{name}/merge`) says what would move and moves nothing;
  the receipt's `moved` says which it was, so a preview and a deed are
  never mistaken for each other.
- **It needs the name said out loud**, as deleting a space does.
- **It takes the same permission as a deletion**, not an ordinary write:
  moving a whole space away is as final as removing it.
- **Retained attachments move too**, including unlinked holds. Preview and
  completion receipts report attachment counts/bytes, skipped forgotten sources
  and omitted references to known forgotten sources.
- **The source closes after evidence verification.** Quiesce source and destination
  writers throughout the operation. Separate storage observations do not provide
  an atomic cutover. See [space merge](space-merge.md) for failure/retry behavior
  and the ledger and runtime state this operation does not preserve.

## What an archive says it is

For verified linked attachment bytes and remapped episode links, use the opt-in
[attachment archive profile](attachment-archives.md). The default below remains
compatible with existing text archives.

`scone export` writes a header first:

```json
{"type": "archive", "profile": "scone.archive/1", "space": "alpha", "wrote_at": "…"}
```

An archive that does not name its own shape can only be read by guessing,
and a reader that guesses will one day drop something and call it a
success. So:

- **A profile this engine does not know is refused**, naming it, rather
  than read hopefully. An archive with no header at all is read as the
  first profile, which is what archives written before the header are.
- **A record carrying a field this engine cannot keep is refused**, naming
  every field it did not know. Importing the part we recognise would look
  like a success and quietly lose the rest, which is the one failure an
  archive must not have. The fields each record may carry come from the
  models the exporter writes from, so the list cannot go stale and refuse
  an archive this engine could have read perfectly well.
- **An unknown record type is refused**, as it always was.
- **The header says what the profile does not carry.** This one carries
  episodes, facts, links and restatements; a space's attachments are not
  in it, so a dump of an illustrated space says
  `"not_carried": {"attachments": 3}` rather than letting a reader take
  it for the whole of the space. Carrying them is a later profile's job,
  and it will be a later profile, refused by this engine until it
  understands it.
- `ImportSummary.profile` says which profile an import was read as.

## A batch where one record is wrong

One bad record refuses the whole batch. That is the right default: a
caller who sent one usually wants to fix it and send the lot again, and a
half-stored batch nobody asked for is worse than a clear refusal.

A caller importing from somewhere messy wants the opposite, and can ask
for it — `remember_many(..., partial=True)`, or `"partial": true` in the
body of `POST /v1/episodes/batch`. Each record is then judged on its own:
one that cannot be stored comes back as `outcome: "failed"` with the
`reason`, the rest are stored, and the answers stay in the order they
were sent so a caller can line them up against what they sent. The
records that do pass are still stored together, so they deduplicate
against each other the way a batch does.

A reader of the old shape sees the old shape: `failed` appears in the
counts only for a caller who asked for a partial batch, which is the only
caller who can get one.

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
- **What follows** from the relations, when the space says what its
  predicates mean. Its own kind of item, never a relation (see below).
- **Attributes** are value claims grouped by `(entity, predicate, exact value)`.
  Values keep their exact text: `3 MB` and `3 mb` are two attributes.
- **Names** are the entity's recorded spellings. The label is the most
  common one, and casing is recovered from the source quote, so `alice chen`
  is labelled `Alice Chen`. A code name (a path, or a path and a
  declaration) is shown as the code declares it, whatever the count: a
  declaration is called by its lowercased key once per call it receives,
  and a busy method would otherwise be labelled `directorysync._finish`
  beside a quiet one labelled `DirectorySync.open`.
- **Kinds** (person, organisation, place, project, product, event, concept;
  and for a code graph file, declaration, module)
  are inferred hints from the predicates around an entity. They carry
  `kind_status: "inferred"` and list the fact ids that suggested them. Hints
  that disagree give `"conflict"` and no kind, never a guess. A code
  graph's entities are hinted by their shape, since a predicate alone
  cannot tell a file from the class it holds: a name whose last segment
  carries a suffix a reader knows is a **file** (`pkg/a.py`, `README.md`,
  `pyproject.toml`); a file, a colon and a name is a **declaration**
  (`pkg/a.py:Thing.run`); a name with neither that a file imports is a
  **module** (`typing`, `github.com/gorilla/mux`); a module a manifest
  also depends on is the package it comes from, so it is a **product**
  rather than a conflict. A ratio, a time or a URL is none of these.
  Before this (`kinds/1`) every code entity was `kind_unknown`, which
  made the report's Kind column empty and `graph health` count a whole
  codebase as entities nothing says the kind of.

### What a predicate means, and what follows from it

A ledger holds what was said. "Alice Chen works at Acme Robotics" says
nothing, by itself, about what Acme employs, whether Bob is married to
Alice when Alice is married to Bob, or where a shelf is when its aisle is
in a warehouse. A space can say what three of its predicates mean, and
the projection works out the rest:

```bash
SCONE_RELATION_INVERSE=works_at:employs,wrote:written_by
SCONE_RELATION_SYMMETRIC=married_to,colleague_of
SCONE_RELATION_TRANSITIVE=part_of,located_in
```

Configured, never guessed: nothing here decides that two predicates are
opposites because they look alike. A vocabulary that cannot mean what it
says is refused where it is built — a predicate cannot be its own
opposite (that is what symmetric means), cannot have two opposites, and
cannot be both symmetric and have another side.

What follows is kept apart from what was said, everywhere it travels:

- It is an `Implied`, not a `Relation`, so nothing can pass one where the
  other is expected, and `paths_between`, the export and retrieval walk
  the stated relations as before.
- Its id is an `imp:` id under its own scheme, so an implication joining
  the same two things by the same predicate can never be read as the
  claim.
- It names the facts under it (`fact_ids`) and the relations it was
  worked out from (`follows_from`), so a reader can check the sources.
- It holds only over the stretches of valid time its claims actually
  shared, worked out from the claims themselves. A relation made of two
  spells with a gap between them does not hold during the gap, so a chain
  through it does not either: `periods` lists every stretch it held over,
  and `first_valid_from` and `last_valid_until` are the first beginning
  and the last ending of those. A chain whose legs never shared a moment
  is not recorded at all.
- It is no better grounded than its least grounded link — one unsourced
  leg makes an unsourced chain. Exclude a leg and the chain goes; its
  neighbour stands.

Nothing already said is implied, nothing is implied twice, nothing is
implied about a thing and itself, and a ring goes round once.

Three bounds, all reported rather than assumed. A chain is followed four
claims at most (`max_steps`); a projection holds at most 50,000
implications (`max_implied`); and a whole walk examines at most 200,000
claims (`max_walked`), because a dense graph has more paths than anyone
can walk. Each thing is also reached by the shortest route from a claim
and never expanded again, which is what keeps the walk to the claims
rather than the paths; the cost of that is stated plainly: a longer
route that would hold over a stretch the shortest does not is not
searched for. `coverage.meanings` names the vocabulary and all three
bounds, and `coverage.implied_capped` says when a walk stopped early.

A structured question can be asked over it too. `graph match --follows`
(HTTP `follows=true`, tool parameter `follows`) matches what follows as
well as what was claimed, so `?who employs ?whom` answers from "Alice
works at Acme". It is off by default, because a question about the graph
is a question about what was said unless it says otherwise; a row that
used one says which meaning it followed, and cites the claims underneath,
which are re-read like any others. A chain needs **every** link: if one
claim under it stops counting between the match and the answer, the row
goes rather than standing on the link that survived, and a chain is
joined to other patterns only over the time its own legs shared.

Where to see it: `implied` in the knowledge view, `follows` on an
entity's page, `follows:` lines in the packet `graph context` writes
for a model, and `follows` on a matched row, each naming what it
followed from. `scone graph meanings`
prints the vocabulary this process is running with, and the bounds it
keeps; it says "as this process is configured", because a vocabulary is
an engine option and two processes reading one store can disagree about
what follows.

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
  "projection": {"version": "scone.entities/1", "classifier": "objects/1", "kinds": "kinds/2",
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
  cohesion, kinds and predicates. A community of files is named by the
  directory they share, then its central members with that directory
  left off (`scone_memory/ingestion/ · code_graph.py · manifests.py`):
  when at least two members are paths, most are, and every path sits
  under one directory. Three file names are not what a person calls a
  subsystem; its directory is.
  Before communities are named, two guards look again at the
  communities a reader cannot use, each re-partitioning one community on
  its own links at the same resolution:
  - one holding over a quarter of the analysed entities (and at least 10)
    is split into what its links give, since a map by community draws it
    as one blob;
  - one of 50 or more is split when its own partition reaches modularity
    0.3, because over a large graph modularity merges small modules into
    one community that holds several. The share of member pairs linked is
    not the test: in a sparse graph it falls with size, and on this
    project's own code graph it fired on 12 of the 17 communities of 50
    or more.

  The resolution is never raised to force a split. A community a guard
  looked at and kept whole (one piece, or a weak split of a large one) is
  counted in `unsplittable`, and splits in `split_oversized` and
  `split_nested`. The pieces of a split are looked at again, like any
  community, until none splits: a community's own partition can leave a
  piece that still holds two. Finer communities often score lower on
  modularity, so `modularity_before_guards` gives the modularity before
  any split beside the final `modularity`, both over the graph's own
  entities (externals and held-apart hubs left out, detached hubs
  rejoined). Measured on an
  earlier base, before externals were kept out of the partition, on this
  project's retrieval, entities, API and ingestion code (2,713 entities):
  28 communities became 158, the largest three (341, 205 and 199 entities)
  became 62, 48 and 42, and modularity went from 0.702 to 0.584, with 19
  nested splits and one community kept whole.
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
- `exclude_hubs` (a degree percentile, 50 to 100; `--exclude-hubs` on
  the command line) holds the graph's own entities whose number of
  neighbours is above that percentile (nearest rank, among the graph's
  own) apart, the way externals are held apart without asking: out of
  the community partition, the community names, the surprising
  connections and the counts of links between communities, and out of
  the central ranking; each is attached for reading to the community it
  links most (one whose every neighbour is itself held apart stands
  alone, as an external does), and listed under `hubs_excluded` with
  that community. A base class every file inherits, or a person every
  note mentions, otherwise glues every community into one and leads
  every ranking. The coverage says how many were held apart
  (`hubs_held_apart`), and the Markdown lists them under "Hubs held
  apart". A report built over an analysis that was not run with the
  percentile ranks the same hubs out by the same rule and says the
  partition was found with them in it.
- `detach_hubs` (a degree percentile, 50 to 100; CLI `--detach-hubs`)
  leaves those entities out while communities are found, so an entity
  everything links to does not pull unrelated groups into one; each then
  joins the community most of its link weight goes to as a full member,
  so, unlike a hub held apart by `exclude_hubs`, it names communities,
  counts in their links and in modularity, and can be a surprising
  connection. `hubs_detached` counts them. Removing hubs can leave
  groups with no link between them, so expect more communities. Hubs
  are picked among the graph's own entities, by their links to each
  other, once externals (below) and any hubs held apart by `exclude_hubs`
  are out; the guards above run on the partition found without them.

**What the graph names and never reads is kept apart, without asking.**
In a code graph every file imports `typing`, so `typing` was the most
central entity of the codebase, the strongest tie between any two
communities, a "surprising connection" from each, and the middle of
every drawing; `pydantic.BaseModel`, `json` and a cited `ADR-12` were
close behind. An entity that is the object of `imports`,
`imports_when_called`, `imports_for_types`, `depends_on`, `develops_with`,
`cites`, `uses_type` or `references` and the subject of nothing at all is
*external*: named here, read nowhere here (a module of
the codebase is imported too, but it also defines its own things, so it
is the graph's own). Externals are left out of the community partition
(so modularity is the codebase's), attached for reading to the community
that names each most (by the weight of what names it; one whose namers
were all cut by the entity budget stands alone), kept out of the central
ranking, bridging entities, participation, surprising connections and
the counts of links between communities, and listed apart under
`external_dependencies` ("Named but never read" in the Markdown), by how
many things name them. A community's `members` list its own members and
then its attached externals; its link counts and cohesion are of its own.
The drawing spends its room on the graph's own entities first. The
coverage says how many were set apart (`external_entities`); a graph of
people and places has none.

The report echoes these under `analysis`, and `analysis.coverage` carries
`resolution`, `external_entities` and what the guards and `detach_hubs`
did.

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

#### `GET /v1/graph/health`

What in the graph wants attention, counted with examples. A knowledge
graph goes wrong quietly, and each way it does is countable. Advertised
as `graph.health`; `scone graph health`, the MCP tool
`memory_graph_health` and the ToolBox tool `graph_health` give the same
answer. It reads and changes nothing.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `limit` | 10 (1–100) | Examples shown for each concern |
| `max_bytes` | 8,000 (512–64,000) | Byte budget for the text |
| `status`, `as_of` | `current`, now | Which facts count, and when |

- **ungrounded**: claims that name a source but cannot be checked against
  it, because the source is gone, the claim has no quote, or the quote is
  not in the source. Grounding is checked against the sources kept now,
  not against what the projection recorded when the claim was written, so
  a claim whose source was forgotten reads as `quote_source_missing`. At
  most 500 claims are checked that way; past that the projection's record
  is used and the coverage says `grounding_checked N of M`.
- **unsourced**: claims nothing cites a source for. They rest on whoever
  wrote them, which is how a claim asserted by hand is meant to work, so
  they are counted apart from claims whose checks failed.
- **contested_kind**: entities whose kind hints disagree, so they have no
  kind. **kind_unknown**: entities nothing implies a kind for.
- **unconnected**: entities nothing links to and that link to nothing.
- **thin_predicate**: predicates exactly one claim uses, which is what a
  bad extraction looks like.
- **likely_duplicate**: pairs from `/v1/entities/duplicates` at its
  default score, carried in so one read shows everything; the detail
  stays on that route, and anything it cut is said here with a
  `duplicates:` prefix.
- Each concern says **where to see the whole of it**: the grounding audit
  for ungrounded claims, the entity page for kinds, `/v1/graph/knowledge`
  for unconnected entities, `/v1/graph/schema` for thin predicates and
  `/v1/entities/duplicates` for the pairs. A count is only useful beside
  the place that shows all of it.
- A graph with none of these says `nothing to fix`, and a capped read
  says `among the facts read` instead of claiming the whole space.
- **One answer, one revision.** The totals, the grounding checks and the
  duplicate pairs are all read at one revision; the answer says which. A
  ledger that changed while it was read is read again, and one that keeps
  moving is said (`ledger_moved_during_read`) rather than answered from
  two different moments.

#### `GET /v1/graph/cycles`

The dependency cycles a space's code graph holds. A cycle is the one
shape a dependency graph should not have, and the one nobody sees
reading files one at a time; it is also the shape most codebases work
around rather than remove, by writing an import inside a function (it
runs when the function is called) or under `if TYPE_CHECKING:` (it never
runs). Counting every `imports` fact would report every one of those
workarounds as a cycle, so the Python reader records when an import runs
-- `imports` at load, `imports_when_called` in a function body,
`imports_for_types` under the type-checking guard -- and this route
reads two things apart: the groups of files that cannot load without
each other (the strongly connected parts of `imports` and `depends_on`),
each with one shortest loop and the fact ids behind every hop; and the
groups that join only once the deferred imports are counted, each naming
the deferred facts that hold it open. Advertised as `graph.cycles`;
`scone graph cycles`, the MCP tool `memory_graph_cycles` and the ToolBox
tool `graph_cycles` give the same answer. It reads and changes nothing.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `limit` | 20 (1–100) | Groups shown of each kind; the rest are counted |
| `max_bytes` | 8,000 (512–64,000) | Byte budget for the text |
| `status`, `as_of` | `current`, now | Which facts count, and when |

- `status` is `cycles` when a load-time loop exists, `held_apart` when
  only deferred loops do, and `none` otherwise. `totals` counts both
  kinds whole; `cycles` and `held_apart` list up to `limit` of each,
  largest first, and `coverage` says what was cut.
- The loop shown for a group held apart crosses one of its deferred
  imports, so it shows what the group is about rather than a load-time
  cycle inside it. A loop is walked up to 32 hops; a group whose loop is
  longer is still counted and listed, its example left empty, and the
  coverage says `loops_over_bound N`.
- A self-import is not a cycle, and a file importing a module the space
  has no file for (`json`) is a leaf. Other languages' readers record
  every import as `imports`, so their function-level imports count as
  load-time until the reader tells them apart; the answer does not guess.
- A projection over 50,000 entities is declined with
  `entities_over_bound` in the coverage rather than answered slowly.
- **One answer, one revision.** The components and the loops are read at
  one revision; a ledger that keeps moving is said
  (`ledger_moved_during_read`) rather than answered from two moments.

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

#### `POST /v1/entities/merges`, `POST /v1/entities/merges/close`, `GET /v1/entities/merges`

Record that two names are one entity, and undo it. A duplicate suggestion
is evidence; a merge is the decision a person makes on it, so both writes
belong to the `review` role, beside approving and declining claims, and a
`write` key is refused.

- **The decision is a ledger claim.** `{"alias", "into", "reason"}` stores
  `alias` under the reserved predicate `scone:same entity` with `into` as
  its object, valid from now. The reason and the actor (the key's
  fingerprint and `X-Scone-Actor`) go on the `entity_merge` event. No
  assertion may use the predicate, whether stated, proposed or extracted,
  so no document and no model can merge two entities; a document saying
  two names are one is a claim to weigh under a predicate of its own.
- **One target per alias.** The predicate holds one value at a time, so
  merging an alias somewhere new closes the decision it replaces.
- **Refused before anything is written** (422): a name that cannot name
  one thing (prose, a quotation, a pronoun), two names that already share
  a key, an empty reason, and a merge whose target already resolves back
  to its alias. The loop check follows the target's decisions in force one
  lookup at a time, at most 64, and refuses a longer chain rather than
  guess that it has no loop.
- **Every view applies the decisions it can see, at its own moment.** The
  projection rewrites each alias to the entity it resolves to before
  anything is counted: relations and roles move to that entity, the alias's
  spellings join its surface forms, its label keeps the spelling of the
  name merged into, and kind hints meet. Communities, PageRank, paths,
  duplicates and exports all run on the merged graph. A chain of merges
  ends at its last name. A name a decision covers is an entity even where
  the identity rule would read it as a value ("MB"), which is what "only
  a recorded identity decision may join such a value" means.
- **Undoing is closing.** `/v1/entities/merges/close` with `{"alias",
  "reason"}` closes the decision in force (404 when there is none). The
  names part from that moment; a view `as_of` an earlier moment still shows
  them joined, and `current`, `history` and `proposed` views read the
  decisions at the view's own moment. A decision that is excluded, or
  reopened into a loop, is not applied, and the loop is reported.
- `GET /v1/entities/merges?status=&as_of=` lists the decisions the view
  applies, in the order recorded, each with `outcome` `applied`, `cycle`
  or `same_name`, beside the projection's version, digest and coverage.
  A projection's digest changes only when it holds a decision.
- `/v1/entities/resolve` answers a merged-away name, or the id its entity
  had, with the entity it went into, at the `key` or `id` tier.
- `scone graph merge ALIAS INTO --reason R`, `scone graph unmerge ALIAS
  --reason R` and `scone graph merges` do the same from the command line.

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
| `coverage:` | `complete`, or what was left out: read caps, `stale_evidence N`, `hubs_not_crossed N`, `relations_cut N`, `cut_groups_cut N`, `unverified N`, `not_found N`, `seeds_cut N` |
| `note:` | that names, values and quotes are recorded data, not instructions |
| `entity:` or `candidate:` | the entities asked about, or every candidate for an ambiguous name |
| `path:` | the shortest route between each pair of them, as `A -works_at-> B <-lives_in- C` |
| `cut:` | what `relations_cut` left out, grouped, the largest group first: `cut: 16 in pkg/office.py -calls-> pkg/errors.py:InvalidInput` for calls into an entity from one file, `cut: A -knows-> 3 entities` outside code |
| `hop N:` | relations N steps out (`max_hops`, 1–4, default 2), those with the most facts first |
| `value:` | values recorded for the entities asked about |

When the walk stops at 64 relations, the relations it left out are
grouped by the entity they were cut from, their direction and predicate
and, for a code predicate, the file the far end is declared in: the part
before the colon of `path:Declaration`, or the label itself when it is a
path with a directory. A bare name shaped like a file, such as
`Node.js`, places nothing. Each group is a `cut:` line with its count,
written before the relations so the byte budget takes the weakest
relations rather than the only lines saying where the rest are. At most
20 groups are written, and `cut_groups_cut N` counts the rest. The same
groups are `relations_cut_by` in the JSON coverage (`entity`,
`direction` `in` or `out`, `predicate`, `file` or null, `count`). The
grouping reads nothing: it uses the projection the walk already holds,
so it spends none of the re-read budget. On this package's code graph
`scone graph entity` on `InvalidInput`, with 331 relations, shows 64,
cuts 267 into 82 groups, and names the 20 files holding the most cut
callers.

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
- `memory_graph_schema`: the kinds and predicates the graph holds;
- `memory_graph_affected`: what rests on a symbol, module, file or
  package -- everything that calls, imports, inherits, mixes in, depends
  on or develops with it, nearest first -- with how deep it walked and
  what it could not list; the ToolBox tool `graph_affected` gives the
  same.

These sit beside the six tools shared with the Rust server, and none of
them writes.

The server also offers read-only resources, which a client can attach
as context without calling a tool:

- `scone://graph/report`: the knowledge report of the server's space,
  in Markdown;
- `scone://graph/schema`: its graph schema, as lines;
- `scone://graph/health`: what in its graph wants attention, as lines;
- `scone://{space}/graph/report`, `scone://{space}/graph/schema` and
  `scone://{space}/graph/health`: the same for any space by name.

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
| `obsidian` | zip of one Markdown note per entity and one per community, wiki-linked and tagged by community, plus `index.md`, `graph.canvas`, a canvas of the notes, and graph-view colours per community; or written into a vault a person already keeps with `scone graph export --format obsidian --into VAULT` | Obsidian and other note tools |
| `wiki` | zip of `index.md`, one article per topic and one per entity, in plain Markdown links | agents reading instead of the raw ledger |
| `mermaid` | a Mermaid flowchart of the 60 most connected entities and the relations between them, grouped by community and styled by kind | GitHub, Markdown viewers, docs |
| `svg` | a drawing of the 200 most connected entities by community, with no script | browsers, READMEs, slides, documents |
| `canvas` | JSON Canvas 1.0: a group per community, a card per entity, a labelled arrow per relation | Obsidian's canvas, other JSON Canvas tools |
| `communities` | an SVG map of the graph by community: a circle per community sized by its members, a line between communities weighted by the links joining them | a graph too large to draw entity by entity |
| `html` | one page: the drawing, search that moves the view to the entity chosen, a legend that shows or hides each community, a panel of each entity's relations and facts, zoom and pan; it fetches nothing | anyone with a browser, offline |
| `explorer` | the whole graph as one page: laid out in the browser by the page's own force simulation, coloured by community with a legend that turns each on and off, searched, hovered (the neighbourhood lit), clicked for an entity's relations and their facts, filtered by predicate, and read as a module tree (directories, files, what each defines) whose entries choose their entity; what the graph names but never reads (`typing`, a package) hidden until the legend shows it, then drawn small; up to 5,000 entities and it says what it left out; it fetches nothing | reading a codebase's graph, not a poster of it |
| `tree` | one page: the graph's code as directories, files and declarations in folds, with a filter, expand and collapse all, and a panel of what the chosen node calls, imports and defines and what does each to it, each link naming its facts; it fetches nothing | finding a declaration by where it lives in a codebase read with `scone map --graph` |

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

**The code tree export** is built from what the code readers record: a
file `defines` a declaration, a declaration defines a method, files and
declarations `import`, `call`, `inherit` and `depend on` one another, and
a document `references` a file it links; the page lists each relation
from both ends. It
places an entity only when a code relation holds it and its label is a
path or a declaration qualified by one, so a prose name shaped like a
file (`Node.js`) is not taken for one. A name with no directory is
placed only when the graph read it or it holds a declaration, so a
package a manifest depends on or a file imports (`lodash.merge`,
`socket.io`) is not a file here. The rest are counted as not in the
tree. A file or declaration that only a call, an import or a link in a
document names was not read where it is defined and is marked "not
read". A chain of
directories each holding only the next is one fold. At most 200 children
are listed under one node and 50 links in one list, and the rest are
counted where they were cut. A graph with no code gets a page that says
so.

How each format places values and escapes its own syntax:

- **The explorer** is the one format laid out by simulation: the page
  runs its own force layout (repulsion through a grid, springs along
  relations, a pull toward each community's centre) from a start seeded
  by each entity's id, so the same graph opens the same way, and settles
  after a few hundred steps; Fit, Pause and Resume are on the page. The
  drawing's room goes to the graph's own entities first (`typing` and
  `json`, imported everywhere, are drawn last and hidden by default);
  every name reaches the page as text through one JSON block, and the
  page's content security policy allows only its own style and code,
  pinned by hash. Beside the legend, a module tree lists the drawn
  codebase as a person reads it: directories made from the files'
  paths, the files in them, and what each defines nested as the code
  nests it (`Box` under `x.py`, `open` under `Box`), a declaration
  placed by its `defines` relation from its file or another declaration
  or, when none is drawn, under its file by name; a chain of definitions
  is followed at most 32 deep, and a loop or a deeper chain sits under
  the file, so nothing drawn is lost. An entry chooses its entity in the
  drawing and shows a hidden community again. A person, a package or a cited record is not
  in the tree, and a graph with no file has none. At most 200 entries
  are listed under one directory or file; the rest fold into `+N more`
  and the page's notes say how many were folded.
- **Into a vault.** `scone graph export --format obsidian --into VAULT`
  writes the same notes under `VAULT/scone/` instead of a zip, by three
  rules. A file this did not write is never written over: every note it
  writes opens with `scone_projection: <digest>` in its frontmatter and
  a folder manifest (`scone/.scone-vault.json`) lists what the last write
  left, so a file at a target path that is neither listed nor signed is
  the person's, kept, and counted under `kept_theirs` (the wiki links to
  that name reach their note, which is about the same thing). What it
  wrote last time and does not write now is removed, so a forgotten
  entity keeps no note; a person's file is never removed, and nothing
  outside `scone/` is read or touched, `.obsidian/` least of all. The
  receipt says what happened: `written`, `updated`, `unchanged`,
  `removed`, `kept_theirs` and the projection the notes carry. Writes are
  atomic per file. The canvas names its cards' notes by their path under
  the vault (`scone/entities/...`), so it opens where it is written.
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
  - Typing in the search box dims the entities whose names do not hold it
    and lists up to eight that do, names that begin with it first; a longer
    list says how many more there are. Choosing one, or pressing Enter for
    the first, moves the view to it at the current zoom and opens its
    panel. A relation in the panel moves the view to the entity at its
    other end the same way.
  - A legend lists the drawn communities in the drawing's order, each with
    its colour and how many of its entities are drawn. That is a count of
    the drawing, not the community's size: the drawing may leave entities
    out, and the header says how many. Clearing a community's box hides its
    box, its entities and every relation with an end in it; the first box
    shows or hides every community, and is half-set when only some are
    hidden. Choosing a hidden entity from the search shows its community
    again, and the search list says which matches are hidden.
  - The SVG carries what the page needs for this: each community's box and
    title, and each entity, name the community (`data-group`); each arrow
    names its relation and the communities at both ends
    (`data-relation`, `data-from-group`, `data-to-group`), loops included.
- **Recall use on the drawings.** With `usage=true` (or `usage_since`, as
  on the knowledge view; `scone graph export --usage` on the command line)
  the SVG and the page say what recent recalls reach. Other formats
  ignore it.
  - Each drawn entity's title says how many of the recalls read returned
    one of its facts, each recall counted once for an entity.
  - The description says over which recalls: how many the event log
    keeps, since when, the oldest read, and whether older ones were left
    unread. It is a window, never all time.
  - When the engine keeps no events, or the log holds no recalls, it says
    recall use is unknown and draws no count, never a zero.
  - The page carries the counts in its data and on each entity
    (`data-recalled`). It adds a box that dims the entities no recall
    returned, and the panel says how many recalls returned the chosen one.
- **The community map** (`communities`) is for a graph the drawings cannot
  show whole. They draw the 200 most connected entities, and when they
  leave some out their description says the community map exists.
  - Every community the analysis found is a circle, up to 80, largest
    first. A circle's area grows with its members, and entities with no
    relation to another share a grey one.
  - The circles sit on one ring, each given arc in proportion to its size.
    A line between two communities is a chord across the middle, so it
    never runs through a third, and names are written outward, clear of
    the lines.
  - A line's weight and title count the links between the two
    communities. A link is a pair of entities that one or more relations
    join, which is how the analysis counts a community's links inside
    and to others. What the graph names and never reads (`typing`) is
    attached to a community for reading, as the analysis attaches it, but
    a link through it joins no two communities, so when no community or
    line is left out a circle's lines add up to its links to other
    communities. Each circle's title gives its members and those two
    counts. Up to 400 lines are drawn, strongest first.
  - The description says which view the map draws. It says whether the
    analysis found communities among every entity with a relation to
    another or only among the 20,000 most strongly linked. It gives how many
    communities and their entities were left out, how many links between
    how many pairs of communities, and what a link is.
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
  - Entities of one community drawn together sit in a subgraph named for
    the community (`c1`, `c2`, ..., largest first), from the same
    analysis the drawings box them by. An entity drawn without another of
    its community, because it has none or the chart cut the rest, sits
    outside any subgraph. Relations run between nodes as before, across
    subgraphs included.
  - Each entity is styled by its kind: a class named for the kind
    (`person`, `organisation`, `place`, `project`, `product`, `event`,
    `concept`), a white node with a border in the drawings' colours. An
    entity of unknown kind is left plain. Kind names are fixed words, not
    stored text. The palette means something different here: in the SVG,
    the page and the canvas a colour is a community, and in this chart a
    border colour is a kind; the subgraphs already show the communities.
  - The subgraphs and classes count toward the 45,000 units like every
    other line.
  - A name, and a community's name, is a quoted label, in which `"`, `#`, `<`, `>`, `&`, `` ` ``
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
  - Every community the analysis found has a hub note in `communities/`.
    It holds the community's members, most central first, and its counts
    of entities, links inside and links to other communities, and cohesion.
    It also lists the kinds, the most used predicates, and each community
    it links to with how many links, counted as the map counts them, never
    through what the graph only names. A hub's file name is kept distinct
    from every entity note's, since Obsidian opens a bare `[[name]]`
    wherever that file sits.
  - Each community has one tag, `community/c<rank>-<words of its first
    name>`, rank first so no tag is only digits and no two share one. The
    tag is on the hub and on every member's note, and a member's note
    links to its hub. An entity with no relation to another has neither.
  - `.obsidian/graph.json` gives the graph view a colour group per
    community, querying that same tag, in the drawings' colours. Unzipped
    into an existing vault, it replaces that vault's graph view settings.
    Written with `--into`, the notes keep their tags and hubs but this file
    is left out: the vault's `.obsidian/` is not the writer's to touch, and
    a copy under `scone/` would be read by nothing.
  - `index.md` lists the communities, then the entities.

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
| `scone graph merge ALIAS INTO --reason R` | records that ALIAS is the entity INTO names, as `merged:` |
| `scone graph unmerge ALIAS --reason R` | closes the merge in force for ALIAS, as `unmerged:` |
| `scone graph merges` | the merges the current graph applies, one line each |
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

- Identity is key identity, plus the merges a person recorded. Two
  spellings of one thing ("Dr. Alice Chen" and "alice chen") stay two
  entities until someone merges them; `/v1/entities/duplicates` suggests
  the pairs worth that decision.
- The graph applies merges; retrieval's walk does not yet. Multi-hop
  expansion still joins a claim's object to another claim's subject by the
  identity rule alone, so a walk from "alice chen" does not reach claims
  about "dr. alice chen" through a merge.
- A merge older than the newest 50,000 facts a projection reads is not
  seen by that projection, which already reports the read as truncated.
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
