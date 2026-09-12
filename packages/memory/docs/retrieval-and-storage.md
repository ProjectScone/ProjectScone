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
wrong. `GET /v1/answer`, `scone answer`. Nothing here calls a model.

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

`map` walks a directory, remembers every source file under the path it
was read from, and with `--graph` records what each says. An answer
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
resolve: a bare name that is one of the file's own declarations, and
`self.method` inside the class that defines it.

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
Kinds: `email`, `phone`, `ip`, `card`, `secret`.

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

Built on `ingestion/structure.py`, which already finds headings, fenced
code and pipe tables and is already used by retrieval and source
inspection. Only what that parser deliberately leaves out is new:
setext headings, numbered and lettered clauses, `Q:`/`A:` pairs.

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
# joined 3 chunk(s) into #12: 11, 12, 13
```

**No hierarchy and no re-index**, because every chunk already carries the
byte span it came from: neighbours from one episode are merged by reading
the span that contains them. The shape of a merge is therefore decided by
what was actually retrieved, not by a decision taken at ingestion.

Three rules it keeps:

- **A merged passage says what went into it.** `from_chunks` names every
  chunk absorbed, because a citation nobody can check is worse than three
  that can.
- **It keeps the best score of its parts, never their sum.** A sum would
  make a merged passage outrank everything by arithmetic rather than by
  relevance.
- **It is a passage, not a document.** Fragments further apart than
  `max_merged` bytes are left alone and the report says so — silently
  returning most of a document to answer a question about a sentence
  would be worse than not merging.
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

`map` remembers the files under a directory and notices when it has seen
one before. What it cannot notice is that a file has **changed** or that
a file is **gone** — and those two are the difference between an import
you run once and a sync you run on a schedule.

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
leaves **one** memory and not two.

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
  collision, and is not counted.

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
- **The space merged from is closed for good.** Everything it held is
  somewhere else now, and leaving the name open would invite somebody to
  write into a space whose contents have moved and find them missing.

## What an archive says it is

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
  is labelled `Alice Chen`.
- **Kinds** (person, organisation, place, project, product, event, concept)
  are inferred hints from the predicates around an entity. They carry
  `kind_status: "inferred"` and list the fact ids that suggested them. Hints
  that disagree give `"conflict"` and no kind, never a guess.

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
