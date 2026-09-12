# Entity graph quality, fixtures-v1

`scone bench-graph --fixtures benchmarks/entity_graph/fixtures-v1.jsonl`
scores the entity projection on a synthetic, gold-labelled fixture: 15
facts, 5 name clusters, 15 labelled objects and 4 paths, one of them a
path that must not be found. Every number except the timing is
deterministic and hashed into `artefact_sha256`.

Scores are read from what the projection carries, and something missing
is never scored as right. A fact is carried only by an item that has its
own subject, predicate and object or value. An item that merely cites
the fact's id doesn't count. So a labelled value counts only if such an
attribute carries it, and a labelled thing only if such a relation does.
A gold name with no entity adds nothing to B-cubed. Three counts report
missing evidence outright:

- facts the view should hold at `as_of` that nothing carries;
- gold names with no entity;
- path ends with no entity.

What should hold is the fixture's own timeline, not the ledger's
answer. For each subject and predicate, the row that began last by
`as_of` holds, since the ledger keeps one value per pair; on a tie the
later line wins. Claims are compared as the ledger keeps them: subject
and predicate with case and spacing folded, and the object trimmed.

- A claim that begins later, or that a later row supersedes, is counted
  apart as out of view.
- An age of 34, then 35, then 34 again holds 34, whichever order the rows
  were told in.
- A ledger that let the order decide would be reported as missing. Told
  Acme from 2020, Acme again from 2023, then a late Globex from 2021,
  the ledger once folded the second Acme into the first and let the
  Globex cut it short. It now keeps the restatement as an affirmation,
  so every order agrees.

A fixture is refused (exit 2), naming each line, rather than scored, in
two cases:

- **A malformed row.** A required field is missing or an unknown one is
  present. Or a field has the wrong type: `null` or `"false"` where
  `true` or `false` is due, hops that are not a whole number from 1 to
  4, or a time that is not RFC 3339.
- **Gold about something the view does not hold.** A name, a path end
  or a labelled claim that no fact due by `as_of` states.

Either kind of gold could only be scored by being reinterpreted, or by
being vacuously right. A score with no gold at all is reported as
unmeasured, and its threshold fails.

## Baseline (September 11, 2026)

| Metric | Value | Threshold |
| --- | --- | --- |
| Claims missing from the projection | 0 | at most 0 |
| Claims out of view at `as_of` | 0 | recorded |
| Gold names with no entity | 0 | at most 0 |
| Path ends with no entity | 0 | at most 0 |
| Literal error rate | 0.0 | at most 0.0 |
| Connected-claim share | 1.0 | at least 1.0 |
| Path recall, 2 and 3 hops | 1.0, 1.0 | at least 1.0 |
| Path false positives | 0 | at most 0 |
| Fragmentation | 1.2 | at most 1.2 |
| Alias B-cubed F1 | 0.935 | at least 0.9 |
| View bytes (knowledge, report, context) | 9,239, 5,509, 684 | recorded |
| Build time per 10,000 facts | about 0.5 s | recorded |

Artefact: `e92a2920baf658b8a413753cee07ed25ba81796a8240f4e0345a6cf8cff584a2`. The scores
match the first recording (`44ceae91…`). The hash has moved twice since:
when the missing-evidence and out-of-view counts joined the report
(`daceac58…`), and now that the knowledge view says when things held.
No score changes. The view is 465 bytes longer for two reasons, both of
them the view saying what it used to leave a reader to assume: every
relation now lists the stretches of valid time it held over, rather than
a first and last that read as one unbroken spell; and the view says that
nothing follows from this fixture's claims, because it configures no
vocabulary (`implied: []`, `meanings: null`, and the two counts).

## What the numbers say

- **Fragmentation 1.2:** one cluster of five is split. "dr. alice chen"
  is its own entity beside "alice chen", because identity is key
  identity until identity decisions exist. This pins the pre-merge
  state, so that work can show the change.
- **Literal error rate 0:** dates, quantities, an email, a unit
  (`512 MB`), a status, a URL and a quoted sentence all stay values.
  Every labelled thing becomes an entity, including `the Robotics Lab`,
  a determiner followed by a title-cased name.
- **Path recall 1.0 and no false positives:** every expected connection
  is found within its hop count. That includes one through a shared
  place in reverse (`bob stone` to `acme robotics`, via Lisbon). The
  unconnected pair stays unconnected.

Thresholds start at this baseline and only tighten. Tests check that
the gate fails on each of these:

- a classifier that calls every object a thing;
- a projection that loses every value;
- a projection that loses one relation while its role survives;
- a projection that pins six lost values' fact ids onto the one value
  left;
- a projection that moves a relation's fact ids onto another relation;
- a projection that loses an entity.
