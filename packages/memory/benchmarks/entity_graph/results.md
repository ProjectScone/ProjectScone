# Entity graph quality, fixtures-v1

`scone bench-graph --fixtures benchmarks/entity_graph/fixtures-v1.jsonl`
scores the entity projection on a synthetic, gold-labelled fixture: 15
facts, 5 name clusters, 15 labelled objects and 4 paths, one of them a
path that must not be found. Every number except the timing is
deterministic and hashed into `artefact_sha256`.

Scores are read from what the projection carries, and something missing
is never scored as right. A labelled value counts only if an attribute
carries its fact, and a labelled thing only if a relation does. A gold
name with no entity adds nothing to B-cubed. Three counts report missing
evidence outright:

- loaded facts that became neither a relation nor an attribute;
- gold names with no entity;
- path ends with no entity.

A fixture whose gold names something no fact states is refused (exit 2)
rather than scored, because such gold could only be vacuously right. A
score with no gold at all is reported as unmeasured, and its threshold
fails.

## Baseline (September 11, 2026)

| Metric | Value | Threshold |
| --- | --- | --- |
| Claims missing from the projection | 0 | at most 0 |
| Gold names with no entity | 0 | at most 0 |
| Path ends with no entity | 0 | at most 0 |
| Literal error rate | 0.0 | at most 0.0 |
| Connected-claim share | 1.0 | at least 1.0 |
| Path recall, 2 and 3 hops | 1.0, 1.0 | at least 1.0 |
| Path false positives | 0 | at most 0 |
| Fragmentation | 1.2 | at most 1.2 |
| Alias B-cubed F1 | 0.935 | at least 0.9 |
| View bytes (knowledge, report, context) | 8,774, 5,509, 684 | recorded |
| Build time per 10,000 facts | about 0.5 s | recorded |

Artefact: `ca795461b85212cd25cd1109aa82b4ae0f4b49320ea9a0315d70f151cb584d57`. The scores
match the first recording (`44ceae91…`). The hash moved when the three
missing-evidence counts joined the report.

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
- a projection that loses an entity.
