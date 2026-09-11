# Entity graph quality, fixtures-v1

`scone bench-graph --fixtures benchmarks/entity_graph/fixtures-v1.jsonl`
scores the entity projection on a synthetic, gold-labelled fixture: 15
facts, 5 name clusters, 15 labelled objects and 4 paths, one of them a
path that must not be found. Every number except the timing is
deterministic and hashed into `artefact_sha256`.

## Baseline (September 11, 2026)

| Metric | Value | Threshold |
| --- | --- | --- |
| Literal error rate | 0.0 | at most 0.0 |
| Connected-claim share | 1.0 | at least 1.0 |
| Path recall, 2 and 3 hops | 1.0, 1.0 | at least 1.0 |
| Path false positives | 0 | at most 0 |
| Fragmentation | 1.2 | at most 1.2 |
| Alias B-cubed F1 | 0.935 | at least 0.9 |
| View bytes (knowledge, report, context) | 8,774, 5,509, 684 | recorded |
| Build time per 10,000 facts | about 0.5 s | recorded |

Artefact: `44ceae91814cc2f504bde4ae280795a3d3bfa43438a3313366d2630500b6e147`.

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

Thresholds start at this baseline and only tighten. A classifier that
calls every object a thing breaches the literal error rate, and a test
checks that the gate fails on it.
