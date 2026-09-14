# Query rewrite: a question restated for search, and said so

The lexical lane finds the words a passage has, and a question is not
written in them: "What happened with the bills?" misses a passage about
the billing run. A model can restate the question as the words the
passage that answers it would contain — that is the whole of what
`retrieval/query_transforms.py` asks of one, and it is never on by
default.

```python
from scone_memory.retrieval.query_transforms import rewrite, rewritten_recall

asked = await rewrite(model, "What happened with the bills?")
# Transformed(question=..., query="bills billing run invoices", applied=True, reason=None, model_calls=1)
found, asked = await rewritten_recall(engine, model, "notes", "What happened with the bills?", limit=5)
```

The restatement is taken only when it can be read as a JSON query, is
not empty, is within the query bound, and still shares a word with the
question — a rewrite that shares nothing is a different question. In
every other case, and when the model fails or the deadline passes, the
question itself is searched and the record says why (`applied: false`,
`reason`). A recall made this way returns the transform beside its
results, so a reader sees what was actually searched.

## Measured before claimed

The bench runner takes a `transform`: `run(make_engine, items,
transform=...)` restates every question and records on each item the
query searched and whether the transform was applied, so the same items
measure the rewrite against the questions as asked. The numbers from
LongMemEval-S with the local model are on the pull request that added
this; the rule is that a transform earns a flag in the conversation
path only after it has moved a number there.
