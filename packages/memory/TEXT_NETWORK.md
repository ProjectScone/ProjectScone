# Temporary text networks

`scone_memory.text_network` analyzes explicitly supplied text without a memory
engine, model, index, persistence or network connection. It is useful for
inspecting terms across notes, documents, research or manuscripts without
recording inferred facts.

```python
from scone_memory.text_network import TextDocument, TextTerm, build_text_network

result = build_text_network(
    [TextDocument(key="note:1", kind="note", id="1", title="Inspection",
                  content="The pump connects to the valve.")],
    [TextTerm(label="pump", kind="equipment"),
     TextTerm(label="valve", kind="equipment")],
)
assert result.basis == "literal_shared_passage"
assert result.persisted is False
```

An edge means only that two explicit terms share an original prose paragraph
or table row. It is not a recorded relationship, identity, chronology or
causal claim. Fenced code, including list and blockquote fences, is omitted.
Unicode case folding preserves original source offsets, including expanding
characters such as `ß`. Whole-term boundaries include combining marks.

The result includes content-bound document hashes, nodes, cited edges, exact
source excerpts, capitalization-based name candidates, spelling suggestions,
communities, cut/cross-community bridges and isolated nodes. Suggestions never
automatically add or merge terms. Communities reuse Scone's deterministic
weighted partition algorithm through `entities.analysis.partition_associations`;
the API does not construct fictitious ledger facts to use it.

The caller owns authorization and source freshness. Read the selected content
under its proper scope, then recheck that scope and document versions before
returning a result. Offsets are Python Unicode codepoints, not JavaScript UTF-16
positions. Excerpts are unmodified substrings; their offset is `excerpt_start`.
The surrounding original passage is identified separately by `start/end`.

`NetworkLimits` caps 15 documents, 60 explicit terms, three aliases per term,
512 KiB total UTF-8 content, 4,000 scanned passages, 600 returned edges,
200 cited passages and 512 KiB encoded output. An edge retains at most three
citations but counts every matching scanned passage. Coverage explicitly marks
sampling, so an absent displayed connection does not prove absence in omitted
text. Input and final output byte excesses raise `ValueError`; content is not
included in error messages. A caller's `should_cancel` callback can interrupt
work between passages and terms with `TextNetworkCancelled`.

No default local or remote provider is consulted. Applications can add their
own domain fields, provenance and UI without narrowing these generic kinds or
silently storing the supplied documents.
