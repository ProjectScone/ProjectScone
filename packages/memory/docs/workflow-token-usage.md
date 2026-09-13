# Durable workflow token usage

Completed model tasks persist their [native token reports](agent-token-usage.md)
in the existing encrypted workflow journal. This includes sequential and
parallel tasks, handoff hops, and model tasks in human-input workflows. The
receipt records the selected agent, model, configuration binding, and one usage
report per accepted model call. Human replies contain no model usage.

Each model receipt has optional `usage`. New executions record
`{"calls":[{"prompt_tokens":120,"completion_tokens":18,"total_tokens":138}]}`.
Old receipts that predate accounting remain readable with unknown usage;
opening or reading them never invokes a model to reconstruct counts. The number
of reports must equal the receipt's `model_calls`. Invalid counts, inconsistent
totals, and incomplete report arrays are rejected during receipt validation.
Missing provider fields remain null in a report, independently per category.

Reading or reusing a saved task returns its original counts. It does not record
another model request. A handoff result's `final` is the last hop's output; count
the `hops` once when comparing work and do not add `final` again. Reports stay
outside downstream prompts and human-input context.

## HTTP negotiation

Hosts with a run service advertise `agents.usage: true` in `/v1/capabilities`.
The default result response retains its previous shape. Opt in explicitly:

```text
GET /v1/agent-runs/{run_id}/result?include_usage=true
```

Every model output then contains `usage`, including the final handoff copy.
Historical receipts return `usage: null`; human outputs do not gain this field.
`include_usage=false` is equivalent to omitting the option. Only the literal
strings `true` and `false` are accepted, once each request; duplicate options and
other values return 422. Result reads retain space authorization, current model
bindings, source verification and `Cache-Control: no-store`. Usage is withheld
with the rest of the result if those checks fail.

The standalone Python client exposes the same opt-in contract:

```python
result = client.agents(expected_space="team").result("report-1", include_usage=True)
output = result.results["answer"]
if output.usage is not None:
    print(output.model_id, output.usage.total_tokens)
```

The client requires the advertised capability, validates each report, and keeps
the server's source-verifying result read as its final remote operation. The
console negotiates the same feature and shows each task's counts and reporting
coverage. A partial category displays **Unknown**, preserving available reports
without presenting a partial sum as a complete total. Older servers keep the
existing result view.

## Interpretation and upgrades

These are provider assertions for successfully completed model tasks, not
billing totals, independently measured tokenization, prices, or quality scores.
Interrupted or failed tasks may consume tokens without a completed receipt.
Repeated UI reads do not imply repeated model use; inspect the saved-work labels.

The new runtime reads historical journals. Older runtimes with strict receipt
schemas cannot read newly enriched journal entries; retain a compatible runtime
when reopening state written with usage. The HTTP default remains compatible
with older clients. Journal payload limits still apply to the stored reports.
There is no embedding accounting, cost attribution, failed-attempt ledger,
distributed billing reconciliation, or Rust parity in this feature.
