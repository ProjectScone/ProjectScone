# Agent-authored tools with human review

An agent can discover a host's allowed capabilities and propose a reusable tool by
composing them. The proposal is durable, encrypted and scoped to one space. It
cannot run until a host-authenticated reviewer approves its immutable version.
When offered through an agent workflow, the approved tool also requires the
existing exact-call approval, including the user's arguments and tool revision.
Version approval alone never activates a call.

Native Python and authenticated HTTP version review are available. The reviewer
UI, SDK management methods, isolated test-report workflow and sandboxed Python
authoring are not implemented by this milestone. Its scripted-model tests are not
a real-model or UI acceptance claim.

## Author, review, execute

Install the `agents` and `structured-output` extras. Provide trusted `AgentTool`
capabilities whose handlers enforce their resource policies. Hosts must change a
capability's revision whenever its implementation or external configuration changes.

```python
from scone_memory.agents.tool_recipe_store import ToolRecipeStore

recipes = ToolRecipeStore(private_path, key=encryption_key)
# Register these on an authoring agent with an explicitly selected model.
authoring_tools = [
    recipes.capabilities_tool(space="alpha", tools=allowed_tools),
    recipes.proposal_tool(
        space="alpha", proposed_by="author-agent", tools=allowed_tools,
    ),
]
```

`recipe_capabilities` returns dependency metadata, digests and the `ToolRecipe`
JSON schema. The agent submits a `proposal_id` and `recipe_json` through
`propose_tool_recipe`. Neither adapter accepts a reviewer identity or a decision.
Proposing again with the same ID is idempotent only for identical content, author
and dependency metadata; changed content needs a new ID and review.

```python
page = recipes.list("alpha", limit=20)
proposal = recipes.get("alpha", proposal_id)
# Present proposal.recipe, proposal.requirements, proposal.dependencies(),
# proposal.reviews and proposal.revision to the authenticated human reviewer.
reviewed = recipes.decide(
    "alpha", proposal_id, decision="approve", actor=reviewer_identity,
    reason=review_reason, expected_revision=proposal.revision,
)
new_tool = recipes.bind("alpha", proposal_id, tools=allowed_tools)
# Register new_tool on an execution agent. Use the normal AgentWorkflow or
# AgentRunService exact-call approval, decision, and explicit activation flow.
```

The host must authenticate the human and supply their identity; a native Python
method cannot establish that a string belongs to a human. Self-review under the
proposal author's identity is refused. The bound tool's fingerprint includes the
reviewed recipe, dependency metadata, approval record and store authority. Moving
a store or changing dependency metadata requires a new agent/plan binding.

`decide(..., decision="deny")` prevents binding. `revoke(..., expected_revision=2)`
retires an approved version while preserving the approval and revocation records.
Review decisions use compare-and-swap revisions. A copied database is a different
review authority and cannot silently inherit an existing call/plan fingerprint.

## Authenticated HTTP review

Pass `tool_recipe_store=recipes` to `create_app`. The caller owns the store's
lifetime. Hosts without that store expose neither the routes nor the
`agents.tool_recipes.review` capability. This does not automatically register an
approved tool in an agent catalog.

- `GET /v1/tool-recipes?limit=50&after=...` lists proposals in the current space.
- `GET /v1/tool-recipes/{proposal_id}` returns recipe, dependency metadata and reviews.
- `POST /v1/tool-recipes/{proposal_id}/decision` accepts `decision` (`approve` or
  `deny`), a nonblank `reason`, and `expected_revision: 1`.
- `POST /v1/tool-recipes/{proposal_id}/revoke` accepts `reason` and
  `expected_revision: 2`.

Review and full-access keys can decide or revoke; read and write keys cannot.
Reviewer identity is a keyed fingerprint of the authenticated credential, never
a client-supplied label. The API rechecks key, role and scope after acquiring the
write lock. It checks deleted-space state after receiving the request body and
before returning success; this separate memory-store check is not an atomic
transaction with recipe storage. Concurrent external space deletion requires host
coordination. Reviews never execute handlers or activate pending calls.

Malformed input returns `422`, oversized review bodies `413`, missing proposals
`404`, revision/transition conflicts `409`, and unavailable storage `503`. After
an ambiguous response or revision conflict, read the current record before deciding
again. The API does not retry review mutations. Successful responses prohibit
caching. Proposal identifiers cannot be URL navigation segments (`.` or `..`).

## Execution contract

Recipes have up to eight sequential steps, scalar inputs, literal JSON values,
input references and paths into prior results. Forward references, recursive
calls, reserved names and unknown dependencies are refused before review. There
is no source evaluation, import, shell, endpoint discovery or implicit filesystem
access. Arbitrary Python remains a separate planned sandboxed authoring path.

Nested dependencies that require their own call approval or return directly are
refused; a composition must not bypass those semantics. Dependencies remain
trusted host code. `AgentTool.invoke()` is a host-level interface, not the human
approval boundary: applications should execute through the agent workflow's
approval machinery, as with existing guarded tools.

Approval is checked at binding, before each step, and when its actual handler
starts, including after a worker queue delay. Revocation prevents later dispatch;
an already admitted handler may finish and its effects cannot be undone. A blocked
authorization read cannot authorize starting a handler after deadline/cancellation.
A missing or closed review store refuses execution.

Each recipe shares the invocation deadline, caps aggregate dependency packets at
64 KB and exposes at most 16 KB of result JSON. A recipe is one outer journaled
tool call; its bounded internal calls are not yet separate progress events. If a
step fails after earlier effects, the outcome may be unknown. The existing journal
refuses automatic replay of an uncertain call. Results stay unverified application
output and are not inserted into the factual memory ledger.

## End-to-end acceptance still required

A release claim needs a real local browser, API, selected local model and durable
workflow: request a tool, inspect its proposal/dependency versions and actual test
results, approve or deny its version, review the exact call, activate it, and
inspect effects, results and audit history. Verify refresh/restart, cancellation,
revocation before and during execution, changed dependencies, rejected decisions
and isolation between spaces. Screenshots and backend receipts must describe the
same actual run. Native unit and workflow tests alone do not satisfy that gate.
