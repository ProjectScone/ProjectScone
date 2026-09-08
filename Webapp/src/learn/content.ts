export const sourceExample = `# Choose your Scone server and set SCONE_KEY in your shell first.
# Use a test space: the POST below saves one example source.
: "\${SCONE_URL:?Set SCONE_URL to your Scone server URL}"
: "\${SCONE_KEY:?Set SCONE_KEY to a Scone space key}"

curl --fail-with-body --silent --show-error \\
  "$SCONE_URL/v1/episodes" \\
  -H "Authorization: Bearer $SCONE_KEY" \\
  -H 'Content-Type: application/json' \\
  --data '{"content":"The observatory uses Polaris for calibration.","tags":["quickstart"]}'

curl --fail-with-body --silent --show-error --get \\
  "$SCONE_URL/v1/recall" \\
  -H "Authorization: Bearer $SCONE_KEY" \\
  --data-urlencode 'q=Polaris calibration' \\
  --data-urlencode 'tags=quickstart'`;

import {workflowPages} from './topics.ts';
export type ConceptId = 'overview' | 'how-it-works' | 'graph-memory' | 'quickstart' | 'sources' | 'search' | 'review' | 'profiles' | 'conversations' | 'spaces' | 'api';
export type ConceptSection = {id: string; title: string; body: string};
export type ConceptPage = {
  id: ConceptId; path: string; label: string; title: string; intro: string;
  sections: ConceptSection[]; actions: {label: string; to: string}[];
};
export const conceptPages: ConceptPage[] = [
  {id:'overview',path:'/learn',label:'Overview',title:'What is Scone?',
    intro:'Persistent knowledge and real-time conversations for agents. Bring context in, connect what it means, and carry it into the next interaction.',
    sections:[
      {id:'one-system',title:'One system. Different kinds of context.',body:`An **episode** is an original source: a note, a retained message, or imported text. **Chunks** are pieces of that source used for retrieval. **Claims** record statements about subjects, with status, origin and validity dates. A **profile** assembles a compact view of memory for context.

These are not interchangeable. Finding a passage does not approve its claims. A profile is not an independent source. A model-generated claim is not established truth.`},
      {id:'use-it',title:'Follow the evidence, then act.',body:`Start in **Documents** to add or inspect retained material. Use **Search** to retrieve passages. In **Review**, compare proposed claims with their sources before approving or declining them. **Memory claims** shows recorded statements and their history.

The **Playground** connects records through stored references. Open a node to inspect the underlying record. Grouping and layout make the map readable; they do not create semantic relationships.`},
      {id:'conversation',title:'Memory meets conversation.',body:`A configured conversation service can retrieve context and save public messages and replies. Persona settings describe independent reply, transcription, speech and activity-detection choices. Your memory key alone does not configure these services.

No hidden reasoning is captured. A completed reply is not a token stream. Public reply previews are shown only when the service supports them, and missing events remain missing. Voice configuration is not proof that browser audio is available.`},
      {id:'boundaries',title:'Your space is the boundary.',body:`A Scone space key determines which memory space a request can access. Tags and metadata narrow results **inside** that boundary; they do not grant access to another space.

Rust and Python are independent native implementations, with libraries and CLIs as well as HTTP hosts. Their supported operations can differ. The workspace checks the connected host’s capabilities before offering an operation.`},
    ],actions:[{label:'Open Documents',to:'/memory#documents'},{label:'Explore the Playground',to:'/playground'}]},
  {id:'how-it-works',path:'/learn/how-it-works',label:'How it works',title:'How Scone works',
    intro:'Keep the original. Prepare it for retrieval. Extract claims separately. Inspect what actually returned.',
    sections:[
      {id:'source',title:'01 / Retain the source',body:`Add text through Documents or POST /v1/episodes. The source receives an episode ID. That ID is how you reopen the original behind a result; a source label or filename is not a replacement for it.

The example below saves a note to the space selected by your key. It never runs automatically. Set SCONE_URL and SCONE_KEY in your shell, and use a test space rather than your production memory.`},
      {id:'retrieval',title:'02 / Prepare retrieval',body:`The native ingest path chunks the source and prepares its retrieval representation. A successful add receipt identifies the episode; recall returns matching items with references to retained material.

Search ranking is not factual confidence. An embedding backend or fallback can affect ranking quality. Check the response’s degradation indicators, and open the source instead of treating a high score as proof.`},
      {id:'consolidation',title:'03 / Consolidate and review',body:`With a configured extraction model and worker, Scone can process retained text into proposed claims. This is separate from adding searchable source material. Without that configuration, storing a note does not automatically populate Review.

A proposed claim must be reviewed before it becomes eligible as an accepted claim. Checking that a quote occurs in a source only establishes textual grounding—not that the statement is correct.`},
      {id:'receipts',title:'Know which stage you have reached',body:`**Source saved** means the ingest call returned a receipt. **Passage retrieved** means recall returned source material. **Claim proposed** means extraction produced something to review. **Claim accepted** records a decision about that claim.

Python hosts advertising **jobs.read** expose batch receipts in Memory → Status. Searchable and consolidated counts are separate. A batch receipt is not proof of a restartable, stage-by-stage pipeline; full stage recovery and cancellation remain under development.`},
      {id:'identity',title:'Imports, retries and failures',body:`Repeated content may be deduplicated. A filename is not a versioned document identity, and this HTTP example does not offer replace-by-key semantics. Native dedup keys are not an update API: reusing a known key can return the earlier episode instead of storing changed content.

On an uncertain write, inspect what was retained before retrying. Authentication errors need the correct space key; validation errors need corrected input. Do not blindly retry all failures. Format support and limits depend on the host; universal file extraction, resumable batch imports and deletion cascades are not yet complete.`},
    ],actions:[{label:'Open Documents',to:'/memory#documents'},{label:'Try Search',to:'/memory#search'}]},
  {id:'graph-memory',path:'/learn/graph-memory',label:'Graph memory',title:'Graph memory',
    intro:'A useful graph explains where a statement came from, when it applies, and how it relates to other statements. It also makes uncertainty visible.',
    sections:[
      {id:'nodes',title:'What is a node? What is a link?',body:`A record node represents an identified source, chunk, claim, session, interaction or recall record supplied by the host. A group node summarizes records for navigation; it is not an additional stored fact.

A link represents a stored reference between records. Two nodes being near each other does not imply agreement, causality or a new relationship. Following a link explores evidence; it does not capture an agent’s hidden reasoning.`},
      {id:'updates',title:'Updates: a statement changes',body:`Imagine a source saying “Mira works at Cedar,” followed later by “Mira now works at Rowan.” These are illustrative examples, not records in your space.

Scone’s fact ledger can retain superseded statements with validity boundaries. Current and historical reads serve different questions: “What applies now?” versus “What applied then?” A validity date is not the same as the date the system first learned something.`},
      {id:'extensions',title:'Extensions: a statement gains detail',body:`“Mira leads the payments team” might enrich a role without replacing the employer statement. An extension should preserve both statements and record an explicit typed link.

Python hosts advertising **facts.links** support typed extends, derived_from, contradicts and supports links. The fact-detail API returns a fact, its links and source episode IDs; dependency links are checked for cycles and cross-space targets are refused. The graph endpoint includes stored claim-to-claim links.

Rust parity and complete relationship inspection workflows remain unfinished. Inspect the link’s recorded type; a generic line or visual grouping does not establish an extension.`},
      {id:'derivations',title:'Derivations: an inference names its premises',body:`Several statements can suggest a new claim, but the inference can still be wrong. Python’s native derivation operation records named premises and an inferred origin. Those links let you inspect the basis of the claim; they do not prove its conclusion.

**Proposed** derived claims require approval before they become accepted claims. An explicit authorized API assertion can create an **active** inferred claim without a separate review decision; do not mistake that for a human approval. The origin label alone does not prove that supporting links were stored.

Automatic extraction of these relationships, full cross-host support and dependency-aware retrieval remain unfinished. Approval of the premises is not proof that the conclusion is correct.`},
      {id:'decisions',title:'Review changes what can be recalled',body:`Use Review to inspect proposed claims and their source quotes. Approve or decline deliberately. Bulk approval freezes the matching set across all pages, not just the visible cards; inspect that confirmation before applying it. New arrivals are not included. Memory claims lets you inspect status and history. Available actions depend on the connected host.

An approval records a decision, not a mathematical guarantee. Keep original evidence available so people can revisit that decision.`},
      {id:'forgetting',title:'Time, forgetting and deletion are different',body:`Closing a claim’s validity says it no longer applies after a point in time. Excluding a claim controls retrieval where supported. Forgetting a source removes retained material through that host’s operation; it is not automatically a complete erasure of every claim, link, event or attachment derived from it.

Use the host’s impact-preview and deletion receipts where available. Complete cross-runtime retention and erasure guarantees remain under development. Missing evidence is shown as missing rather than replaced with an invented explanation.`},
    ],actions:[{label:'Open Review',to:'/memory#review'},{label:'Inspect Memory claims',to:'/memory#beliefs'},{label:'Explore the Playground',to:'/playground'}]},
];
conceptPages.splice(1,0,workflowPages[0]);
conceptPages.push(...workflowPages.slice(1));

export const documentationGroups=[
  {label:'Start here',ids:['overview','quickstart','how-it-works']},
  {label:'Knowledge & memory',ids:['sources','graph-memory','search','review','profiles']},
  {label:'Build with Scone',ids:['conversations','spaces','api']},
] satisfies {label:string;ids:ConceptId[]}[];

export function conceptMarkdown(page: ConceptPage): string {
  return `# ${page.title}\n\n${page.intro}\n\n` + page.sections.map(section =>
    `## ${section.title}\n\n${section.body}` + ((page.id==='how-it-works'||page.id==='quickstart')&&section.id==='source'?`\n\n\`\`\`sh\n${sourceExample}\n\`\`\``:'')
  ).join('\n\n') + '\n\n## In the workspace\n\n' + page.actions.map(action=>`- [${action.label}](${action.to})`).join('\n')+'\n';
}
