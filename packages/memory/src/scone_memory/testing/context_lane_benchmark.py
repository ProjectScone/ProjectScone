"""Passages found by what they are under, with and without the context lane.

A deterministic synthetic set, version ``under-v1``: each case is a
document with a title and a heading whose words the question uses, and
a body under that heading that says none of them -- the passage the
question is about. Eight distractor passages per case repeat the question's own words
without answering it, as the corpus's chatter about a topic always does,
so the top five must choose.

For "What does the <topic> section say about <heading>?", the body is
the answer's evidence and the text lane cannot see it: it finds the
words a passage has, and this one has none of the question's. The
benchmark reports how often the body appears in the top five with the
context lane off and on, and how often a distractor takes the first
place either way, so a lane that helped recall cannot hide that it also
moved chatter up.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

VERSION = "under-v1"
_TOPICS = ["billing", "shipping", "hiring", "security", "pricing", "support", "returns", "payroll",
           "roadmap", "licensing", "onboarding", "compliance", "backups", "branding", "expenses", "travel",
           "vendors", "inventory", "training", "escalation"]
_HEADINGS = ["refunds", "deadlines", "approvals", "exceptions", "audits", "renewals", "disputes", "holidays",
             "thresholds", "handover", "rotation", "sign-off", "retention", "quotas", "waivers", "credits",
             "recalls", "outages", "reviews", "backlog"]
_BODIES = ["A request goes to the desk that handled the sale, and the money returns the way it came.",
           "Nobody is asked why; the form is stamped the same day and filed with the quarter's records.",
           "Two people sign, one of them from another team, and the note names both by initials.",
           "After thirty days the case is closed unless the desk reopened it in writing.",
           "The amount is capped at what was paid, never at what was promised in conversation."]
_TAILS = ["The desk keeps the receipt number and the file closes at quarter end.",
          "A copy goes to the archive and the original stays with the desk.",
          "The initials are checked against the roster before the file is closed.",
          "Reopening needs a reason in writing and a date on the note.",
          "Anything above the cap is written off and the difference is logged."]


@dataclass(frozen=True)
class UnderReport:
    version: str
    cases: int
    limit: int
    #: How often the tail of the body under the heading -- the chunk that says
    #: none of the question's words -- was in the top ``limit``.
    body_found_without: int
    body_found_with: int
    #: How often a distractor -- chatter repeating the question -- took first place.
    chatter_first_without: int
    chatter_first_with: int
    store: str

    def as_payload(self) -> dict[str, object]:
        return asdict(self)


def cases(count: int = 20) -> list[dict[str, str]]:
    out = []
    for index in range(count):
        topic = _TOPICS[index % len(_TOPICS)]
        heading = _HEADINGS[(index * 7) % len(_HEADINGS)]
        # The body runs past the chunk target, so its tail is a chunk of
        # its own that says none of the question's words: the lexical miss.
        tail = _TAILS[(index * 3) % len(_TAILS)] + f" Case {index} closes here."
        body = _BODIES[index % len(_BODIES)] + " " + tail
        document = (f"# {topic.title()} rules\n\nWhat the company does about {topic}.\n\n"
                    f"## {heading.title()}\n\n{body}\n")
        chatter = [f"The {topic} team talked about {heading} again; {heading} and {topic}, as always.",
                   f"See the {topic} handbook's page on {heading}; every {heading} note cites it.",
                   f"{heading.title()} came up at the {topic} review, and {topic} at the {heading} one.",
                   f"A reminder that the {topic} section covers {heading}; ask the {topic} desk.",
                   f"Minutes: {heading} raised under {topic}; no decision on {heading} yet.",
                   f"The {topic} channel pinned a thread on {heading} and {topic} questions.",
                   f"Old {topic} slides mention {heading} twice and {topic} on every page.",
                   f"Someone asked what the {topic} section says about {heading}; nobody looked it up."]
        out.append({"topic": topic, "heading": heading, "body": body, "tail": tail, "document": document,
                    "question": f"What does the {topic} section say about {heading}?", "chatter": "\n".join(chatter)})
    return out


async def run_under_benchmark(count: int = 20, limit: int = 5, *, sqlite_path: str | None = None) -> UnderReport:
    """Every case in its own space: the lane off, then on, over the same passages."""
    from .. import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine

    counts = {"body_found_without": 0, "body_found_with": 0, "chatter_first_without": 0, "chatter_first_with": 0}
    store_name = "memory"
    for case in cases(count):
        for lane in (False, True):
            if sqlite_path is not None:
                from ..backends import SqliteDocumentStore

                documents = SqliteDocumentStore(f"{sqlite_path}.{'on' if lane else 'off'}.{case['topic']}.db")
                store_name = "sqlite"
            else:
                documents = InMemoryDocumentStore()
            # Word families are a lever of their own, measured elsewhere; held
            # off here so what the lane recovers is the lane's, not a prefix's.
            engine = await MemoryEngine(documents, InMemoryVectorIndex(), HashEmbedder(), chunk_target=120,
                                        context_lane=lane, lexical_stems=False).open()
            try:
                for line in case["chatter"].splitlines():
                    await engine.remember("s", line)
                await engine.remember("s", case["document"], source=f"docs/{case['topic']}-rules.md")
                found = await engine.recall("s", case["question"], limit=limit)
            finally:
                await engine.close()
            suffix = "with" if lane else "without"
            if any(case["tail"] in item.text for item in found.items):
                counts[f"body_found_{suffix}"] += 1
            if found.items and case["tail"] not in found.items[0].text and case["topic"] in found.items[0].text:
                counts[f"chatter_first_{suffix}"] += 1
    return UnderReport(VERSION, count, limit, counts["body_found_without"], counts["body_found_with"],
                       counts["chatter_first_without"], counts["chatter_first_with"], store_name)
