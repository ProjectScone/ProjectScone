"""Summary-hit expansion on broad questions: recall_at(10) of the gold chunks, off and on.

A reproducible local fixture, not a general retrieval-quality claim. Four
short documents of three sections of three paragraphs each are stored
with ``HashEmbedder``, one paragraph per chunk (checked, not assumed). A
summary tree is built for each with ``FakeChat`` scripted as a model would
answer: a level-one sentence for each of the first ``--cite`` (default two)
paragraphs of a section
naming the section's theme and quoting the paragraph's opening words, and a
root sentence per section quoting that section node. The theme word is in
the section's first paragraph only, as a heading word usually is. Each
question asks what a document says about one theme; its gold chunks are
that section's three paragraphs. When the scripted summaries cite two of
three, cited-only expansion can reach at most two thirds of the gold: the
citation ratio is the script's, and so is that ceiling.

Arms, all at ``limit=10`` and scored on the first ten items returned:

- ``off``: the summaries are stored and recalled as ordinary passages.
- ``follow`` / ``replace``: ``expand_summaries`` with the default cap.
- ``covers``: every chunk under each summary, in place of it -- the shape of
  a document summary index returning the nodes under a summary. Computed
  here from the stored span, not a Scone option.

Run from packages/memory: ``PYTHONPATH=src python benchmarks/summary_expansion.py --repeats 5 --cite 2``.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.bench.metrics import recall_at
from scone_memory.core.models import RecallItem
from scone_memory.providers.llm import FakeChat
from scone_memory.retrieval.summary_tree import build_summary_tree

CHUNK_TARGET = 200
LIMIT = 10

DOCUMENTS: list[tuple[str, str, list[tuple[str, list[str]]]]] = [
    ("Vellmar town report", "vellmar.md", [
        ("harbour", [
            "The harbour at Vellmar closes to sailing boats every November, when the winter swell starts to run in from the northwest across the outer bar.",
            "Its lighthouse was rebuilt in 1904 after a storm took the first tower down to its foundations, and the lamp now turns on a mercury bath.",
            "Pilots board arriving ships two miles out at the red buoy, in every season, and the pilot cutter is moored beside the fish market steps.",
        ]),
        ("orchard", [
            "The orchard on the ridge above Vellmar grows only Bramley apples, planted in rows of forty on terraces cut by hand in the last century.",
            "Picking starts in the final week of September and ends before the first frost, with school children given two days away from lessons.",
            "A cider press in the tithe barn turns out two thousand litres a season, sold at the Saturday market beside the old church wall on the square.",
        ]),
        ("council", [
            "The Vellmar council meets on the first Tuesday of each month in the old customs house, and any resident may speak for three minutes.",
            "Seven members are elected every four years; the chair rotates each spring, and a tie is broken by drawing a name from a felt hat.",
            "Minutes are pinned in the post office window within a week and kept in bound ledgers that go back without a gap to the year 1871.",
        ]),
    ]),
    ("Brannock mill survey", "brannock.md", [
        ("river", [
            "The river Brann drives the mill wheel through a stone leat that was dug in 1760 and relined with brick after the great flood of 1952.",
            "In a dry August the flow falls below what the wheel needs, so the sluice is closed at night to let the pond fill again for the morning.",
            "Otters returned to the lower weir in 2011, and anglers are asked to keep away from the holt under the alder roots by the footbridge.",
        ]),
        ("machinery", [
            "The mill machinery is driven by an overshot wheel of fourteen feet, turning two pairs of French burr stones through its wooden gears.",
            "Cogs of apple wood are replaced every twelve years; the last set was cut by a wheelwright from Dunmore using the templates from 1890.",
            "A sack hoist lifts grain to the top floor, where it falls by gravity through chutes into the hoppers above the two turning stones.",
        ]),
        ("visitors", [
            "Visitors to the mill may tour it on Sundays from Easter to October, and the miller grinds a batch of wholemeal flour for sale at noon.",
            "School groups book through the parish office; each child leaves with a paper bag of flour and a pencil drawing of the big wheel.",
            "The tea room in the old drying kiln serves scones made from the mill's own flour, and dogs are welcome in the cobbled yard outside.",
        ]),
    ]),
    ("Hollin Fell weather station log", "hollin.md", [
        ("instruments", [
            "The weather station instruments on Hollin Fell include a Stevenson screen, a tipping-bucket rain gauge and a three-cup anemometer.",
            "Thermometers inside the louvred screen are read at nine each morning and checked against a calibrated reference thermometer once a year.",
            "The anemometer mast stands ten metres high and was guyed with steel cable after gales bent the original wooden pole in March 1987.",
        ]),
        ("rainfall", [
            "Rainfall on the fell averages 2,400 millimetres a year, most of it between October and February on wet south-westerly winds off the sea.",
            "The wettest day on record brought 187 millimetres in December 2015, and the beck below the station burst its banks well before noon.",
            "Snow is melted in a warm jug before it is measured, so a heavy fall is counted as water rather than as depth lying on the ground.",
        ]),
        ("volunteers", [
            "Volunteers keep the station log in turns, walking up from the village of Thwaite by the old peat track in all weathers and seasons.",
            "Each observer signs the ledger, and a missed reading is marked with a dash rather than estimated from the neighbouring days.",
            "A retired shepherd, Agnes Birkett, kept the readings unbroken for thirty-one years and was given a county medal for it in 2009.",
        ]),
    ]),
    ("Saltmere reserve plan", "saltmere.md", [
        ("birds", [
            "Birds on the Saltmere reserve are counted monthly; winter brings up to nine thousand wigeon and a few hundred elegant pintail.",
            "Avocets nested on the scrape for the first time in 1998, and a wardened hide now overlooks the islands from the top of the sea wall.",
            "Ringing at the reed bed each autumn has recaptured sedge warblers first marked in Senegal, on their long way south for the winter.",
        ]),
        ("grazing", [
            "Grazing on the marsh uses a herd of forty Red Poll cattle, turned out in May to keep the grass short for the nesting waders.",
            "The herd is moved between fields by a stockman on horseback, since vehicles would rut the soft ground and crush the eggs of waders.",
            "Water levels in the ditches are held high through spring with wooden boards, then lowered in August for the grazing cattle.",
        ]),
        ("access", [
            "Access to the reserve is by the sea wall path from Saltmere quay; the car park fills early on sunny bank holiday weekends.",
            "The eastern trail closes from March to July so that ground-nesting lapwings are not disturbed by walkers and their loose dogs.",
            "A wheelchair-accessible boardwalk leads to the nearest hide, and binoculars can be borrowed from the visitor hut by the gate.",
        ]),
    ]),
]


def notes(rows: list[tuple[str, str, str]]) -> str:
    return json.dumps({"notes": [{"sentence": s, "passage": p, "quote": q} for s, p, q in rows]})


def opening(text: str, count: int = 5) -> str:
    return " ".join(text.split()[:count])


def scripted_replies(title: str, sections: list[tuple[str, list[str]]], chunk_ids: list[int], cite: int) -> list[str]:
    """The replies a model would give, in the order the builder asks (fan_in 3: three sections, then the root)."""
    replies, firsts = [], []
    for position, (theme, paragraphs) in enumerate(sections):
        ids = chunk_ids[position * 3:position * 3 + 3]
        rows = [(f"On the {theme}, the {title} notes {opening(text)}.", f"chunk:{ids[i]}", opening(text))
                for i, text in enumerate(paragraphs[:cite])]
        replies.append(notes(rows))
        firsts.append(rows[0][0])
    replies.append(notes([(f"The {title} covers the {theme}.", f"node:1:{position}", firsts[position])
                          for position, (theme, _) in enumerate(sections)]))
    return replies


async def build(cite: int) -> tuple[MemoryEngine, list[tuple[str, set[str]]], dict[int, list[int]], dict[str, str]]:
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), chunk_target=CHUNK_TARGET).open()
    questions: list[tuple[str, set[str]]] = []
    by_episode: dict[int, list[int]] = {}
    texts: dict[str, str] = {}
    for title, source, sections in DOCUMENTS:
        paragraphs = [text for _, texts_of in sections for text in texts_of]
        added = await engine.remember("bench", "\n\n".join(paragraphs), kind="file", source=source)
        chunks = await engine.documents.chunks_of("bench", added.episode_id)
        if [chunk.text.strip() for chunk in chunks] != paragraphs:
            raise SystemExit(f"{source}: expected one chunk per paragraph, got {len(chunks)} chunks")
        ids = [chunk.chunk_id for chunk in chunks]
        by_episode[added.episode_id] = ids
        texts.update({str(chunk.chunk_id): chunk.text for chunk in chunks})
        tree = await build_summary_tree(engine, FakeChat(scripted_replies(title, sections, ids, cite)), "bench", added.episode_id,
                                        fan_in=3, store=True)
        if tree.levels != 2 or tree.root is None or len(tree.nodes) != 4:
            raise SystemExit(f"{source}: the scripted tree did not come out as three sections and a root")
        for position, (theme, _) in enumerate(sections):
            questions.append((f"What does the {title} say about the {theme}?",
                              {str(chunk_id) for chunk_id in ids[position * 3:position * 3 + 3]}))
    return engine, questions, by_episode, texts


def covers(items: list[RecallItem], by_episode: dict[int, list[int]]) -> list[tuple[str, str, bool]]:
    """Every chunk under each summary in its place, first occurrence kept, as (id, text, is_summary)."""
    out: list[tuple[str, str, bool]] = []
    for item in items:
        meta = item.metadata
        if "summary_of" in meta:
            ids = by_episode[int(meta["summary_of"])]
            first, last = int(meta["summary_first_chunk"]), int(meta["summary_last_chunk"])
            under = [(str(c), "", False) for c in ids[ids.index(first):ids.index(last) + 1]]
        else:
            under = [(str(item.chunk_id), item.text, False)]
        out.extend(one for one in under if one[0] not in {kept[0] for kept in out})
    return out


def scored(listed: list[tuple[str, str, bool]], gold: set[str]) -> dict[str, float]:
    ids = [one[0] for one in listed]
    ten = listed[:LIMIT]
    return {"recall_at_10": recall_at(ids, gold, LIMIT), "recall_whole": recall_at(ids, gold, max(len(ids), 1)),
            "gold_in_ten": sum(1 for one in ten if one[0] in gold), "summaries_in_ten": sum(1 for one in ten if one[2]),
            "items": len(listed), "bytes_in_ten": sum(len(one[1].encode()) for one in ten)}


async def main(repeats: int, cite: int) -> None:
    engine, questions, by_episode, texts = await build(cite)
    arms = ("off", "follow", "replace")
    try:
        rows: dict[str, list[dict[str, float]]] = {arm: [] for arm in (*arms, "covers")}
        timings: dict[str, list[float]] = {arm: [] for arm in arms}
        for repeat in range(repeats):
            per_repeat: dict[str, list[float]] = {arm: [] for arm in arms}
            for question, gold in questions:
                for arm in arms:  # interleaved, so machine load falls on every arm alike
                    started = time.perf_counter()
                    found = await engine.recall("bench", question, limit=LIMIT,
                                                **({} if arm == "off" else {"expand_summaries": arm}))
                    per_repeat[arm].append((time.perf_counter() - started) * 1000)
                    if repeat:
                        continue
                    listed = [(str(item.chunk_id), item.text, "summary_of" in item.metadata) for item in found.items]
                    rows[arm].append(scored(listed, gold))
                    if arm == "off":
                        spread = [(ident, texts[ident], False) for ident, _, _ in covers(found.items, by_episode)]
                        rows["covers"].append(scored(spread, gold))
            for arm in arms:
                timings[arm].append(statistics.median(per_repeat[arm]))
        print(f"{len(questions)} broad questions, limit={LIMIT}, {repeats} repeat(s), level-one summaries citing {cite} of 3 paragraphs")
        print("| arm | recall_at(10) | recall, whole list | gold chunks in first 10 | summaries in first 10 | items returned | bytes in first 10 | ms per recall |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|")
        for arm in (*arms, "covers"):
            mean = {key: statistics.mean(row[key] for row in rows[arm]) for key in rows[arm][0]}
            ms = (f"{statistics.median(timings[arm]):.2f} ({', '.join(f'{t:.2f}' for t in timings[arm])})"
                  if arm in timings else "not timed")
            print(f"| {arm} | {mean['recall_at_10']:.3f} | {mean['recall_whole']:.3f} | {mean['gold_in_ten']:.2f} | "
                  f"{mean['summaries_in_ten']:.2f} | {mean['items']:.1f} | {mean['bytes_in_ten']:.0f} | {ms} |")
        print("means over questions; ms is the median over repeats of each repeat's median over questions (each repeat listed)")
        print("per question recall_at(10) off / follow / replace / covers:")
        for index, (question, _) in enumerate(questions):
            print(f"  {question}: " + " / ".join(f"{rows[arm][index]['recall_at_10']:.2f}" for arm in (*arms, "covers")))
    finally:
        await engine.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cite", type=int, choices=(1, 2, 3), default=2,
                        help="paragraphs of each section a level-one summary cites")
    arguments = parser.parse_args()
    asyncio.run(main(arguments.repeats, arguments.cite))
