"""Summary-tree descent on broad questions: recall_at(10) of the gold chunks, beside flat recall and expansion.

A reproducible local fixture, not a general retrieval-quality claim. Four
documents of four sections of five paragraphs each (twenty chunks a
document, one paragraph per chunk -- checked, not assumed) are stored with
``HashEmbedder``. A summary tree of fan_in 5 is built for each with
``FakeChat`` scripted as a model would answer: a level-one node per section
with a sentence for each of the first ``--cite`` (default two) paragraphs,
naming the section's theme and the document and quoting the paragraph's
opening words, then a root with a sentence per section quoting that
section's node. The theme word is in the section's first paragraph only, as
a heading word usually is. Each question asks what a document says about
one theme; its gold chunks are that section's five paragraphs.

Arms, all asking for ten chunks and scored on the first ten items returned:

- ``chunks only``: flat recall over a store holding the same documents and
  no trees.
- ``flat``: flat recall with the trees stored, so summaries are passages too.
- ``follow`` / ``replace``: ``expand_summaries`` with the default cap.
- ``tree bN``: ``tree_recall`` with ``branching=N``; ``+text`` adds BM25.
- ``tree b2, no stored vectors``: the same descent over an index that cannot
  hand back its vectors, so every step is embedded -- the cost of that path.
- ``llama_index select_leaf_embedding bN``: the reference framework's own tree
  retriever (``TreeSelectLeafEmbeddingRetriever``, installed and unmodified)
  over the same trees -- the same chunk and summary texts and the same
  parent-to-child links, read from the stored trees into its ``IndexGraph``
  -- embedding through the adapter the comparative runner uses around the
  same ``HashEmbedder``. Its ``child_branch_factor`` is ``N``; it returns the
  leaves it selects, so it returns at most ``N`` chunks.

Embedder calls and texts per question are counted by a wrapper around the
embedder, not by the traversal's own report. Latency is the median over
repeats of each repeat's median over questions, arms interleaved.

Run from packages/memory: ``PYTHONPATH=src python benchmarks/summary_traversal.py --repeats 3 --cite 2``.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from typing import Awaitable, Callable, Sequence

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.bench.metrics import recall_at
from scone_memory.core.models import RecallItem
from scone_memory.providers.llm import FakeChat
from scone_memory.retrieval.summary_tree import build_summary_tree, stored_summaries

CHUNK_TARGET = 200
LIMIT = 10
FAN_IN = 5

DOCUMENTS: list[tuple[str, str, list[tuple[str, list[str]]]]] = [
    ("Vellmar town report", "vellmar.md", [
        ("harbour", [
            "The harbour at Vellmar closes to sailing boats every November, when the winter swell starts to run in across the outer bar.",
            "Its lighthouse was rebuilt in 1904 after a storm took the first tower down to its foundations, and the lamp turns on a mercury bath.",
            "Pilots board arriving ships two miles out at the red buoy in every season, and the pilot cutter is moored beside the fish market steps.",
            "Dredging of the channel happens every third spring, when a hired barge lifts the silt that the river carries down onto the sand bar.",
            "Mooring fees are collected by the quay master each quarter day, and a boat left unpaid for a year is hauled out and sold at auction.",
        ]),
        ("orchard", [
            "The orchard on the ridge above Vellmar grows only Bramley apples, planted in rows of forty on terraces cut by hand in the last century.",
            "Picking starts in the final week of September and ends before the first frost, with school children given two days away from lessons.",
            "A cider press in the tithe barn turns out two thousand litres a season, sold at the Saturday market beside the old church wall.",
            "Pruning is done in January by volunteers from the allotment society, who burn the cuttings in a brazier at the foot of the terraces.",
            "Canker took a whole row of old trees in 2019, and the grafts that replaced them came from a nursery that still keeps the local stock.",
        ]),
        ("council", [
            "The Vellmar council meets on the first Tuesday of each month in the old customs house, and any resident may speak for three minutes.",
            "Seven members are elected every four years; the chair rotates each spring, and a tie is broken by drawing a name from a felt hat.",
            "Minutes are pinned in the post office window within a week and kept in bound ledgers that go back without a gap to the year 1871.",
            "The annual budget is set in February, and most of it goes on the sea wall, the street lamps and the grant to the lifeboat station.",
            "A planning objection must be lodged in writing within twenty-one days, and the clerk reads every one aloud before the members vote.",
        ]),
        ("school", [
            "The school in Vellmar has taught the children of the parish since 1882, in a granite building that still has its original bell tower.",
            "Forty-six pupils were on the roll this year across three classes, and the youngest walk down from the ridge farms with a teaching assistant.",
            "Swimming lessons are held in the harbour pool each June, once the water has warmed enough for the children to stay in for an hour.",
            "The headteacher publishes a letter at the end of every term, listing trips, prizes and the names of staff who are leaving or arriving.",
            "A new roof was paid for by a fete and a sponsored walk around the headland, and the builders finished the work over the summer holiday.",
        ]),
    ]),
    ("Brannock mill survey", "brannock.md", [
        ("river", [
            "The river Brann drives the mill wheel through a stone leat that was dug in 1760 and relined with brick after the great flood of 1952.",
            "In a dry August the flow falls below what the wheel needs, so the sluice is closed at night to let the pond fill again for the morning.",
            "Otters returned to the lower weir in 2011, and anglers are asked to keep away from the holt under the alder roots by the footbridge.",
            "A gauging board on the tail race is read every morning, and a level above the red line means the miller must lift all three hatches.",
            "Eels still come up the stream in May, and a pass of bristle brushes was fitted beside the weir so they can climb past the fall of water.",
        ]),
        ("machinery", [
            "The mill machinery is driven by an overshot wheel of fourteen feet, turning two pairs of French burr stones through its wooden gears.",
            "Cogs of apple wood are replaced every twelve years; the last set was cut by a wheelwright from Dunmore using the templates from 1890.",
            "A sack hoist lifts grain to the top floor, where it falls by gravity through chutes into the hoppers above the two turning stones.",
            "The stones are dressed by hand every spring, with a mill bill cutting fresh furrows so the grain is sheared rather than crushed flat.",
            "An iron governor lifts the upper stone as the wheel speeds up, which keeps the meal the same fineness whatever the water is doing.",
        ]),
        ("visitors", [
            "Visitors to the mill may tour it on Sundays from Easter to October, and the miller grinds a batch of wholemeal flour for sale at noon.",
            "School groups book through the parish office; each child leaves with a paper bag of flour and a pencil drawing of the big wheel.",
            "The tea room in the old drying kiln serves scones made from the mill's own flour, and dogs are welcome in the cobbled yard outside.",
            "Parking is in the field beyond the ford, and people with a blue badge may drive up to the gate and leave their car beside the barn.",
            "Guides are volunteers from the preservation trust, and each of them trains for a season before leading a tour on their own.",
        ]),
        ("flour", [
            "Flour from Brannock is stoneground wholemeal, milled from wheat grown on three farms within five miles of the mill house.",
            "Each bag is stamped with the day it was milled and should be used within three months, since the germ keeps its oil and turns.",
            "Bakers in two market towns buy it by the sack, and one of them has sold a Brannock loaf every Saturday for more than twenty years.",
            "A sieve of silk mesh can take out the coarsest bran for customers who ask for a lighter bag, though most prefer it left whole.",
            "Output is about four tonnes a year, limited by the water in summer and by how many days the volunteer miller can give to the work.",
        ]),
    ]),
    ("Hollin Fell weather station log", "hollin.md", [
        ("instruments", [
            "The weather station instruments on Hollin Fell include a Stevenson screen, a tipping-bucket rain gauge and a three-cup anemometer.",
            "Thermometers inside the louvred screen are read at nine each morning and checked against a calibrated reference once a year.",
            "The mast stands ten metres high and was guyed with steel cable after gales bent the original wooden pole in March 1987.",
            "A sunshine recorder with a glass sphere burns a trace onto a card, and the card is changed at dusk and filed in a drawer by month.",
            "Batteries for the logger are charged by a small solar panel, with a spare set kept in a tin box in case the winter days are too dark.",
        ]),
        ("rainfall", [
            "Rainfall on the fell averages 2,400 millimetres a year, most of it between October and February on wet south-westerly winds.",
            "The wettest day on record brought 187 millimetres in December 2015, and the beck below the station burst its banks before noon.",
            "The funnel of the gauge is cleared of leaves and midges every week, because a blocked funnel under-reads a heavy shower badly.",
            "A second gauge lower down the valley shows how much less falls in the lee of the ridge, often only two thirds of the total on top.",
            "Monthly totals are sent to the national archive on the first of the month, with a note of any day the reading had to be estimated.",
        ]),
        ("volunteers", [
            "Volunteers keep the station log in turns, walking up from the village of Thwaite by the old peat track in all weathers and seasons.",
            "Each observer signs the ledger, and a missed reading is marked with a dash rather than estimated from the neighbouring days.",
            "A retired shepherd, Agnes Birkett, kept the readings unbroken for thirty-one years and was given a county medal for it in 2009.",
            "New observers shadow an old hand for a month, learning to read a wet bulb and to reset the maximum thermometer without a jolt.",
            "The rota is drawn up each Christmas at the village hall, and anyone who cannot climb in the snow swaps their weeks for summer ones.",
        ]),
        ("snow", [
            "Snow on Hollin Fell lies for an average of forty days a winter, and drifts against the stone wall can bury the gate to the enclosure.",
            "Depth is measured with a graduated pole at three marked points, and the mean of the three is written in the ledger each morning.",
            "A fall is melted in a warm jug before it is added to the rain total, so heavy snow is counted as water rather than as lying depth.",
            "Observers carry a shovel in the porch of the hut, since the screen door has to be dug out before the thermometers can be read.",
            "The longest spell of lying cover ran from January to April in 1963, when the track was closed and readings were sent by telephone.",
        ]),
    ]),
    ("Saltmere reserve plan", "saltmere.md", [
        ("birds", [
            "Birds on the Saltmere reserve are counted monthly; winter brings up to nine thousand wigeon and a few hundred elegant pintail.",
            "Avocets nested on the scrape for the first time in 1998, and a wardened hide now overlooks the islands from the top of the sea wall.",
            "Ringing at the reed bed each autumn has recaptured sedge warblers first marked in Senegal on their long way south for the winter.",
            "Marsh harriers quarter the reeds on most spring mornings, and a pair has raised young in the same corner for eleven years running.",
            "Counts are entered into the county database within a week, and a rare visitor is reported to the recorder before it is posted online.",
        ]),
        ("grazing", [
            "Grazing on the marsh uses a herd of forty Red Poll cattle, turned out in May to keep the grass short for the nesting waders.",
            "The herd is moved between fields by a stockman on horseback, since vehicles would rut the soft ground and crush the eggs of waders.",
            "Water troughs are filled from a wind pump by the old sluice, and a float valve stops them overflowing onto the lapwing fields.",
            "The cattle are brought in to a farm on higher ground in October, before the autumn tides start to cover the lowest of the pastures.",
            "Rushes that the herd will not eat are topped with a light tractor in late summer, once the last of the chicks have fledged.",
        ]),
        ("access", [
            "Access to the Saltmere reserve is by the sea wall path from the quay; the car park fills early on sunny bank holiday weekends.",
            "The eastern trail closes from March to July so that ground-nesting lapwings are not disturbed by walkers and their loose dogs.",
            "A wheelchair-accessible boardwalk leads to the nearest hide, and binoculars can be borrowed from the visitor hut by the gate.",
            "The bus from the market town stops at the quay four times a day in summer, and twice a day from October to the end of March.",
            "Cycling is allowed on the sea wall path only, and a rack beside the visitor hut has room for twelve bicycles and a cargo trailer.",
        ]),
        ("flooding", [
            "Flooding of the Saltmere marsh is managed by a sluice in the sea wall, opened on falling tides to let the fresh water drain away.",
            "A surge in 2013 overtopped the wall at two low points, and salt water lay on the eastern fields for most of the following month.",
            "The wall was raised by half a metre along its weakest stretch, using clay dug from a borrow pit that has since become a new pool.",
            "Engineers inspect the sluice gates every winter and grease the winding gear, since a jammed gate could drown the spring nests.",
            "A long-term plan accepts that the lowest fields may return to salt marsh, and a line of higher ground has been bought inland.",
        ]),
    ]),
]


def notes(rows: list[tuple[str, str, str]]) -> str:
    return json.dumps({"notes": [{"sentence": s, "passage": p, "quote": q} for s, p, q in rows]})


def opening(text: str, count: int = 5) -> str:
    return " ".join(text.split()[:count])


def scripted_replies(title: str, sections: list[tuple[str, list[str]]], chunk_ids: list[int], cite: int) -> list[str]:
    """The replies a model would give, in the order the builder asks: a node per section, then the root."""
    replies, firsts = [], []
    for position, (theme, paragraphs) in enumerate(sections):
        ids = chunk_ids[position * FAN_IN:(position + 1) * FAN_IN]
        rows = [(f"On the {theme}, the {title} notes {opening(text)}.", f"chunk:{ids[i]}", opening(text))
                for i, text in enumerate(paragraphs[:cite])]
        replies.append(notes(rows))
        firsts.append(rows[0][0])
    replies.append(notes([(f"The {title} covers the {theme}.", f"node:1:{position}", firsts[position])
                          for position, (theme, _) in enumerate(sections)]))
    return replies


class Counting(HashEmbedder):
    """The engine's embedder, counting the calls and texts it answers."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.texts = 0

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        self.texts += len(texts)
        return await super().embed(texts)


class NoStoredVectors(InMemoryVectorIndex):
    vectors_of = None  # an index that cannot hand back what it stored


async def build(cite: int, *, trees: bool, index: InMemoryVectorIndex | None = None
                ) -> tuple[MemoryEngine, Counting, list[tuple[str, set[str]]]]:
    embedder = Counting()
    engine = await MemoryEngine(InMemoryDocumentStore(), index or InMemoryVectorIndex(), embedder,
                                chunk_target=CHUNK_TARGET).open()
    questions: list[tuple[str, set[str]]] = []
    for title, source, sections in DOCUMENTS:
        paragraphs = [text for _, texts in sections for text in texts]
        added = await engine.remember("bench", "\n\n".join(paragraphs), kind="file", source=source)
        chunks = await engine.documents.chunks_of("bench", added.episode_id)
        if [chunk.text.strip() for chunk in chunks] != paragraphs:
            raise SystemExit(f"{source}: expected one chunk per paragraph, got {len(chunks)} chunks")
        ids = [chunk.chunk_id for chunk in chunks]
        if trees:
            tree = await build_summary_tree(engine, FakeChat(scripted_replies(title, sections, ids, cite)), "bench",
                                            added.episode_id, fan_in=FAN_IN, store=True)
            if tree.levels != 2 or tree.root is None or len(tree.nodes) != len(sections) + 1:
                raise SystemExit(f"{source}: the scripted tree did not come out as {len(sections)} sections and a root")
        for position, (theme, _) in enumerate(sections):
            questions.append((f"What does the {title} say about the {theme}?",
                              {str(chunk_id) for chunk_id in ids[position * FAN_IN:(position + 1) * FAN_IN]}))
    return engine, embedder, questions


Arm = Callable[[str], Awaitable[Sequence[RecallItem]]]


async def llama_tree(engine: MemoryEngine, embedder: Counting) -> Callable[[int], Arm]:
    """The reference's tree index over the trees ``engine`` stored: every chunk and summary as one of its
    nodes, each summary's children the ones its account says it was written from, the tops its roots."""
    from llama_index.core import StorageContext, TreeIndex
    from llama_index.core.data_structs.data_structs import IndexGraph
    from llama_index.core.indices.tree.select_leaf_embedding_retriever import TreeSelectLeafEmbeddingRetriever
    from llama_index.core.llms import MockLLM
    from llama_index.core.schema import QueryBundle, TextNode

    from scone_memory.bench.comparative import SconeEmbedding

    nodes: dict[str, TextNode] = {}
    roots: list[str] = []
    graph = IndexGraph()
    episodes = await engine.documents.recent_episodes("bench", (await engine.documents.counts("bench")).episodes)
    for document in (episode for episode in episodes if episode.kind == "file"):
        chunks = await engine.documents.chunks_of("bench", document.episode_id)
        for chunk in chunks:
            nodes[f"chunk:{chunk.chunk_id}"] = TextNode(id_=f"chunk:{chunk.chunk_id}", text=chunk.text)
        summaries = await stored_summaries(engine, "bench", document.episode_id)
        top = max(summary.level for summary in summaries)
        for summary in summaries:
            name = f"{document.episode_id}:node:{summary.level}:{summary.index}"
            nodes[name] = TextNode(id_=name, text=summary.text)
            _, raw = await engine.attachment("bench", summary.detail)
            graph.node_id_to_children_ids[name] = [
                one if one.startswith("chunk:") else f"{document.episode_id}:{one}" for one in json.loads(raw)["written_from"]]
            if summary.level == top:
                roots.append(name)
    for position, name in enumerate(nodes):
        graph.all_nodes[position] = name
        graph.node_id_to_children_ids.setdefault(name, [])
    graph.root_nodes = {graph.node_id_to_index[name]: name for name in roots}
    storage = StorageContext.from_defaults()
    storage.docstore.add_documents(list(nodes.values()))
    index = TreeIndex(index_struct=graph, storage_context=storage, llm=MockLLM(), build_tree=False)
    adapter = SconeEmbedding(embedder)
    chunk_items = {name: node for name, node in nodes.items() if name.startswith("chunk:")}

    def arm(branching: int) -> Arm:
        retriever = TreeSelectLeafEmbeddingRetriever(index, embed_model=adapter, child_branch_factor=branching)

        async def ask(question: str) -> Sequence[RecallItem]:
            found = retriever.retrieve(QueryBundle(question))
            return [RecallItem(chunk_id=int(hit.node.node_id.split(":")[1]), episode_id=0, text=chunk_items[hit.node.node_id].text,
                               score=0.0, created_at="") for hit in found if hit.node.node_id in chunk_items]
        return ask
    return arm


async def main(repeats: int, cite: int) -> None:
    plain, plain_embedder, questions = await build(cite, trees=False)
    engine, embedder, same = await build(cite, trees=True)
    blind, blind_embedder, unindexed = await build(cite, trees=True, index=NoStoredVectors())
    if not [question for question, _ in same] == [question for question, _ in unindexed] == [question for question, _ in questions]:
        raise SystemExit("the stores disagree on the questions")
    # Chunk ids differ between stores, so each arm is scored against its own store's gold.
    gold_of = {id(plain): dict(questions), id(engine): dict(same), id(blind): dict(unindexed)}

    def recall_arm(on: MemoryEngine, **options: object) -> Arm:
        async def ask(question: str) -> Sequence[RecallItem]:
            return (await on.recall("bench", question, limit=LIMIT, **options)).items  # type: ignore[arg-type]
        return ask

    def tree_arm(on: MemoryEngine, **options: object) -> Arm:
        async def ask(question: str) -> Sequence[RecallItem]:
            return (await on.tree_recall("bench", question, limit=LIMIT, **options)).items  # type: ignore[arg-type]
        return ask

    arms: dict[str, tuple[MemoryEngine, Counting, Arm]] = {
        "chunks only": (plain, plain_embedder, recall_arm(plain)),
        "flat": (engine, embedder, recall_arm(engine)),
        "follow": (engine, embedder, recall_arm(engine, expand_summaries="follow")),
        "replace": (engine, embedder, recall_arm(engine, expand_summaries="replace")),
        "tree b1": (engine, embedder, tree_arm(engine, branching=1)),
        "tree b2": (engine, embedder, tree_arm(engine, branching=2)),
        "tree b3": (engine, embedder, tree_arm(engine, branching=3)),
        "tree b2 +text": (engine, embedder, tree_arm(engine, branching=2, text=True)),
        "tree b2, no stored vectors": (blind, blind_embedder, tree_arm(blind, branching=2)),
    }
    reference = await llama_tree(engine, embedder)
    for branching in (1, 2, 10):
        arms[f"llama_index select_leaf_embedding b{branching}"] = (engine, embedder, reference(branching))
    try:
        scores: dict[str, list[dict[str, float]]] = {name: [] for name in arms}
        timings: dict[str, list[float]] = {name: [] for name in arms}
        for repeat in range(repeats):
            per_repeat: dict[str, list[float]] = {name: [] for name in arms}
            for question, _ in questions:
                for name, (on, counting, ask) in arms.items():  # interleaved, so load falls on every arm alike
                    calls, texts = counting.calls, counting.texts
                    started = time.perf_counter()
                    items = await ask(question)
                    per_repeat[name].append((time.perf_counter() - started) * 1000)
                    if repeat:
                        continue
                    gold = gold_of[id(on)][question]
                    ids = [str(item.chunk_id) for item in items]
                    scores[name].append({
                        "recall_at_10": recall_at(ids, gold, LIMIT),
                        "gold_in_ten": float(sum(1 for one in ids[:LIMIT] if one in gold)),
                        "summaries_in_ten": float(sum(1 for item in items[:LIMIT] if "summary_of" in item.metadata)),
                        "items": float(len(items)),
                        "embed_calls": float(counting.calls - calls),
                        "embedded_texts": float(counting.texts - texts)})
            for name in arms:
                timings[name].append(statistics.median(per_repeat[name]))
        print(f"{len(questions)} broad questions over {len(DOCUMENTS)} documents of {len(DOCUMENTS[0][2])} sections of "
              f"{FAN_IN} chunks, limit={LIMIT}, level-one summaries citing {cite} of {FAN_IN} paragraphs, {repeats} repeat(s)")
        print("| arm | recall_at(10) | gold chunks in first 10 | summaries in first 10 | items returned | "
              "embedder calls / question | texts embedded / question | ms / question |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|")
        for name in arms:
            mean = {key: statistics.mean(row[key] for row in scores[name]) for key in scores[name][0]}
            ms = f"{statistics.median(timings[name]):.2f} ({', '.join(f'{t:.2f}' for t in timings[name])})"
            print(f"| {name} | {mean['recall_at_10']:.3f} | {mean['gold_in_ten']:.2f} | {mean['summaries_in_ten']:.2f} | "
                  f"{mean['items']:.1f} | {mean['embed_calls']:.2f} | {mean['embedded_texts']:.1f} | {ms} |")
        print("means over questions; ms is the median over repeats of each repeat's median over questions (each repeat listed)")
        print("per question recall_at(10): " + " / ".join(arms))
        for index, (question, _) in enumerate(questions):
            print(f"  {question}: " + " / ".join(f"{scores[name][index]['recall_at_10']:.2f}" for name in arms))
    finally:
        for on in (plain, engine, blind):
            await on.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cite", type=int, choices=(1, 2, 3, 4, 5), default=2,
                        help="paragraphs of each section a level-one summary cites")
    arguments = parser.parse_args()
    asyncio.run(main(arguments.repeats, arguments.cite))
