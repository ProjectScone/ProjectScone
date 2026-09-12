"""Second-hop passage recall with and without the entity lane.

A deterministic synthetic set, version ``bridge-v1``: each case is a
person who works at an organisation based in a city. One passage says
where the person works; another says where the organisation is, naming
neither the person nor any word of the question. Distractor passages
repeat the question's own words. The ledger records both relations.

For "Which city is <person>'s employer in?", the second-hop passage is
the answer's evidence. The benchmark reports how often it appears in the
top five with the entity lane off and on, and the same for the first-hop
passage, which the other lanes find unaided.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

VERSION = "bridge-v1"
_FIRST = ["Ana", "Bruno", "Carla", "Duarte", "Elena", "Filipe", "Gloria", "Hugo", "Ines", "Joao",
          "Karin", "Luis", "Marta", "Nuno", "Olga", "Pedro", "Quinta", "Rita", "Sergio", "Tania"]
_LAST = ["Alves", "Barros", "Costa", "Dias", "Esteves", "Faria", "Gomes", "Horta", "Isidro", "Jardim",
         "Leal", "Matos", "Neves", "Osorio", "Pires", "Quental", "Ramos", "Sousa", "Tavares", "Vidal"]
_ORGS = ["Aurora", "Basalt", "Cobalt", "Delta", "Ember", "Fjord", "Granite", "Harbor", "Indigo", "Juniper",
         "Kestrel", "Lumen", "Meridian", "Nimbus", "Orchard", "Pioneer", "Quartz", "Radian", "Summit", "Tidal"]
_KINDS = ["Labs", "Works", "Systems", "Foundry"]
_CITIES = ["Lisbon", "Porto", "Braga", "Faro", "Coimbra", "Evora", "Aveiro", "Leiria", "Viseu", "Setubal"]


@dataclass(frozen=True)
class BridgeReport:
    version: str
    cases: int
    second_hop_recall_off: float
    second_hop_recall_on: float
    first_hop_recall_off: float
    first_hop_recall_on: float

    def record(self) -> dict[str, object]:
        return asdict(self)


def cases(count: int = 20) -> list[dict[str, str]]:
    return [{"person": f"{_FIRST[i % 20]} {_LAST[(i * 7) % 20]}",
             "org": f"{_ORGS[(i * 3) % 20]} {_KINDS[i % 4]}", "city": _CITIES[(i * 3) % 10]} for i in range(count)]


async def run_bridge_benchmark(count: int = 20, limit: int = 5) -> BridgeReport:
    from ..backends.memory import InMemoryDocumentStore, InMemoryVectorIndex
    from ..embedders.hash import HashEmbedder
    from ..memory.engine import MemoryEngine
    from . import Clock

    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(), clock=Clock()).open()
    try:
        first: dict[str, int] = {}
        second: dict[str, int] = {}
        chosen = cases(count)
        for case in chosen:
            first[case["person"]] = (await engine.remember(
                "bench", f"{case['person']} joined {case['org']} after a long search.")).episode_id
            second[case["person"]] = (await engine.remember(
                "bench", f"{case['org']} keeps its headquarters in {case['city']}, near the old harbour.")).episode_id
            await engine.assert_fact("bench", case["person"].casefold(), "works_at", case["org"],
                                     valid_from="2024-01-01T00:00:00Z")
            await engine.assert_fact("bench", case["org"].casefold(), "based_in", case["city"],
                                     valid_from="2024-01-01T00:00:00Z")
        for number in range(30):
            await engine.remember("bench", f"Survey {number}: which city does each employer prefer, and why?")
        hits = {"first_off": 0, "first_on": 0, "second_off": 0, "second_on": 0}
        for case in chosen:
            question = f"Which city is {case['person']}'s employer in?"
            for mode, boost in (("off", False), ("on", True)):
                found = {item.episode_id for item in (await engine.recall("bench", question, limit=limit,
                                                                           graph_boost=boost)).items}
                hits[f"first_{mode}"] += first[case["person"]] in found
                hits[f"second_{mode}"] += second[case["person"]] in found
    finally:
        await engine.close()
    return BridgeReport(VERSION, count, round(hits["second_off"] / count, 4), round(hits["second_on"] / count, 4),
                        round(hits["first_off"] / count, 4), round(hits["first_on"] / count, 4))
