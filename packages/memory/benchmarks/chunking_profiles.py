"""Plain structure chunking against a chunking profile, on generated genre fixtures.

Two deterministic fixtures -- a statute and a Q&A file, from a fixed seed --
are cut three ways at the default target: by length, by plain structure and
under the matching profile. All three are scored against the profile's own
units, so they answer the same questions:

- how many chunks start at a genre boundary;
- how often a part or article (or a question) lands in a different chunk
  from its first clause (or its answer);
- how often a unit that fits the target is split anyway.

With ``--corpus DIR`` it also runs every profile over the Markdown files in
DIR and prints how many lines each rule matched, which is how a rule that is
right on its fixture but fires on everything shows itself.

No model, no store: this measures cut positions, not answers.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import random
from typing import Callable

from scone_memory.ingestion.chunker import DEFAULT_TARGET, Span, chunk_spans
from scone_memory.ingestion.chunking_profiles import PROFILES, profile_named, profiled_spans
from scone_memory.ingestion.structure_chunks import Unit, structured_spans, units

SEED = 20260914
WORDS = ("controller processor record transfer authority request person data purpose consent measure "
         "security breach notice period member body contract assessment register officer").split()


class Fixtures:
    def __init__(self, seed: int = SEED) -> None:
        self.rng = random.Random(seed)

    def sentence(self, words: int) -> str:
        chosen = [self.rng.choice(WORDS) for _ in range(words)]
        return chosen[0].capitalize() + " " + " ".join(chosen[1:]) + "."

    def prose(self, low: int, high: int) -> str:
        out: list[str] = []
        size = self.rng.randint(low, high)
        while sum(len(s) + 1 for s in out) < size:
            out.append(self.sentence(self.rng.randint(8, 18)))
        return " ".join(out)

    def title(self) -> str:
        return " ".join(self.rng.choice(WORDS) for _ in range(3))

    def statute(self) -> str:
        parts = ["DATA RECORDS ACT\n\n"]
        article = 0
        for part in range(1, 4):
            parts.append(f"PART {part}\n{self.title().upper()}\n\n")
            for _ in range(self.rng.randint(3, 6)):
                article += 1
                parts.append(f"Article {article}\n{self.title().capitalize()}\n\n")
                style = self.rng.choice(["numbered", "lettered", "both", "prose"])
                if style == "prose":
                    parts.append(self.prose(120, 900) + "\n\n")
                    continue
                for n in range(1, self.rng.randint(2, 5)):
                    if style in ("numbered", "both"):
                        parts.append(f"{n}. {self.prose(60, 400)}\n")
                    if style in ("lettered", "both"):
                        for letter in "abcd"[:self.rng.randint(2, 4)]:
                            parts.append(f"({letter}) {self.prose(40, 300)}\n")
                    if style == "lettered":
                        break
                parts.append("\n")
        return "".join(parts)

    def faq(self) -> str:
        out = ["# Frequently asked questions\n\n"]
        for _ in range(24):
            question = " ".join(self.rng.choice(WORDS) for _ in range(self.rng.randint(4, 9))).capitalize() + "?"
            answer = self.prose(40, 650)
            if self.rng.random() < 0.3:
                answer += "\n\n" + self.prose(80, 300)
            out.append(f"Q: {question}\nA: {answer}\n\n")
        return "".join(out)


def holding(spans: list[Span], offset: int) -> int:
    return next(index for index, span in enumerate(spans) if span.start <= offset < span.end)


def first_deeper(content: str, read: tuple[Unit, ...], index: int) -> int | None:
    after = index + 1
    return read[after].start if after < len(read) and read[after].depth > read[index].depth else None


def answer_of(content: str, read: tuple[Unit, ...], index: int) -> int | None:
    at = content.find("\nA:", read[index].start)
    return at + 1 if at != -1 else None


def score(label: str, content: str, spans: list[Span], profile: str, parents: set[str],
          child_of: Callable[[str, tuple[Unit, ...], int], int | None]) -> None:
    read = units(content, reader=profile_named(profile).reader())
    starts = {unit.start for unit in read if unit.kind not in ("text", "table")}
    begun = sum(1 for span in spans if span.start in starts)
    split = checked = fits = fit_split = 0
    for index, unit in enumerate(read):
        if unit.kind not in parents:
            continue
        child = child_of(content, read, index)
        if child is None:
            continue
        checked += 1
        split += holding(spans, unit.start) != holding(spans, child)
        after = next((later.start for later in read[index + 1:] if later.depth <= unit.depth), len(content))
        if after - unit.start <= DEFAULT_TARGET:
            fits += 1
            fit_split += holding(spans, unit.start) != holding(spans, after - 1)
    sizes = [span.end - span.start for span in spans]
    print(f"  {label:10} chunks {len(spans):4}  at a genre boundary {begun}/{len(spans)} ({begun / len(spans):.0%})"
          f"  split from first child {split}/{checked}  fitting unit split {fit_split}/{fits}"
          f"  over {DEFAULT_TARGET}: {sum(1 for size in sizes if size > DEFAULT_TARGET)}  mean {sum(sizes) // len(sizes)}")


def fixtures() -> None:
    made = Fixtures()
    for label, content, profile, parents, child in [
        ("statute-like", made.statute(), "statute", {"part", "article"}, first_deeper),
        ("Q&A-like", made.faq(), "qa", {"question"}, answer_of),
    ]:
        print(f"{label}: {len(content)} characters, profile {profile}, seed {SEED}")
        score("length", content, chunk_spans(content), profile, parents, child)
        score("structure", content, list(structured_spans(content).spans), profile, parents, child)
        found = profiled_spans(content, profile=profile)
        score("profile", content, list(found.spans), profile, parents, child)
        print("  receipt:", {key: value for key, value in found.record().items()
                             if key in ("matched", "began", "by_size", "over_target")})


def corpus(directory: Path) -> None:
    documents = [path.read_text() for path in sorted(directory.glob("*.md"))]
    print(f"{directory}: {len(documents)} documents, {sum(len(text) for text in documents)} characters")
    print(f"  plain structure: {sum(len(structured_spans(text).spans) for text in documents)} chunks")
    for name in PROFILES:
        matched: Counter[str] = Counter()
        chunks = 0
        for text in documents:
            found = profiled_spans(text, profile=name)
            matched.update(found.matched)
            chunks += len(found.spans)
        print(f"  {name:8} {chunks} chunks, rules matched {dict(matched)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, help="also run every profile over the Markdown files here")
    args = parser.parse_args()
    fixtures()
    if args.corpus is not None:
        corpus(args.corpus)


if __name__ == "__main__":
    main()
