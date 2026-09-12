"""What recall does with code, measured on a corpus of source files.

Every source file under a root is stored as one record with its path as
its source, and every documented function becomes a question. Two ways
of asking, over the same functions:

* **docstring** — the docstring's first paragraph, as written. This is
  close to an exact-match measurement: those words are in the chunk,
  verbatim, so a high score mostly says the lexical lane finds text it
  has. It is still worth knowing, because it says whether chunking a
  file buries a function.
* **name** — "what <the function's name, in words> does". Nobody wrote
  this sentence anywhere, so it is the one that says something about
  retrieval by intent.

A hit is a returned chunk that holds the function's own ``def`` line.
``whole`` counts the hits whose chunk is that declaration and nothing
else, which is what declaration-aware chunking is for: the answer is a
function, not the end of one and the start of the next. Nothing here is
paid or fetched: the hash embedder, an in-memory store, and the files on
disk. A question is never asked of a store it was not built from.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence

from ..backends import InMemoryDocumentStore, InMemoryVectorIndex
from ..embedders.hash import HashEmbedder
from ..ingestion.chunker import DEFAULT_TARGET
from ..ingestion.code import PYTHON_SUFFIXES, declarations
from ..memory.engine import MemoryEngine

#: A docstring shorter than this says too little to ask about.
MIN_WORDS = 6
#: A corpus this large is a mistake, not a measurement.
MAX_FILES = 5_000
#: Past this a file is a build artefact rather than something to read.
MAX_FILE_BYTES = 400_000


@dataclass(frozen=True)
class CodeScore:
    """What a run found, and what it was asked."""

    files: int = 0
    questions: int = 0
    #: Questions whose own definition came back in the top k.
    found: int = 0
    #: Questions whose file came back at all.
    same_file: int = 0
    #: Of the hits, the ones whose chunk is that declaration alone.
    whole: int = 0
    k: int = 5
    asked: str = "docstring"
    code_aware: bool = True
    #: The chunk size the corpus was stored at, which is a retrieval
    #: setting like any other and is reported with the numbers it made.
    chunk_target: int = DEFAULT_TARGET

    def record(self) -> dict[str, object]:
        return {"files": self.files, "questions": self.questions, "found": self.found,
                "same_file": self.same_file, "whole": self.whole, "k": self.k,
                "asked": self.asked, "code_aware": self.code_aware,
                "chunk_target": self.chunk_target}

    def text(self) -> str:
        """One line for a person: what came back, out of how many."""
        if not self.questions:
            return f"{self.files} file(s), no documented function to ask about"
        return (f"{self.questions} question(s) over {self.files} file(s), asked by {self.asked}, "
                f"{'cut at declarations' if self.code_aware else 'cut by length'}: "
                f"own definition in top {self.k} {self.found} ({self.found / self.questions:.0%}), "
                f"own file {self.same_file} ({self.same_file / self.questions:.0%}), "
                f"a whole declaration {self.whole} ({self.whole / self.questions:.0%})")


@dataclass(frozen=True)
class _Asked:
    question: str
    definition: str
    source: str


def sources(root: str | Path, suffixes: Sequence[str] = PYTHON_SUFFIXES) -> list[Path]:
    """Every source file under a root, in a settled order."""
    found: list[Path] = []
    for path in sorted(Path(root).rglob("*")):
        if (path.is_file() and path.suffix in suffixes
                and not any(part.startswith(".") or part == "__pycache__" for part in path.parts)):
            found.append(path)
        if len(found) >= MAX_FILES:
            break
    return found


def questions(root: str | Path, files: Sequence[Path], *, asked: str = "docstring") -> Iterator[_Asked]:
    """One question per documented function whose docstring does not give
    its own name away."""
    for path in files:
        text = _read(path)
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            said = ast.get_docstring(node)
            if not said:
                continue
            first = " ".join(said.strip().split("\n\n")[0].split())
            if len(first.split()) < MIN_WORDS or node.name in first:
                continue
            words = node.name.strip("_").replace("_", " ")
            yield _Asked(first if asked == "docstring" else f"what {words} does",
                         f"def {node.name}(", str(Path(path).relative_to(root)))


async def run_code_bench(root: str | Path, *, k: int = 5, limit: Optional[int] = None,
                         asked: str = "docstring", code_aware: bool = True,
                         chunk_target: int = DEFAULT_TARGET,
                         suffixes: Sequence[str] = PYTHON_SUFFIXES) -> CodeScore:
    """Store the corpus, ask every question of it, and count what came back."""
    if k < 1:
        raise ValueError("k must be at least 1")
    if asked not in ("docstring", "name"):
        raise ValueError("asked must be 'docstring' or 'name'")
    files = sources(root, suffixes)
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder(),
                                code_aware=code_aware, chunk_target=chunk_target).open()
    try:
        for path in files:
            text = _read(path)
            if text.strip():
                await engine.remember("code", text, kind="file", source=str(path.relative_to(root)))
        found = same_file = whole = counted = 0
        for question in list(questions(root, files, asked=asked))[:limit]:
            items = (await engine.recall("code", question.question, limit=k)).items
            counted += 1
            hit = [item for item in items if question.definition in item.text]
            found += bool(hit)
            same_file += any((item.source or "") == question.source for item in items)
            whole += any(_is_one_declaration(item.text) for item in hit)
        return CodeScore(files=len(files), questions=counted, found=found, same_file=same_file,
                         whole=whole, k=k, asked=asked, code_aware=code_aware,
                         chunk_target=chunk_target)
    finally:
        await engine.close()


def _read(path: Path) -> str:
    raw = path.read_bytes()[:MAX_FILE_BYTES]
    return raw.decode("utf-8", errors="replace")


def _is_one_declaration(text: str) -> bool:
    """Whether a chunk is exactly one declaration and nothing else."""
    found = declarations(text.lstrip("\n"), language="python")
    if not found:
        return False
    first = found[0]
    body = text.lstrip("\n")
    return first.start == 0 and body[first.end :].strip() == ""
