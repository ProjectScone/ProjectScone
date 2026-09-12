"""Measuring what recall does with code, on a corpus of source files."""

import pytest

from scone_memory.bench.code import CodeScore, run_code_bench

pytestmark = pytest.mark.asyncio

PLANNER = '''"""The planner."""


def plan(question: str) -> str:
    """Work out what is being asked, and say so in one plain line."""
    return question.strip().lower()


def widen(query: str, extra: list[str]) -> str:
    """Add the words a person left out, so a search finds more of them."""
    return " ".join([query, *extra])
'''

STORE = '''"""Where things are kept."""


class Shelf:
    """Holds records until they are asked for."""

    def put(self, record: str) -> None:
        """Keep a record, and say nothing about it afterwards."""
        self.records.append(record)

    def take(self, name: str) -> str:
        """Hand back the record that was kept under that name."""
        return next(r for r in self.records if r.startswith(name))
'''


@pytest.fixture()
def corpus(tmp_path):
    (tmp_path / "planner.py").write_text(PLANNER, encoding="utf-8")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "store.py").write_text(STORE, encoding="utf-8")
    (tmp_path / "notes.md").write_text("A note that quotes `def plan(question)` and no more.\n", encoding="utf-8")
    return tmp_path


async def test_every_documented_function_becomes_a_question(corpus):
    score = await run_code_bench(corpus, k=5)
    assert score.questions == 4, score
    assert score.files == 2, "only source files are the corpus"


async def test_a_hit_is_the_functions_own_definition_coming_back(corpus):
    score = await run_code_bench(corpus, k=5, chunk_target=200)
    assert score.found == score.questions, score
    assert score.same_file == score.questions
    assert 0 < score.whole <= score.questions


async def test_asking_in_a_persons_words_is_scored_apart(corpus):
    written = await run_code_bench(corpus, k=5, asked="docstring")
    by_name = await run_code_bench(corpus, k=5, asked="name")
    assert by_name.questions == written.questions
    assert by_name.asked == "name" and written.asked == "docstring"
    assert by_name.found <= written.found, "the docstring is in the chunk verbatim"


async def test_the_ordinary_chunker_can_be_measured_beside_it(corpus):
    aware = await run_code_bench(corpus, k=5, chunk_target=200)
    ordinary = await run_code_bench(corpus, k=5, code_aware=False, chunk_target=200)
    assert aware.code_aware and not ordinary.code_aware
    assert ordinary.questions == aware.questions
    assert aware.whole >= ordinary.whole


async def test_a_corpus_with_nothing_to_ask_about_scores_nothing(tmp_path):
    (tmp_path / "empty.py").write_text("x = 1\n", encoding="utf-8")
    score = await run_code_bench(tmp_path, k=5)
    assert score == CodeScore(files=1, files_total=1, k=5, asked="docstring", code_aware=True)


async def test_a_question_limit_is_honoured(corpus):
    score = await run_code_bench(corpus, k=5, limit=2)
    assert score.questions == 2


async def test_a_corpus_larger_than_the_bench_reads_says_how_much_it_left(corpus, monkeypatch):
    """A bench that reads 5,000 files of a 12,000-file repository and then
    reports 5,000 has told you the size of its own bound."""
    from scone_memory.bench import code as bench

    monkeypatch.setattr(bench, "MAX_FILES", 1)
    score = await run_code_bench(corpus, k=5)
    assert score.files == 1 and score.files_total == 2
    assert score.record()["files_total"] == 2
    assert "1 file(s) of 2" in score.text(), score.text()


async def test_a_file_too_long_to_read_whole_is_counted(corpus, monkeypatch):
    from scone_memory.bench import code as bench

    monkeypatch.setattr(bench, "MAX_FILE_BYTES", 40)
    score = await run_code_bench(corpus, k=5)
    assert score.files_cut == 2, "both were longer than that"
    assert "cut" in score.text()
