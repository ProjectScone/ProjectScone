"""Retrieval settings chosen by measurement, not by taste."""

import pytest

from scone_memory.bench.tune import (
    DEFAULT_SETTINGS,
    Measured,
    Setting,
    chosen_setting,
    tune,
)

pytestmark = pytest.mark.asyncio


def written(tmp_path):
    """A two-question dataset in the shape the bench reads."""
    import json

    from tests.benchmarks.test_bench import DATASET

    path = tmp_path / "items.json"
    path.write_text(json.dumps(DATASET[:2]), encoding="utf-8")
    return path


def measured(setting: Setting, recall: float, p50: float = 10.0) -> Measured:
    return Measured(setting=setting, questions=10, recall_any=recall, recall_all=recall,
                    recall_ms_p50=p50, errors=0)


def test_the_setting_that_found_most_is_the_one_taken():
    rows = [measured(DEFAULT_SETTINGS, 0.6), measured(Setting(candidate_limit=200), 0.8)]
    taken, why = chosen_setting(rows)
    assert taken.candidate_limit == 200 and "0.8" in why and "0.6" in why


def test_a_setting_that_only_matches_the_default_does_not_displace_it():
    """Changing a setting to find exactly what the default found is a
    change for nothing, and the answer says so."""
    rows = [measured(DEFAULT_SETTINGS, 0.7), measured(Setting(candidate_limit=200), 0.7)]
    taken, why = chosen_setting(rows)
    assert taken == DEFAULT_SETTINGS and "nothing measured better" in why


def test_a_tie_between_two_changes_goes_to_the_quicker_one():
    slow = Setting(candidate_limit=400)
    quick = Setting(candidate_limit=100)
    rows = [measured(DEFAULT_SETTINGS, 0.5), measured(slow, 0.9, p50=40.0), measured(quick, 0.9, p50=12.0)]
    taken, why = chosen_setting(rows)
    assert taken == quick and "quicker" in why


def test_a_setting_that_was_not_timed_does_not_win_a_tie_on_that():
    """A missing timing is not a fast one. The row that was timed wins."""
    timed = Setting(candidate_limit=100)
    rows = [measured(DEFAULT_SETTINGS, 0.4),
            Measured(setting=Setting(candidate_limit=200), questions=10, recall_any=0.9,
                     recall_all=0.9, recall_ms_p50=None, errors=0),
            measured(timed, 0.9, p50=30.0)]
    taken, _ = chosen_setting(rows)
    assert taken == timed


def test_a_sweep_with_nothing_measured_chooses_nothing():
    taken, why = chosen_setting([])
    assert taken is None and "nothing was measured" in why


def test_a_setting_says_how_to_put_it_in_force():
    lines = Setting(candidate_limit=200, demote_restated=False, contextual_embeddings=True).environment()
    assert lines == ["SCONE_RECALL_CANDIDATES=200", "SCONE_DEMOTE_RESTATED=0",
                     "SCONE_CONTEXTUAL_EMBEDDINGS=1"]
    assert DEFAULT_SETTINGS.environment() == [], "the default needs nothing said"


async def test_a_sweep_measures_every_setting_it_names(tmp_path):
    dataset = written(tmp_path)
    swept = [DEFAULT_SETTINGS, Setting(candidate_limit=100)]
    report = await tune(dataset, settings=swept, k=5, sample=2)
    assert [row.setting for row in report.rows] == swept
    assert all(row.questions == 2 for row in report.rows)
    assert report.embedder and report.dataset == str(dataset)
    assert report.seed == 42 and report.record()["seed"] == 42, "a sample can be drawn again"
    assert report.chosen is not None or "nothing" in report.reason


async def test_a_sweep_refuses_a_k_it_cannot_measure(tmp_path):
    from scone_memory.core.errors import InvalidInput

    with pytest.raises(InvalidInput):
        await tune(written(tmp_path), settings=[DEFAULT_SETTINGS], k=0, sample=1)


async def test_a_sweep_of_no_settings_is_refused(tmp_path):
    from scone_memory.core.errors import InvalidInput

    with pytest.raises(InvalidInput):
        await tune(written(tmp_path), settings=[], k=5, sample=1)


async def test_every_setting_really_runs_under_itself(tmp_path):
    """A sweep that reported numbers without applying the settings would
    look exactly like a sweep where nothing mattered. One candidate per
    lane cannot find both sessions a two-session question needs, so the
    two rows must differ."""
    report = await tune(written(tmp_path), settings=[DEFAULT_SETTINGS, Setting(candidate_limit=1)],
                        k=5, sample=2)
    defaults, narrowed = report.rows
    assert defaults.recall_all == 1.0 and narrowed.recall_all == 0.5
    assert report.chosen == DEFAULT_SETTINGS
