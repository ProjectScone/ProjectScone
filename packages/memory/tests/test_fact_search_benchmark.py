"""The scale diagnostic has real scoped targets and reports fallback scans."""
from pathlib import Path
from typing import cast

import pytest

from scone_memory.testing.fact_search_benchmark import benchmark_size, main


async def test_fixture_requires_late_records_and_filters_before_limiting() -> None:
    report = await benchmark_size(100, 1)
    cases = cast(list[dict[str, object]], report["cases"])
    expected = [[98, 99, 100], [1, 2, 3, 4, 5, 6, 7, 8, 9, 11], [10, 20, 30, 40, 50, 60, 70, 80, 90, 99]]
    for case, identifiers in zip(cases, expected, strict=True):
        scan = cast(dict[str, object], case["scan"])
        indexed = cast(dict[str, object], case["indexed"])
        assert scan["fact_ids"] == indexed["fact_ids"] == identifiers
        assert scan["ledger_rows_materialized"] == 100
        assert indexed["ledger_rows_materialized"] == 0
        assert indexed["fact_point_reads"] == len(identifiers)


def test_report_cannot_replace_existing_results(tmp_path: Path) -> None:
    report = tmp_path/"report.json"
    report.write_text("original measurement")
    with pytest.raises(FileExistsError):
        main(["--sizes", "100", "--repeats", "1", "--output", str(report)])
    assert report.read_text() == "original measurement"


@pytest.mark.parametrize("size,repeats", [(99, 1), (250001, 1), (100, 0), (100, 21), (True, 1)])
async def test_invalid_workloads_fail_before_creating_a_store(size: int, repeats: int) -> None:
    with pytest.raises(ValueError):
        await benchmark_size(size, repeats)
