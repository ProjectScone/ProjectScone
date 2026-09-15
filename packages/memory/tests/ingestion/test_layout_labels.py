"""What each region of a page is: an engine's boxes on one vocabulary, or the rules."""
import pytest

from scone_memory.ocr.labels import (ENGINE_LABELS, LayoutLabels, as_label, infer_labels, label_from_engine,
                                     running_key)
from scone_memory.ocr.types import REGION_LABELS, LayoutRegion, LayoutResult, OcrRegion


def line(text: str, top: float, height: float = 0.02, left: float = 0.1, right: float = 0.9, key: int = 0):
    return OcrRegion(text=text, box=(left, top, right, top + height), block=0, paragraph=0, line=key)


def words(text: str, top: float, height: float = 0.02, key: int = 0, left: float = 0.1):
    """One line as the words a recognizer gives, side by side."""
    made, x = [], left
    for word in text.split():
        width = 0.01 * len(word)
        made.append(OcrRegion(text=word, box=(x, top, x + width, top + height), block=0, paragraph=0, line=key))
        x += width + 0.006
    return made


def test_the_rules_tell_a_page_number_running_lines_a_title_headings_lists_and_footnotes():
    page = [
        *words("Acme Robotics Annual Report", 0.05, height=0.04, key=1),
        *words("1 Introduction", 0.16, height=0.03, key=2),
        *words("The year was good and the robots were many in every warehouse.", 0.20, key=3),
        *words("They worked and they rested and the report goes on at length here.", 0.225, key=4),
        *words("• first item of the list", 0.30, key=5),
        *words("• second item of the list", 0.325, key=6),
        *words("Acme confidential", 0.95, key=7),
        *words("12", 0.97, key=8),
        *words("1 See the appendix for the method", 0.90, height=0.015, key=9),
    ]
    labels, receipt = infer_labels(page, first_page=True, running={"Acme confidential"})
    by_key = {}
    for region, label in zip(page, labels):
        by_key.setdefault(region.line, set()).add(label)
    assert by_key == {1: {"title"}, 2: {"heading"}, 3: {"paragraph"}, 4: {"paragraph"}, 5: {"list"}, 6: {"list"},
                      7: {"footer"}, 8: {"page_number"}, 9: {"footnote"}}, by_key
    assert receipt.source == "inferred" and receipt.strategy == "labels-v1"
    assert set(receipt.rules) == {"page_number", "running", "list", "title", "heading", "footnote"}
    assert receipt.counts["paragraph"] == 25 and receipt.counts["title"] == 4 and receipt.unlabeled == 0
    second, _ = infer_labels(page, first_page=False, running={"Acme confidential"})
    assert "title" not in second and second[0] == "heading", "a title is the first page's; elsewhere its size is a heading"


def test_a_text_layer_page_tells_a_heading_by_its_isolation_and_a_lone_short_line_is_not_one():
    grid = [line("Methods", 0.30, key=3), line("We measured the thing carefully over the whole", 0.36, key=6),
            line("year and here is what we found in the end.", 0.38, key=7), line("See table 2.", 0.44, key=10),
            line("A first paragraph line that runs long enough here", 0.10, key=1),
            line("and a second line of it right below the first.", 0.12, key=2)]
    labels, receipt = infer_labels(grid, sized=False)
    by_text = {region.text: label for region, label in zip(grid, labels)}
    assert by_text["Methods"] == "heading", "set apart above and below, short, no terminal mark"
    assert by_text["See table 2."] == "paragraph", "a short line with a full stop is a sentence"
    assert by_text["and a second line of it right below the first."] == "paragraph"
    assert receipt.rules == ("heading",)


def test_a_grid_of_aligned_rows_is_a_table_and_a_page_over_the_line_bound_is_left_unread(monkeypatch):
    from scone_memory.ocr import labels as labels_module

    rows = []
    for r in range(4):
        top = 0.3 + r * 0.03
        for c, left in enumerate((0.1, 0.4, 0.7)):
            rows.append(OcrRegion(text=f"c{c}r{r}", box=(left, top, left + 0.15, top + 0.02), block=0, paragraph=0, line=r))
    prose = words("A paragraph above the table that runs on for a while here.", 0.1, key=20)
    labels, receipt = infer_labels([*prose, *rows])
    assert set(labels[len(prose):]) == {"table"} and set(labels[:len(prose)]) == {"paragraph"}
    assert "table" in receipt.rules
    monkeypatch.setattr(labels_module, "MAX_LINES", 3)
    labels, receipt = infer_labels([*prose, *rows])
    assert labels == [None] * (len(prose) + len(rows)) and receipt.notes == ("line_limit",), \
        "past the bound nothing is labelled, and the receipt says so"


def test_an_engine_s_boxes_label_by_containment_and_its_words_map_onto_the_vocabulary():
    page = [*words("Title of it", 0.05, key=1), *words("Body text here", 0.5, key=2), *words("stray", 0.95, key=3)]
    layout = LayoutResult(engine="pp-structure", width=100, height=100, dropped=2, regions=(
        LayoutRegion(label="title", box=(0.0, 0.0, 1.0, 0.1), score=0.9, order=0),
        LayoutRegion(label="paragraph", box=(0.0, 0.4, 1.0, 0.6), score=0.8, order=1)))
    labels, receipt = label_from_engine(page, layout)
    assert labels[:3] == ["title"] * 3 and labels[3:6] == ["paragraph"] * 3 and labels[6] is None
    assert receipt.source == "engine" and receipt.engine == "pp-structure" and receipt.unlabeled == 1 and receipt.dropped == 2
    assert receipt.counts == {"title": 3, "paragraph": 3}
    assert as_label("doc_title") == "title" and as_label("Figure caption") == "caption" and as_label("Text") == "paragraph"
    assert as_label("vision_footnote") == "footnote" and as_label("equation") == "formula" and as_label("seal") == "figure"
    assert as_label("something-else") is None and all(as_label(name) == name for name in REGION_LABELS)
    assert running_key("Chapter 3 · 41") == running_key("Chapter  3 · 42")
    with pytest.raises(ValueError):
        LayoutRegion(label="poster", box=(0, 0, 1, 1))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        LayoutLabels(source="inferred", strategy="labels-v1", rules=("page_number",) * 8)


def test_numbered_headings_are_not_a_list_and_a_year_at_the_foot_is_not_a_footnote():
    page = [
        *words("1. Introduction", 0.10, height=0.03, key=1),
        *words("The year was good and the robots were many in every warehouse.", 0.15, key=2),
        *words("• first item of the list", 0.20, key=3),
        *words("• second item of the list", 0.225, key=4),
        *words("3. Results", 0.30, height=0.03, key=5),
        *words("They worked and the report goes on.", 0.35, key=6),
        *words("2024 was a good year for the company.", 0.90, key=7),
        *words("10 people attended the meeting.", 0.92, key=8),
    ]
    labels, receipt = infer_labels(page)
    by_key = {}
    for region, label in zip(page, labels):
        by_key.setdefault(region.line, set()).add(label)
    assert by_key[1] == {"heading"} and by_key[5] == {"heading"}, "a numbered heading is told by its size, not its number"
    assert by_key[3] == {"list"} and by_key[4] == {"list"}
    assert by_key[7] == {"paragraph"} and by_key[8] == {"paragraph"}, "a year, or a count with prose after it, is prose"
    numbered = [*words("1. Buy milk", 0.5, key=1), *words("2. Buy eggs", 0.525, key=2), *words("3. Go home", 0.55, key=3)]
    labels, _ = infer_labels(numbered)
    assert set(labels) == {"list"}, "numbered lines beside each other are a list"
    grid = [line("2024 was a good year for the company.", 0.90, key=45), line("and it goes on right here.", 0.92, key=46),
            line("¹ A note on the year.", 0.96, key=48), line("A paragraph above, long enough to be prose.", 0.5, key=25)]
    labels, _ = infer_labels(grid, sized=False)
    by_text = {region.text: label for region, label in zip(grid, labels)}
    assert by_text["2024 was a good year for the company."] == "paragraph" and by_text["¹ A note on the year."] == "footnote"


def test_two_columns_on_one_grid_row_are_two_lines():
    from scone_memory.ocr.types import OrderedOcrRegion

    def run(text, row, column, left, right):
        return OrderedOcrRegion(text=text, box=(left, row / 40, right, (row + 1) / 40), line=row,
                                provider_index=row * 2 + column - 1, reading_column=column)
    rows = [run("• first thing", 10, 1, 0.05, 0.4), run("prose on the right that runs on and on", 10, 2, 0.55, 0.95),
            run("• second thing", 11, 1, 0.05, 0.4), run("more prose on the right of the page here", 11, 2, 0.55, 0.95),
            run("• third", 12, 1, 0.05, 0.4), run("and a last line of that prose column", 12, 2, 0.55, 0.95)]
    labels, _ = infer_labels(rows, sized=False)
    assert [label for label in labels] == ["list", "paragraph", "list", "paragraph", "list", "paragraph"], \
        "the left column's bullets are not the right column's"


def test_a_definition_list_s_dotted_numbers_are_list_items():
    page = [
        *words("Some prose that introduces the definitions of the plan below here.", 0.20, key=1),
        *words("2.1. “Administrator” means the committee appointed by the board", 0.30, key=2),
        *words("2.2. “Affiliate” means a parent or subsidiary of the company", 0.325, key=3),
        *words("2.3. “Board” means the board of directors of the company", 0.35, key=4),
        *words("(i) designed such controls to ensure that material information", 0.45, key=5),
        *words("(ii) evaluated the effectiveness of the controls and procedures", 0.475, key=6),
        *words("(iii) disclosed in this report any change in internal control", 0.50, key=7),
    ]
    labels, _ = infer_labels(page, first_page=False, running=set())
    by_key = {}
    for region, label in zip(page, labels):
        by_key.setdefault(region.line, set()).add(label)
    assert by_key == {1: {"paragraph"}, 2: {"list"}, 3: {"list"}, 4: {"list"}, 5: {"list"}, 6: {"list"}, 7: {"list"}}, by_key
