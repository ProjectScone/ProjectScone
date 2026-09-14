"""Chinese and Japanese text cut where it divides itself.

The chunker preferred a paragraph break, then ". " and its kin, then any
whitespace, then a hard cut. Text in scripts written without spaces has
none of those inside a paragraph: its sentences end at a full-width stop
with nothing after it, so every chunk was cut at exactly the byte target,
in the middle of a word. And numbered clauses written the way Chinese and
Japanese documents number them -- 第三条, 一、, （二） -- were not clauses to
the structure chunker, which looks for "Article 3" and "(b)". Both are
boundaries now; ASCII text chunks exactly as it did.
"""

from __future__ import annotations

from scone_memory.ingestion.chunker import chunk_spans
from scone_memory.ingestion.structure_chunks import structured_spans, units

SENTENCES = ["港口起重机的年度检查安排在五月的第三周进行。", "检查发现吊臂上有锈迹，回转支承缺少润滑脂。",
             "维修队在同一周内完成了除锈和润滑两项工作。", "起重机于六月一日恢复使用，比原计划晚了一天。",
             "交接单上没有人签字，这在七月引起了争议。"]


def test_chinese_prose_is_cut_after_a_full_width_stop_never_inside_a_sentence():
    text = "".join(SENTENCES * 12)
    spans = chunk_spans(text, target=150)
    assert len(spans) > 3
    for span in spans[:-1]:
        assert text[span.end - 1] == "。", text[span.end - 5:span.end + 5]


def test_a_closing_quote_after_the_stop_stays_with_its_sentence():
    # No stop before the quote, so the last stop inside the window is the quoted one.
    opening = "维修记录已经归档，等待审核人员签字确认，" * 7
    quoted = opening + "他说：「吊臂的锈迹已经处理完毕。」" + "维修记录已经归档，等待审核" * 20
    spans = chunk_spans(quoted, target=200)
    assert 120 < len(opening) + 17 <= 200
    assert quoted[spans[0].start:spans[0].end].endswith("。」")


def test_the_later_of_a_space_and_a_clause_mark_is_the_cut():
    from scone_memory.ingestion.chunker import MIN_CHUNK

    text = "检查完成，next step is repair of the jib " * 20
    spans = chunk_spans(text, target=150)
    ends = []
    for span in spans[:-1]:
        window = text[span.start + MIN_CHUNK:span.start + 150]
        expected = span.start + MIN_CHUNK + max(window.rfind(" "), window.rfind("，")) + 1
        assert span.end == expected, (span, expected)
        ends.append(text[span.end - 1])
    assert set(ends) == {" ", "，"}, "the fixture must reach both cases"


def test_without_a_stop_a_clause_mark_beats_a_cut_inside_a_word():
    text = "维修记录已经归档，" * 40
    spans = chunk_spans(text, target=150)
    for span in spans[:-1]:
        assert text[span.end - 1] == "，"


def test_japanese_stops_and_marks_count_too():
    text = "港のクレーンの点検は五月に行われた。ジブに錆が見つかった！修理はいつ終わるのか？" * 20
    # 190 is not a multiple of the 40-character pattern, so a stop at the target is not a coincidence.
    spans = chunk_spans(text, target=190)
    assert len(spans) > 2 and all(text[span.end - 1] in "。！？" for span in spans[:-1])
    exclaimed = "ジブに錆が見つかった！修理はいつ終わるのか？" * 30
    spans = chunk_spans(exclaimed, target=190)
    assert len(spans) > 2 and all(exclaimed[span.end - 1] in "！？" for span in spans[:-1])


def test_text_without_full_width_punctuation_chunks_exactly_as_it_did():
    """The spans below were taken from the chunker before this change."""
    text = ("The survey of the harbour crane was booked for the third of May. " * 30
            + "It found rust on the jib, and a slew ring that needed grease, " * 20)
    assert [(span.start, span.end) for span in chunk_spans(text, target=200)] == [
        (0, 195), (195, 390), (390, 585), (585, 780), (780, 975), (975, 1170), (1170, 1365), (1365, 1560),
        (1560, 1755), (1755, 1950), (1950, 2150), (2150, 2348), (2348, 2545), (2545, 2741), (2741, 2934), (2934, 3190)]
    mixed = "Harbour crane 港口起重机 inspection, 检查 and repair " * 40
    assert [(span.start, span.end) for span in chunk_spans(mixed, target=150)] == [
        (0, 146), (146, 296), (296, 446), (446, 591), (591, 736), (736, 882), (882, 1032), (1032, 1182),
        (1182, 1327), (1327, 1472), (1472, 1618), (1618, 1840)]


def test_chinese_and_japanese_clause_numbering_is_structure():
    doc = ("总则说明\n第一章 总则\n本章规定适用范围。\n第二条本法所称起重机，是指港口使用的设备。\n"
           "一、检查范围\n（一）吊臂\n（二）回转支承\n1、每年检查一次\n第一次检查在五月。\n他说一、二、三。\n"
           "第3条 検査は毎年行う。\n一九八四年是一个年份。\n")
    found = [(unit.kind, unit.label) for unit in units(doc) if unit.kind == "clause"]
    assert [label for _, label in found] == ["第一章 总则", "第二条本法所称起重机，是指港口使用的设备。", "一、检查范围",
                                             "（一）吊臂", "（二）回转支承", "1、每年检查一次", "第3条 検査は毎年行う。"]


def test_a_chinese_law_is_chunked_at_its_articles():
    articles = [f"第{n}条 " + "起重机的使用单位应当建立安全管理制度，并定期检查设备状况。" * 3 + "\n" for n in "一二三四五六"]
    doc = "".join(articles)
    made = structured_spans(doc, target=120)
    starts = [doc[span.start:span.start + 1] for span in made.spans]
    assert starts == ["第"] * len(made.spans) and made.at_boundary == len(articles)
