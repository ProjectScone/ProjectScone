"""Geometry-derived tables preserve every observed source region exactly once."""
import pytest

from scone_memory.core.errors import InvalidInput
from scone_memory.ocr.types import OcrRegion


def grid(*, top=.15, rows=4):
    result=[]
    for row in range(rows):
        y=top+row*.06
        for column,text in enumerate((f'Café {row}',str(10+row),f'Day {row}')):
            x=.1+column*.3
            result.append(OcrRegion(text=text,box=(x,y,x+.15,y+.025)))
    return result


def test_column_major_provider_order_becomes_linked_row_major_candidate():
    from scone_memory.ocr.tables import infer_tables
    observed=grid()
    observed=observed[::3]+observed[1::3]+observed[2::3]
    found=infer_tables(observed)
    assert found.strategy=='aligned-rows-v2'
    assert found.region_count==12 and found.unassigned==()
    assert len(found.tables)==1
    table=found.tables[0]
    assert (table.rows,table.columns)==(4,3)
    assert [cell.text for cell in table.cells[:3]]==['Café 0','10','Day 0']
    assert [cell.regions for cell in table.cells[:3]]==[(0,),(4,),(8,)]
    assert all(cell.box==observed[cell.regions[0]].box for cell in table.cells)


def test_a_title_across_the_table_is_its_first_row_and_a_footer_stays_unassigned():
    from scone_memory.ocr.tables import infer_tables
    observed=[OcrRegion(text='Report',box=(.1,.02,.8,.07)),*grid(),
              OcrRegion(text='Footnote',box=(.1,.8,.8,.85))]
    found=infer_tables(observed)
    assert len(found.tables)==1 and found.unassigned==(13,), "the title just above the grid spans its columns"
    [table]=found.tables
    title=[cell for cell in table.cells if cell.row==0]
    assert len(title)==1 and title[0].text=='Report' and title[0].column_span==table.columns
    assert 'unassigned_regions' in found.notes
    # A sentence above the grid is prose, not its title; a row at its foot
    # opening with a footnote's mark is its note, not its last row, even
    # when its cells fit the columns.
    prose=[OcrRegion(text='Sales rose in the year.',box=(.1,.02,.8,.07)),*grid(),
           OcrRegion(text='* Café',box=(.1,.4,.2,.425)),OcrRegion(text='means coffee here.',box=(.4,.4,.7,.425))]
    found=infer_tables(prose)
    [table]=found.tables
    assert table.rows==4 and found.unassigned==(0,13,14)
    # Paragraph lines below the grid, each as wide as the table and at
    # line spacing, are not rows of it, however many follow.
    below=[*grid(),*(OcrRegion(text=f'Prose line {n} runs on.',box=(.1,.4+n*.04,.85,.425+n*.04)) for n in range(3))]
    found=infer_tables(below)
    [table]=found.tables
    assert table.rows==4 and found.unassigned==(12,13,14)


def test_prose_between_or_beside_grids_is_not_their_rows():
    from scone_memory.ocr.tables import infer_tables
    def line(text,row,left=.1,right=.85):
        return OcrRegion(text=text,box=(left,.15+row*.06,right,.175+row*.06))
    # Three lines of prose between two grids of the same columns: the
    # first grid ends before them, and they are no one's rows.
    two=[*grid(top=.15,rows=3),*(line(f'Prose line {n} of the passage',3+n) for n in range(3)),*grid(top=.51,rows=3)]
    found=infer_tables(two)
    assert [t.rows for t in found.tables]==[3,3] and found.unassigned==(9,10,11)
    # A sentence closed by a citation is prose, not the title of the grid below.
    cited=[line('Guatemala had a billionaire for the first time in its history.[28]',0),*grid(top=.21,rows=3)]
    found=infer_tables(cited)
    assert found.tables[0].rows==3 and found.unassigned==(0,)
    # A wrapped word over a later column, above the grid, is not its title.
    wrapped=[OcrRegion(text='States',box=(.7,.09,.8,.115)),*grid(top=.15,rows=3)]
    found=infer_tables(wrapped)
    assert found.tables[0].rows==3 and found.unassigned==(0,)
    # A prose line at the foot with a cell across columns is not a row either.
    tail=[*grid(top=.15,rows=3),OcrRegion(text='Spanx founder',box=(.1,.33,.25,.355)),
          OcrRegion(text='Sara Blakely became the youngest self-made',box=(.4,.33,.85,.355))]
    found=infer_tables(tail)
    assert found.tables[0].rows==3 and found.unassigned==(9,10)
    # A subtotal across the number columns at the foot, and a label's
    # wrapped word below its row, are rows of the table.
    subtotal=[*grid(top=.15,rows=3),OcrRegion(text='Subtotal',box=(.1,.33,.22,.355)),OcrRegion(text='36',box=(.4,.33,.85,.355))]
    found=infer_tables(subtotal)
    assert found.tables[0].rows==4 and found.unassigned==() and [c.column_span for c in found.tables[0].cells if c.row==3]==[1,2]
    wrapped_label=[*grid(top=.15,rows=3),OcrRegion(text='development',box=(.1,.33,.22,.355))]
    found=infer_tables(wrapped_label)
    assert found.tables[0].rows==4 and found.unassigned==()


def test_two_columns_of_prose_are_not_a_table_but_a_notation_list_is():
    from scone_memory.ocr.tables import infer_tables
    def cell(text,row,left,right):
        return OcrRegion(text=text,box=(left,.1+row*.05,right,.13+row*.05))
    prose=[c for row in range(4) for c in (cell(f'Figure 2 shows the phase portrait {row}',row,.1,.45),
                                           cell(f'Our method finds both modes {row}',row,.55,.9))]
    assert infer_tables(prose).tables==() and len(infer_tables(prose).unassigned)==8
    notation=[c for row,(symbol,width) in enumerate((('RR',.13),('IPD',.14),('EVec, EVal',.2),('L : Rn -> R',.22)))
              for c in (cell(symbol,row,.1,width),cell(f'What the symbol {symbol} stands for here',row,.35,.8))]
    [table]=infer_tables(notation).tables
    assert (table.rows,table.columns)==(4,2), "a narrow, ragged first column is a table's"


def test_separated_grids_remain_separate_candidates():
    from scone_memory.ocr.tables import infer_tables
    found=infer_tables(grid(top=.05,rows=3)+grid(top=.65,rows=3))
    assert len(found.tables)==2
    assert found.unassigned==()


def test_words_in_a_cell_keep_individual_source_references():
    from scone_memory.ocr.tables import infer_tables
    observed=[]
    for region in grid():
        left,top,right,bottom=region.box
        words=region.text.split()
        for i,word in enumerate(words):
            x=left+i*.035
            observed.append(OcrRegion(text=word,box=(x,top,x+.03,bottom)))
    found=infer_tables(observed)
    assert len(found.tables)==1 and found.tables[0].columns==3
    assert found.tables[0].cells[0].text=='Café 0'
    assert found.tables[0].cells[0].regions==(0,1)


def test_short_or_crossing_rows_do_not_assert_a_grid():
    from scone_memory.ocr.tables import infer_tables
    for observed in (grid(rows=2), [OcrRegion(text='prose',box=(.1,.1+i*.07,.9,.14+i*.07)) for i in range(4)]):
        found=infer_tables(observed)
        assert found.tables==() and found.unassigned==tuple(range(len(observed)))
        assert 'no_aligned_grid' in found.notes
    observed=grid(rows=3)+[OcrRegion(text='spanning annotation',box=(.05,.15,.95,.30))]
    assert not infer_tables(observed).tables


def test_misaligned_cell_boundaries_do_not_join_rows():
    from scone_memory.ocr.tables import infer_tables
    observed=grid(rows=3)
    observed[4]=observed[4].model_copy(update={'box':(.24,.21,.62,.235)})
    found=infer_tables(observed)
    assert not found.tables


def test_limits_and_mutated_extension_observations_are_refused():
    from scone_memory.ocr.tables import infer_tables
    region=grid()[0]
    with pytest.raises(InvalidInput,match='region limit'):
        infer_tables([region]*5001)
    with pytest.raises(InvalidInput):
        infer_tables([region.model_copy(update={'box':(float('nan'),0.,.5,.5)})])


@pytest.mark.parametrize('separator_budget', [0, 19])
def test_joined_cell_separators_count_toward_text_budget(separator_budget):
    from scone_memory.ocr.tables import infer_tables
    regions = [OcrRegion(text='x' * 100000, box=(0.1, 0.1, 0.2, 0.15)) for _ in range(19)]
    regions.append(OcrRegion(text='x' * (99995 - separator_budget), box=(0.1, 0.1, 0.2, 0.15)))
    regions.append(OcrRegion(text='y', box=(0.6, 0.1, 0.7, 0.15)))
    for top in (0.19, 0.28):
        for left in (0.1, 0.6):
            regions.append(OcrRegion(text='y', box=(left, top, left + 0.1, top + 0.05)))
    assert sum(len(region.text.encode()) for region in regions) == 2_000_000 - separator_budget
    if not separator_budget:
        with pytest.raises(InvalidInput, match='text limit'):
            infer_tables(regions)
        return
    result = infer_tables(regions)
    assert len(result.tables) == 1
    assert sum(len(cell.text.encode()) for cell in result.tables[0].cells) == 2_000_000


def test_a_row_of_fewer_cells_spans_the_grid_s_columns_and_a_narrow_one_is_not_widened():
    from scone_memory.ocr.tables import infer_tables
    from scone_memory.ocr.types import OcrRegion

    def cell(text, row, left, right):
        top = 0.1 + row * 0.05
        return OcrRegion(text=text, box=(left, top, right, top + 0.03))
    regions = [cell("Quarterly figures", 0, 0.1, 0.75),  # a title over three columns
               cell("Item", 1, 0.1, 0.25), cell("Q1", 1, 0.4, 0.5), cell("Q2", 1, 0.65, 0.75),
               cell("Widgets", 2, 0.1, 0.28), cell("10", 2, 0.4, 0.48), cell("12", 2, 0.65, 0.73),
               cell("Subtotal", 3, 0.1, 0.24), cell("22", 3, 0.4, 0.75),  # a subtotal across the two number columns
               cell("Gadgets", 4, 0.1, 0.28), cell("7", 4, 0.4, 0.45), cell("9", 4, 0.65, 0.7),
               cell("Total", 5, 0.1, 0.22), cell("17", 5, 0.4, 0.45), cell("21", 5, 0.65, 0.7)]
    layout = infer_tables(regions)
    assert layout.strategy == 'aligned-rows-v2' and len(layout.tables) == 1
    [table] = layout.tables
    assert table.rows == 6 and table.columns == 3
    spans = {(c.row, c.column): c.column_span for c in table.cells}
    assert spans[(0, 0)] == 3 and spans[(3, 1)] == 2 and spans[(3, 0)] == 1 and spans[(2, 1)] == 1
    assert {c.text for c in table.cells if c.row == 0} == {"Quarterly figures"}
    assert 'column_span' not in [c for c in table.cells if c.row == 2][0].model_dump(), "a cell of one column serializes as before"
    assert not layout.unassigned, "every region sits in a cell"
    narrow = [cell("Item", 1, 0.1, 0.25), cell("Q1", 1, 0.4, 0.5), cell("Q2", 1, 0.65, 0.75),
              cell("Widgets", 2, 0.1, 0.28), cell("10", 2, 0.4, 0.48), cell("12", 2, 0.65, 0.73),
              cell("Gadgets", 3, 0.1, 0.28), cell("7", 3, 0.4, 0.45), cell("9", 3, 0.65, 0.7),
              cell("note", 4, 0.3, 0.36)]  # a word in a gutter, in no column band: not a row of the table
    layout = infer_tables(narrow)
    [table] = layout.tables
    assert table.rows == 3 and layout.unassigned, "a cell over no column is not read as spanning"
    grazing = [cell("Quarterly figures for the year", 0, 0.1, 0.42),  # a title over two thirds of the narrow Q1 band
               cell("Item", 1, 0.1, 0.25), cell("Q1", 1, 0.4, 0.43), cell("Q2", 1, 0.65, 0.75),
               cell("Widgets", 2, 0.1, 0.28), cell("10", 2, 0.4, 0.43), cell("12", 2, 0.65, 0.73),
               cell("Gadgets", 3, 0.1, 0.28), cell("7", 3, 0.4, 0.43), cell("9", 3, 0.65, 0.7)]
    [table] = infer_tables(grazing).tables
    assert {(c.row, c.column_span) for c in table.cells if c.row == 0} == {(0, 1)}, "a wide cell grazing a narrow band does not claim it"
    narrow_total = [cell("Item", 1, 0.1, 0.25), cell("Net worth (USD)", 1, 0.36, 0.58), cell("Q2", 1, 0.65, 0.75),
                    cell("Widgets", 2, 0.1, 0.28), cell("10", 2, 0.4, 0.44), cell("12", 2, 0.65, 0.73),
                    cell("Gadgets", 3, 0.1, 0.28), cell("7", 3, 0.4, 0.43), cell("9", 3, 0.65, 0.7),
                    cell("Total", 4, 0.1, 0.22), cell("57", 4, 0.4, 0.44)]  # a number a fifth as wide as its column's band
    layout = infer_tables(narrow_total)
    [table] = layout.tables
    assert table.rows == 4 and not layout.unassigned, "a narrow number inside a wide band sits in it"
    assert {(c.row, c.column, c.column_span) for c in table.cells if c.row == 3} == {(3, 0, 1), (3, 1, 1)}
