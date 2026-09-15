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


def statement(*, top=.2, rows=4, signs=(0,), wide=None):
    """A statement's grid: wide labels, two value columns of right-aligned
    numbers with a currency sign at each column's left edge on ``signs``
    rows, and on the ``wide`` row a value reaching close to the next sign."""
    made=[]
    for row in range(rows):
        y=top+row*.04
        made.append(OcrRegion(text=f'Cost of revenue {row}',box=(.02,y,.38,y+.025)))
        for column,(edge,right) in enumerate(((.5,.68),(.72,.9))):
            if row in signs:
                made.append(OcrRegion(text='$',box=(edge,y,edge+.01,y+.025)))
            left=.6 if (row==wide and column==0) else right-.06
            made.append(OcrRegion(text=f'{1000*(row+1)+column:,}' if not (row==wide and column==0) else '(12,345)',
                                  box=(left,y,right+.02 if (row==wide and column==0) else right,y+.025)))
    return made


def test_a_currency_sign_apart_from_its_number_is_that_number_s_cell():
    from scone_memory.ocr.tables import infer_tables
    # Signs on the first and last rows, and a wide value on the third whose
    # right edge sits two hundredths from the next column's sign: the sign
    # is read with the number to its right, and the gutter before the cell
    # runs to the number, so the wide value does not close the grid.
    regions=statement(signs=(0,3),wide=2)
    [table]=infer_tables(regions).tables
    assert table.rows==4 and table.columns==3 and not infer_tables(regions).unassigned
    signed=[c for c in table.cells if c.text.startswith('$')]
    assert [(c.row,c.column,c.text) for c in signed]==[(0,1,'$ 1,000'),(0,2,'$ 1,001'),(3,1,'$ 4,000'),(3,2,'$ 4,001')]
    assert all(len(c.regions)==2 and c.box[0]==regions[c.regions[0]].box[0] for c in signed), "the cell runs from the sign to the number"
    # A sign before a word is a cell of its own.
    worded=[*statement(rows=3,signs=()),OcrRegion(text='Prices in',box=(.02,.32,.2,.345)),
            OcrRegion(text='$',box=(.5,.32,.51,.345)),OcrRegion(text='USD',box=(.72,.32,.78,.345))]
    [table]=infer_tables(worded).tables
    assert table.rows==4 and [c.text for c in table.cells if c.row==3]==['Prices in','$','USD']


def test_a_statement_s_caption_and_years_above_the_grid_are_its_header_rows():
    from scone_memory.ocr.tables import MAX_HEADER_ROWS, infer_tables
    def line(text,left,right,row):
        return OcrRegion(text=text,box=(left,.08+row*.04,right,.105+row*.04))
    # A units line centred on the page, grazing the label column's band, a
    # caption over the two value columns short of half of the second (a
    # comma closing it, as the years follow), the years over a blank
    # label column, then the grid with a section's name between its
    # first rows.
    regions=[line('(In millions)',.36,.48,0),line('Three Months Ended March 31,',.56,.8,1),
             line('2021',.62,.66,2),line('2022',.84,.88,2),*statement(top=.2,rows=2,signs=(0,)),
             OcrRegion(text='Costs',box=(.02,.28,.05,.305)),*statement(top=.32,rows=2,signs=())]
    found=infer_tables(regions)
    [table]=found.tables
    rows={}
    for cell in table.cells:
        rows.setdefault(cell.row,[]).append((cell.column,cell.column_span,cell.text))
    assert table.rows==7 and found.unassigned==(0,), "the units line is not a row"
    assert rows[0]==[(1,2,'Three Months Ended March 31,')] and rows[1]==[(1,1,'2021'),(2,1,'2022')]
    assert rows[4]==[(0,1,'Costs')], "a short word inside a wide column sits in it"
    assert rows[2][1:]==[(1,1,'$ 1,000'),(2,1,'$ 1,001')] and len(rows[6])==3
    # A sentence's comma is a sentence's: a line across the table from its
    # first column closed by a comma is not its title.
    prose=[line('The figures below are unaudited and in millions,',.02,.7,2),*statement(top=.2,rows=3)]
    found=infer_tables(prose)
    assert found.tables[0].rows==3 and found.unassigned==(0,)
    # Rows above the grid are read up to a bound: of five title lines in a
    # chain, the nearest four are its rows and the fifth is not.
    titled=[*(line(f'Title line {n}',.02,.3,n) for n in range(MAX_HEADER_ROWS+1)),*statement(top=.08+(MAX_HEADER_ROWS+1)*.04,rows=3)]
    found=infer_tables(titled)
    assert MAX_HEADER_ROWS==4 and found.tables[0].rows==3+MAX_HEADER_ROWS and found.unassigned==(0,)


def test_a_bulleted_or_enumerated_list_is_not_a_table_but_an_enumerated_column_of_labels_is():
    from scone_memory.ocr.tables import infer_tables
    def item(mark,text,row,right=.9):
        y=.1+row*.05
        return [OcrRegion(text=mark,box=(.05,y,.07,y+.03)),OcrRegion(text=text,box=(.15,y,right,y+.03))]
    # Bullets before short items: the marks say list, whatever the items'
    # width; a dash before the same items stands for none, and the grid
    # is a table.
    short=[c for row,text in enumerate(('Driver Classification','State Unemployment Taxes','Google v. Levandowski','Other matters'))
           for c in item('•',text,row,right=.25)]
    found=infer_tables(short)
    assert found.tables==() and len(found.unassigned)==8
    dashed=[c for row,text in enumerate(('Driver Classification','State Unemployment Taxes','Google v. Levandowski','Other matters'))
            for c in item('-',text,row,right=.25)]
    assert infer_tables(dashed).tables[0].rows==4
    # Enumerators before lines of prose, with a wrapped line between: a numbered list.
    numbered=[*item('1.','I have reviewed this Quarterly Report on Form 10-Q of the registrant;',0),
              *item('2.','Based on my knowledge, this report does not contain any untrue',1),
              OcrRegion(text='statements made, in light of the circumstances, not misleading;',box=(.15,.2,.85,.23)),
              *item('3.','Based on my knowledge, the financial statements fairly present',3),
              *item('(a)','Designed such disclosure controls and procedures to ensure that',4)]
    found=infer_tables(numbered)
    assert found.tables==() and len(found.unassigned)==9
    # Enumerators before short labels are a table's first column, as a
    # column of plain numbers is.
    labelled=[c for row,text in enumerate(('Mobility','Delivery','Freight','Total')) for c in item(f'{row+1}.',text,row,right=.25)]
    [table]=infer_tables(labelled).tables
    assert (table.rows,table.columns)==(4,2)
    plain=[c for row,text in enumerate(('Mobility','Delivery','Freight','Total')) for c in item(str(row+1),text,row,right=.25)]
    assert infer_tables(plain).tables[0].rows==4


def test_the_layout_says_when_the_rows_above_a_grid_were_read_up_to_their_bound():
    from scone_memory.ocr.tables import MAX_HEADER_ROWS, infer_tables
    def line(text,row):
        return OcrRegion(text=text,box=(.02,.08+row*.04,.3,.105+row*.04))
    grid=statement(top=.08+(MAX_HEADER_ROWS+1)*.04,rows=3)
    over=infer_tables([*(line(f'Title line {n}',n) for n in range(MAX_HEADER_ROWS+1)),*grid])
    assert over.tables[0].rows==3+MAX_HEADER_ROWS and 'header_rows_limit' in over.notes, "a title past the bound was dropped by the cap"
    within=infer_tables([*(line(f'Title line {n}',n+1) for n in range(MAX_HEADER_ROWS)),*grid])
    assert within.tables[0].rows==3+MAX_HEADER_ROWS and 'header_rows_limit' not in within.notes
    broken=infer_tables([*(line(f'Title line {n}',n) for n in range(MAX_HEADER_ROWS-1)),line('This sentence ends the chain.',MAX_HEADER_ROWS-1),line('Title line last',MAX_HEADER_ROWS),*grid])
    assert broken.tables[0].rows==4 and 'header_rows_limit' not in broken.notes, "the chain broke before the cap mattered"
    # A row of too many cells to be a grid's clears the rows held above
    # it, and the cap that bit before it is forgotten with them.
    scattered=[OcrRegion(text=str(n),box=(.02+n*.07,.28,.03+n*.07,.305)) for n in range(13)]
    cleared=infer_tables([*(line(f'Title line {n}',n) for n in range(MAX_HEADER_ROWS+1)),*scattered,line('Title line 6',6),line('Title line 7',7),*statement(top=.40,rows=3)])
    assert cleared.tables[0].rows==5 and 'header_rows_limit' not in cleared.notes


def test_a_colon_closed_heading_between_the_years_and_the_grid_is_a_row_and_a_sentence_s_tail_is_not():
    from scone_memory.ocr.tables import infer_tables
    def line(text,left,right,row):
        return OcrRegion(text=text,box=(left,.08+row*.04,right,.105+row*.04))
    years=[line('2021',.62,.66,0),line('2022',.84,.88,0)]
    # The years, a section's heading closed by a colon, then the grid:
    # the heading is a row and the years still head the grid above it.
    headed=[*years,line('Basic net loss per share:',.02,.2,1),*statement(top=.16,rows=3)]
    found=infer_tables(headed)
    rows={}
    for cell in found.tables[0].cells:
        rows.setdefault(cell.row,[]).append((cell.column,cell.column_span,cell.text))
    assert found.tables[0].rows==5 and rows[0]==[(1,1,'2021'),(2,1,'2022')] and rows[1]==[(0,1,'Basic net loss per share:')]
    # A sentence's tail on a line of its own, or a long colon-closed line,
    # is prose: the chain ends there and the years are lost with it.
    for tail in ('were as follows:','The components of intangible assets, net as of December 31 were:'):
        found=infer_tables([*years,line(tail,.02,.2 if tail.startswith('were') else .7,1),*statement(top=.16,rows=3)])
        assert found.tables[0].rows==3 and found.unassigned==(0,1,2), tail


def test_a_header_cell_beside_a_narrow_column_claims_it_by_its_place_in_the_row():
    from scone_memory.ocr.tables import infer_tables
    def cell(text,left,right,row):
        return OcrRegion(text=text,box=(left,.1+row*.04,right,.125+row*.04))
    def grid(top_row):
        made=[]
        for n in range(3):
            row=top_row+n
            made+=[cell(f'Asset {n}',.02,.2,row),cell(f'{1000+n:,}',.4,.45,row),cell(f'({100+n})',.6,.65,row),cell(str(n),.99,1.,row)]
        return made
    # The last header cell sits left of the digits column, over no band:
    # it takes the one band left between its neighbour's and the row's end.
    header=[cell('Gross Carrying Value',.36,.47,0),cell('Accumulated Amortization',.55,.67,0),cell('Useful Life - Years',.87,.97,0)]
    [table]=infer_tables([*header,*grid(1)]).tables
    assert table.rows==4 and [(c.column,c.column_span,c.text) for c in table.cells if c.row==0]==[
        (1,1,'Gross Carrying Value'),(2,1,'Accumulated Amortization'),(3,1,'Useful Life - Years')]
    # A lone cell over no band is not placed by its place, nor a cell with
    # two bands to choose from.
    lone=infer_tables([cell('Useful Life - Years',.87,.97,0),*grid(1)])
    assert lone.tables[0].rows==3 and lone.unassigned==(0,)
    two=infer_tables([cell('Gross Carrying Value',.36,.47,0),cell('Useful Life - Years',.87,.97,0),*grid(1)])
    assert two.tables[0].rows==3 and two.unassigned==(0,1)
