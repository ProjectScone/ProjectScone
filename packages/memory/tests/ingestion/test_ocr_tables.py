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
    assert found.strategy=='aligned-rows-v1'
    assert found.region_count==12 and found.unassigned==()
    assert len(found.tables)==1
    table=found.tables[0]
    assert (table.rows,table.columns)==(4,3)
    assert [cell.text for cell in table.cells[:3]]==['Café 0','10','Day 0']
    assert [cell.regions for cell in table.cells[:3]]==[(0,),(4,),(8,)]
    assert all(cell.box==observed[cell.regions[0]].box for cell in table.cells)


def test_title_and_footer_remain_explicitly_unassigned():
    from scone_memory.ocr.tables import infer_tables
    observed=[OcrRegion(text='Report',box=(.1,.02,.8,.07)),*grid(),
              OcrRegion(text='Footnote',box=(.1,.8,.8,.85))]
    found=infer_tables(observed)
    assert len(found.tables)==1 and found.unassigned==(0,13)
    assert 'unassigned_regions' in found.notes


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
