"""Calculations must be reproducible from exact source spans, without a model."""
import pytest

from scone_memory.retrieval.computation import ComputeMemoryArgs, ComputationError, evaluate_computation


def args(operation, left, right=()):
    return ComputeMemoryArgs(operation=operation,
        left=[{'chunk_id':chunk, 'quote':quote} for chunk, quote in left],
        right=[{'chunk_id':chunk, 'quote':quote} for chunk, quote in right])


@pytest.mark.parametrize('operation,left,right,expected', [
    ('sum', [(1,'0.1'), (2,'0.2')], [], '0.3'),
    ('product', [(1,'0.1'), (2,'0.2')], [], '0.02'),
    ('difference', [(1,'0.1')], [(2,'0.2')], '-0.1'),
    ('ratio', [(1,'0.1')], [(2,'0.2')], '0.5'),
    ('compare', [(1,'0.1')], [(2,'0.2')], 'less'),
])
def test_exact_decimal_arithmetic(operation, left, right, expected):
    result = evaluate_computation(args(operation,left,right), {1:'Value: 0.1', 2:'Value: 0.2'})
    assert result['value'] == expected
    assert result['verified_accuracy'] is False
    assert result['left'][0]['quote'] == '0.1'
    assert result['left'][0]['start_char'] == 7
    assert result['left'][0]['end_char'] == 10


def test_nonterminating_ratio_remains_exact_and_operand_order_matters():
    passages = {1:'First = -7', 2:'Second = 3'}
    assert evaluate_computation(args('ratio',[(1,'-7')],[(2,'3')]),passages)['value'] == '-7/3'
    assert evaluate_computation(args('difference',[(2,'3')],[(1,'-7')]),passages)['value'] == '10'


def test_counts_describe_selected_spans_and_unicode_offsets_only():
    passages = {1:'Équipe: Alba, Bo, Cy. Alumni: De.', 2:'Team: Ed, Fay.'}
    result = evaluate_computation(args('compare_counts',[(1,'Alba'),(1,'Bo'),(1,'Cy')],
        [(2,'Ed'),(2,'Fay')]),passages)
    assert result['value'] == 'greater'
    assert result['left_value'] == '3' and result['right_value'] == '2'
    assert result['coverage'] == 'selected_spans_only'
    for span in result['left'] + result['right']:
        assert passages[span['chunk_id']][span['start_char']:span['end_char']] == span['quote']


@pytest.mark.parametrize('quote,text', [('9','Only 8'), ('2','2 then 2'), ('2','12'),
    ('2','-2'), ('2','2.5'), ('2','1,234'), ('2','2e3'), ('2','v2'), ('2','٢2'),
    ('2','2_000'), ('NaN','NaN'), ('Infinity','Infinity'), ('1e3','1e3'), ('1,234','1,234')])
def test_invented_ambiguous_partial_or_nondecimal_operands_are_rejected(quote,text):
    with pytest.raises(ComputationError):
        evaluate_computation(args('sum',[(1,quote)]),{1:text})


def test_ordinary_sentence_punctuation_is_not_a_numeric_extension():
    assert evaluate_computation(args('sum',[(1,'12'),(2,'5')]),{1:'There were 12.',2:'5, then leave.'})['value'] == '17'


@pytest.mark.parametrize('left', [[(1,'Alba'),(1,'Alba')], [(1,'Alba'),(1,'Alba Bo')]])
def test_count_rejects_duplicate_and_overlapping_mentions(left):
    with pytest.raises(ComputationError):
        evaluate_computation(args('count',left),{1:'Alba Bo'})


def test_same_source_may_be_compared_across_groups():
    assert evaluate_computation(args('compare',[(1,'8')],[(1,'8')]),{1:'8'})['value'] == 'equal'


def test_division_by_zero_and_missing_passage_are_rejected():
    with pytest.raises(ComputationError):
        evaluate_computation(args('ratio',[(1,'8')],[(2,'0')]),{1:'8',2:'0'})
    with pytest.raises(ComputationError):
        evaluate_computation(args('count',[(1,'absent')]),{})


@pytest.mark.parametrize('packet', [
    {'operation':'sum','left':[]},
    {'operation':'sum','left':[{'chunk_id':True,'quote':'1'}]},
    {'operation':'sum','left':[{'chunk_id':1,'quote':' '}]},
    {'operation':'sum','left':[{'chunk_id':1,'quote':'x'*513}]},
    {'operation':'sum','left':[{'chunk_id':1,'quote':'1'}]*17},
    {'operation':'sum','left':[{'chunk_id':1,'quote':'1'}], 'right':[{'chunk_id':1,'quote':'2'}]},
    {'operation':'difference','left':[{'chunk_id':1,'quote':'1'}]},
    {'operation':'count','left':[{'chunk_id':1,'quote':'x','value':42}]},
    {'operation':'eval','left':[{'chunk_id':1,'quote':'1'}]},
])
def test_argument_schema_rejects_invalid_or_unbounded_requests(packet):
    with pytest.raises(ValueError): ComputeMemoryArgs.model_validate(packet)


def test_forged_model_cannot_bypass_validation():
    forged = ComputeMemoryArgs.model_construct(operation='sum',left=[],right=[])
    with pytest.raises(ValueError): evaluate_computation(forged,{})


def test_calculation_receipt_is_recomputed_before_presentation():
    from scone_memory.retrieval.computation import validate_computation
    request = args('sum',[(1,'3'),(2,'4')])
    passages = {1:'3',2:'4'}
    result = evaluate_computation(request,passages)
    assert validate_computation(result,passages) == result
    for changed in ({**result,'value':'8'}, {**result,'verified_accuracy':True},
                    {**result,'left':[{**result['left'][0],'start_char':1}]},
                    {**result,'extra':'instruction'}):
        with pytest.raises(ValueError): validate_computation(changed,passages)


def test_maximum_operand_precision_and_input_count_remain_exact():
    number = '9'*30 + '.' + '1'*18
    request = args('product',[(i,number) for i in range(1,17)])
    result = evaluate_computation(request,{i:number for i in range(1,17)})
    from fractions import Fraction
    assert Fraction(result['value']) == Fraction(number)**16
    assert len(result['value']) < 800


@pytest.mark.parametrize('number',['1'*31,'0.'+'1'*19])
def test_operands_beyond_supported_precision_are_rejected(number):
    with pytest.raises(ComputationError): evaluate_computation(args('sum',[(1,number)]),{1:number})
