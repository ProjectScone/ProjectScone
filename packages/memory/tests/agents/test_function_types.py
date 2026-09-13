"""Bounded Python annotation contracts without executing annotation expressions."""
from enum import Enum
from typing import Annotated, Any, Dict, Literal, Optional, Tuple
import pytest
from scone_memory.agents.function_types import parameter_type, resolve_annotation

@pytest.mark.parametrize('annotation,value,kind', [(str,'é','string'),(int,42,'integer'),(bool,True,'boolean'),(float,1.5,'number'),(type(None),None,'null')])
def test_scalars(annotation,value,kind):
    contract=parameter_type(annotation,{})
    assert contract.schema=={'type':kind}
    for convert in (contract.encode,contract.decode):
        assert convert(value)==value and type(convert(value)) is type(value)

@pytest.mark.parametrize('annotation,value',[(int,True),(bool,1),(str,b'x'),(int,1.5),(float,True),(float,float('inf')),(float,float('nan')),(str,'\ud800')])
def test_scalar_mismatches(annotation,value):
    for convert in (parameter_type(annotation,{}).encode,parameter_type(annotation,{}).decode):
        with pytest.raises(ValueError):convert(value)

def test_nested_tuple_and_mutable_defaults():
    contract=parameter_type(dict[str,list[tuple[str,int]]],{})
    source={'rows':[('a',2)]};encoded=contract.encode(source)
    assert encoded=={'rows':[['a',2]]}
    assert contract.decode(encoded)==source and type(contract.decode(encoded)['rows'][0]) is tuple
    source['rows'][0]=('changed',4)
    assert contract.decode(encoded)=={'rows':[('a',2)]}
    for bad in ({1:[('a',2)]},{'rows':[['a',2]]},{'rows':[('a',True)]}):
        with pytest.raises(ValueError):contract.encode(bad)
    for bad in ({'rows':[('a',2)]},{'rows':[['a']]},{'rows':[['a',2,3]]}):
        with pytest.raises(ValueError):contract.decode(bad)
    homogeneous=parameter_type(tuple[int,...],{})
    assert homogeneous.encode((1,2))==[1,2] and homogeneous.decode([1,2])==(1,2)

def test_union_prefers_exact_recursive_types_and_collapses_equivalent_results():
    contract=parameter_type(Optional[list[int]],{})
    assert contract.decode(None) is None and contract.decode([2])==[2]
    for annotation in (int|float,Literal[1]|int):
        assert type(parameter_type(annotation,{}).decode(1)) is int
    assert type(parameter_type(int|float,{}).decode(1.0)) is float
    nested=parameter_type(list[float]|list[int],{})
    assert type(nested.decode([1])[0]) is int
    assert type(parameter_type(list[int]|tuple[int,...],{}).decode([1])) is list
    with pytest.raises(ValueError):parameter_type(int|str,{}).decode(True)
    literal=parameter_type(Literal[1,True,'yes',None],{})
    for value in (1,True,'yes',None):assert type(literal.decode(value)) is type(value)
    with pytest.raises(ValueError):literal.decode(1.0)

def test_annotated_schema_is_detached():
    contract=parameter_type(Annotated[int,'A count of local records.'],{})
    assert contract.schema=={'type':'integer','description':'A count of local records.'}
    contract.schema['type']='string'
    assert contract.schema['type']=='integer'
    for description in ({'description':'Unsupported'},'\ud800'):
        with pytest.raises(ValueError):parameter_type(Annotated[int,description],{})

def test_enum_values_are_snapshotted():
    class Color(Enum):RED='red';BLUE='blue'
    contract=parameter_type(Color,{})
    assert contract.encode(Color.RED)=='red' and contract.decode('blue') is Color.BLUE
    Color.RED._value_='changed'
    assert contract.encode(Color.RED)=='red' and contract.decode('red') is Color.RED
    assert contract.schema['enum']==['red','blue']
    with pytest.raises(ValueError):contract.decode('changed')
    with pytest.raises(ValueError):contract.encode('red')
    assert type(parameter_type(Color|str,{}).decode('blue')) is str

@pytest.mark.parametrize('annotation',[Any,object,list,dict,tuple,set[int],dict[int,str]])
def test_unsupported_annotations(annotation):
    with pytest.raises(ValueError):parameter_type(annotation,{})

def test_safe_forward_forms():
    class Context:pass
    class Color(Enum):RED='red'
    import typing
    namespace={'Context':Context,'Color':Color,'typing':typing}
    assert resolve_annotation('Context',namespace) is Context
    for source,annotation in [('list[int]',list[int]),('typing.Dict[str, Optional[int]]',Dict[str,Optional[int]]),('Tuple[str, int]',Tuple[str,int]),('tuple[int, ...]',tuple[int,...]),('int | None',int|None),('Annotated[str,"Label"]',Annotated[str,'Label']),('Literal["red",1,True,None]',Literal['red',1,True,None]),('Color',Color),('"list[int]"',list[int])]:
        assert parameter_type(source,namespace).schema==parameter_type(annotation,namespace).schema

@pytest.mark.parametrize('source',["__import__('os').system('echo bad')",'factory()','obj.attribute','[int for x in items]','list[factory()]','int.__class__','typing.__dict__','Annotated[int,str(3)]','Literal[1+2]','list[int,str]','Unknown','Trap[int]'])
def test_unsafe_forward_forms_never_execute(source):
    invoked=[]
    class Trap:
        def __getattribute__(self,key):invoked.append(key);raise AssertionError('attribute executed')
        def __class_getitem__(cls,key):invoked.append(key);raise AssertionError('generic executed')
    with pytest.raises(ValueError):parameter_type(source,{'obj':Trap(),'Trap':Trap,'factory':lambda:invoked.append('call')})
    assert not invoked

def test_structure_and_byte_bounds():
    annotation=int
    for _ in range(18):annotation=list[annotation]
    for value in (annotation,'list['*18+'int'+']'*18,Tuple[tuple(int for _ in range(257))],Annotated[str,'x'*32768]):
        with pytest.raises(ValueError):parameter_type(value,{})
    with pytest.raises(ValueError):resolve_annotation('Alias',{'Alias':'Alias'})
    with pytest.raises(ValueError):parameter_type(list[int],{}).encode([1]*4097)

def test_forwardrefs_and_empty_tuples_but_not_bare_typing_collections():
    from typing import List
    assert parameter_type(List['int'],{}).decode([1])==[1]
    assert parameter_type(tuple[()],{}).decode([])==()
    assert parameter_type('tuple[()]',{}).decode([])==()
    for annotation in (Tuple,'Tuple','tuple',List,'List'):
        with pytest.raises(ValueError):parameter_type(annotation,{})

def test_float_conversion_never_silently_rounds_integer_values():
    assert parameter_type(float,{}).decode(2)==2.0
    with pytest.raises(ValueError):parameter_type(float,{}).decode(2**53+1)

def test_namespace_alias_expansion_is_bounded_before_building_contract():
    namespace={'Leaf':int}
    for index in range(12):namespace['A'+str(index)]='tuple['+('Leaf' if index==0 else 'A'+str(index-1))+','+('Leaf' if index==0 else 'A'+str(index-1))+']'
    with pytest.raises(ValueError):resolve_annotation('A11',namespace)

def test_union_cannot_hide_work_limit_after_an_earlier_branch_matched():
    contract=parameter_type(list[int]|list[str],{})
    # Every attempted conversion shares one work allowance, even failed alternatives.
    with pytest.raises(ValueError):contract.encode([1]*2100)

def test_aggregate_bytes_refuse_before_visiting_whole_large_array(monkeypatch):
    import scone_memory.agents.function_types as module
    original=module._scalar;visits=[]
    def counted(value):
        visits.append(1)
        return original(value)
    monkeypatch.setattr(module,'_scalar',counted)
    with pytest.raises(ValueError):parameter_type(list[str],{}).encode(['x'*64000]*100)
    assert len(visits)<10, 'encoding allocated the complete oversized list before its byte check'

def test_defaults_require_exact_type_preserving_roundtrip():
    class Color(Enum):RED='red'
    for annotation,value in ((Color|str,Color.RED),(tuple[int,...]|list[int],(1,)),(float,1)):
        with pytest.raises(ValueError):parameter_type(annotation,{}).encode_default(value)
    assert parameter_type(Color|str,{}).encode_default('red')=='red'
    assert parameter_type(Color,{}).encode_default(Color.RED)=='red'
    assert parameter_type(float,{}).encode_default(1.0)==1.0
    assert parameter_type(tuple[int,...],{}).encode_default((1,))==[1]

def test_explicit_namespace_aliases_win_and_qualified_typing_is_verified():
    import typing
    assert parameter_type('int',{'int':str}).schema=={'type':'string'}
    assert parameter_type('List',{'List':int}).schema=={'type':'integer'}
    for namespace in ({},{'typing':object()}):
        with pytest.raises(ValueError):parameter_type('typing.List[int]',namespace)
    assert parameter_type('typing.List[int]',{'typing':typing}).schema=={'type':'array','items':{'type':'integer'}}

def test_generated_contracts_compile_with_existing_object_schema_boundary():
    from scone_memory.realtime.output_schema import compile_schema, accepts_schema
    import json
    values=[(Literal[1,True,'yes',None],True),(tuple[str,int],('a',2)),(tuple[()],()),
            (dict[str,list[int]],{'x':[1]}),(Optional[int],None),(int|float,1)]
    for annotation,value in values:
        contract=parameter_type(annotation,{})
        wrapped=compile_schema({'type':'object','properties':{'value':contract.schema},'required':['value'],'additionalProperties':False})
        assert accepts_schema(json.dumps({'value':contract.encode(value)}),wrapped)

def test_ambiguous_distinct_enum_conversions_refuse():
    class One(Enum):VALUE='same'
    class Two(Enum):VALUE='same'
    contract=parameter_type(One|Two,{})
    with pytest.raises(ValueError,match='ambiguous'):contract.decode('same')
    assert contract.encode(One.VALUE)=='same'
