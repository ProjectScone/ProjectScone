"""Exact bounded arithmetic over quoted passages, independent of any model/store.

This proves where operands occur and how a result was calculated. It does not
prove that operands, units, selected lists or interpretations answer a question.
"""
from __future__ import annotations

import math
import re
from fractions import Fraction
from typing import Literal, Self, TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator

Operation = Literal['sum', 'product', 'count', 'difference', 'ratio', 'compare', 'compare_counts']
OPERATIONS: tuple[Operation, ...] = ('sum', 'product', 'count', 'difference', 'ratio', 'compare', 'compare_counts')


class QuotedInput(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, revalidate_instances='always')
    chunk_id: int = Field(gt=0, lt=2**63)
    quote: str = Field(min_length=1, max_length=512)

    @model_validator(mode='after')
    def bounded_quote(self) -> Self:
        if not self.quote.strip() or len(self.quote.encode()) > 2048:
            raise ValueError('invalid quote')
        return self


class ComputeMemoryArgs(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, revalidate_instances='always')
    operation: Operation
    left: list[QuotedInput] = Field(min_length=1, max_length=16)
    right: list[QuotedInput] = Field(default_factory=list, max_length=16)

    @model_validator(mode='after')
    def bounded_operands(self) -> Self:
        if len(self.left) + len(self.right) > 16:
            raise ValueError('at most 16 quoted inputs')
        if self.operation in ('sum', 'product', 'count') and self.right:
            raise ValueError('operation takes only left inputs')
        if self.operation in ('difference', 'ratio', 'compare') and (len(self.left) != 1 or len(self.right) != 1):
            raise ValueError('operation takes one input per side')
        if self.operation == 'compare_counts' and not self.right:
            raise ValueError('comparison needs both groups')
        return self


class ComputationError(ValueError):
    def __init__(self, reason: str = 'invalid_computation') -> None:
        self.reason = reason if reason in ('invalid_computation', 'evidence_unavailable',
            'ambiguous_quote', 'numeric_literal_required') else 'invalid_computation'
        super().__init__(self.reason)


class MatchedInput(TypedDict):
    chunk_id: int
    quote: str
    start_char: int
    end_char: int


def _resolve(inputs: list[QuotedInput], passages: dict[int, str]) -> list[MatchedInput]:
    spans: list[MatchedInput] = []
    for item in inputs:
        text = passages.get(item.chunk_id)
        if not isinstance(text, str):
            raise ComputationError('evidence_unavailable')
        start = text.find(item.quote)
        if start < 0:
            raise ComputationError('evidence_unavailable')
        if text.find(item.quote, start + 1) >= 0:
            raise ComputationError('ambiguous_quote')
        end = start + len(item.quote)
        if any(row['chunk_id'] == item.chunk_id and start < row['end_char'] and row['start_char'] < end for row in spans):
            raise ComputationError()
        spans.append({'chunk_id':item.chunk_id, 'quote':item.quote, 'start_char':start, 'end_char':end})
    return spans


def _number(span: MatchedInput, text: str) -> Fraction:
    quote, start, end = span['quote'], span['start_char'], span['end_char']
    if not re.fullmatch(r'[+-]?[0-9]{1,30}(?:\.[0-9]{1,18})?', quote):
        raise ComputationError('numeric_literal_required')
    before, after = text[:start], text[end:]
    # Reject fragments of identifiers, signed numbers, decimals or grouped
    # numbers. Sentence punctuation is allowed, but never numeric continuation.
    if (before and (before[-1].isalnum() or before[-1] in '_+-−')
            or after and (after[0].isalnum() or after[0] in '_+-−')
            or before.endswith('.')
            or len(before) >= 2 and before[-1] == ',' and before[-2].isdigit()
            or len(after) >= 2 and after[0] in '.,' and after[1].isdigit()):
        raise ComputationError('numeric_literal_required')
    return Fraction(quote)


def _render(value: Fraction) -> str:
    denominator = value.denominator
    twos = fives = 0
    while denominator % 2 == 0:
        denominator //= 2
        twos += 1
    while denominator % 5 == 0:
        denominator //= 5
        fives += 1
    if denominator != 1:
        return str(value)
    places = max(twos, fives)
    if not places:
        return str(value.numerator)
    numerator = abs(value.numerator) * 2**(places-twos) * 5**(places-fives)
    digits = str(numerator).zfill(places + 1)
    result = (digits[:-places] + '.' + digits[-places:]).rstrip('0').rstrip('.')
    return ('-' if value < 0 else '') + result


def _compare(left: Fraction, right: Fraction) -> str:
    return 'less' if left < right else 'greater' if left > right else 'equal'


def evaluate_computation(args: ComputeMemoryArgs, passages: dict[int, str]) -> dict[str, object]:
    """Resolve at most 16 exact unique quotes and calculate without rounding.

    Offsets are Python Unicode character offsets within the original chunk.
    Lists contain selected mentions only; duplicate entities in separate mentions
    are not deduplicated. No unit conversion or semantic inference is performed.
    """
    args = ComputeMemoryArgs.model_validate(args.model_dump())
    left, right = _resolve(args.left, passages), _resolve(args.right, passages)
    counts = args.operation in ('count', 'compare_counts')
    a = [Fraction(len(left))] if counts else [_number(row, passages[row['chunk_id']]) for row in left]
    b = [Fraction(len(right))] if counts else [_number(row, passages[row['chunk_id']]) for row in right]
    result: dict[str, object] = {'operation':args.operation, 'left':left, 'right':right,
        'coverage':'selected_spans_only', 'verified_accuracy':False,
        'notice':'Exact calculation on selected quotes. Operand meaning, unit compatibility and list completeness are not verified.'}
    if args.operation in ('compare', 'compare_counts'):
        result.update(value=_compare(a[0], b[0]), left_value=_render(a[0]), right_value=_render(b[0]))
        return result
    if args.operation == 'ratio':
        if b[0] == 0:
            raise ComputationError()
        value = a[0] / b[0]
    elif args.operation == 'difference':
        value = a[0] - b[0]
    elif args.operation == 'product':
        value = math.prod(a, start=Fraction(1))
    else:
        value = sum(a, start=Fraction(0))
    result['value'] = _render(value)
    return result


def validate_computation(value: object, passages: dict[int, str]) -> dict[str, object]:
    """Recompute a receipt before presenting it alongside source evidence."""
    import json

    if not isinstance(value, dict):
        raise ComputationError()
    inputs: dict[str, object] = {'operation':value.get('operation')}
    for side in ('left', 'right'):
        rows = value.get(side)
        if not isinstance(rows, list) or len(rows) > 16 or any(not isinstance(row, dict) for row in rows):
            raise ComputationError()
        inputs[side] = [{'chunk_id':row.get('chunk_id'), 'quote':row.get('quote')} for row in rows]
    result = evaluate_computation(ComputeMemoryArgs.model_validate(inputs), passages)
    if json.dumps(result, sort_keys=True, allow_nan=False) != json.dumps(value, sort_keys=True, allow_nan=False):
        raise ComputationError()
    return result
