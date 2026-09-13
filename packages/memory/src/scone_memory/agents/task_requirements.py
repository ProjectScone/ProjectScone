"""Persist authored task contracts; compile a separate copy for execution."""
from __future__ import annotations

from copy import deepcopy
from typing import cast

from pydantic import ConfigDict, field_validator

from ..realtime.answer_requirements import AnswerRequirements


class TaskAnswerRequirements(AnswerRequirements):
    """A saved plan retains the author's schema, including local definitions.

    BoundAgent snapshots this into ordinary AnswerRequirements for generation
    and publication. Schema validation has the same synchronous execution and
    bounded local-reference rules as standalone agent output requirements.
    """

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True, hide_input_in_errors=True)

    @field_validator('output_schema', mode='before')
    @classmethod
    def bounded_schema(cls, value: object) -> dict[str, object] | None:
        if value is None:
            return None
        from ..realtime.output_schema import compile_schema
        try:
            compile_schema(value)
        except ImportError:
            raise ValueError('task schemas require the structured-output extra') from None
        return deepcopy(cast(dict[str, object], value))


def output_schema_available() -> bool:
    """Advertise schema authoring only with the optional validator installed."""
    from ..realtime.output_schema import compile_schema
    try:
        compile_schema({})
    except ImportError:
        return False
    return True
