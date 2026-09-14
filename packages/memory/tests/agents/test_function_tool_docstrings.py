"""A function tool can take each parameter's description from its docstring.

Most documented Python functions describe their parameters in the
docstring -- Google's ``Args:``, NumPy's ``Parameters`` table, Sphinx's
``:param name:`` -- and a function tool gave the model none of it: only
``Annotated[int, "..."]`` reached a parameter's schema, and the whole
docstring, parameter section and all, became the tool's description.
Asked with ``describe_from_docstring=True``, each documented parameter's
text becomes its schema description, and the section leaves the tool's
description. It is opt-in because a tool's schema is part of the digest
that binds a pending approval.
"""
from __future__ import annotations

import inspect
from typing import Annotated

import pytest

from scone_memory.agents.custom_tools import ToolContext
from scone_memory.agents.docstring_parameters import parameter_descriptions, without_parameters
from scone_memory.agents.function_tools import function_tool


def google(city: str, days: int = 3, *, context: ToolContext) -> object:
    """Forecast the weather for a city.

    Looks a few days ahead.

    Args:
        city (str): The city to forecast,
            as its common English name.
        days: How many days ahead.
        context: Supplied by the host.

    Returns:
        A forecast.
    """
    return {}


def numpy(city: str, days: int = 3) -> object:
    """Forecast the weather for a city.

    Parameters
    ----------
    city : str
        The city to forecast.
    days : int, optional
        How many days ahead.

    Returns
    -------
    dict
    """
    return {}


def sphinx(city: str, days: Annotated[int, "Explicit wins."] = 3) -> object:
    """Forecast the weather for a city.

    :param city: The city
        to forecast.
    :type city: str
    :param int days: How many days ahead.
    :returns: A forecast.
    """
    return {}


@pytest.mark.parametrize("function", [google, numpy, sphinx])
def test_each_docstring_style_describes_its_parameters(function):
    found = parameter_descriptions(function.__doc__)
    assert found["city"].startswith("The city to forecast") and found["days"] == "How many days ahead."


def test_a_description_continued_on_the_next_line_is_one_line():
    assert parameter_descriptions(google.__doc__)["city"] == "The city to forecast, as its common English name."


def test_the_schema_carries_the_descriptions_and_the_tool_description_loses_the_section():
    tool = function_tool(google, revision="1", context_parameter="context", describe_from_docstring=True)
    properties = tool.parameters["properties"]
    assert properties["city"]["description"] == "The city to forecast, as its common English name."
    assert properties["days"]["description"] == "How many days ahead."
    assert "context" not in properties
    assert tool.description == "Forecast the weather for a city.\n\nLooks a few days ahead.\n\nReturns:\n    A forecast."


def test_an_annotated_description_wins_over_the_docstring():
    tool = function_tool(sphinx, revision="1", describe_from_docstring=True)
    assert tool.parameters["properties"]["days"]["description"] == "Explicit wins."
    assert tool.parameters["properties"]["city"]["description"] == "The city to forecast."
    assert tool.description == "Forecast the weather for a city.\n\n:returns: A forecast."


def test_without_asking_the_tool_is_exactly_as_before():
    tool = function_tool(numpy, revision="1")
    assert "description" not in tool.parameters["properties"]["city"]
    assert tool.description == inspect.getdoc(numpy)


def test_an_explicit_tool_description_is_kept_whole():
    tool = function_tool(google, revision="1", context_parameter="context", describe_from_docstring=True,
                         description="Given by the host.")
    assert tool.description == "Given by the host."
    assert tool.parameters["properties"]["city"]["description"].startswith("The city")


def test_names_the_signature_does_not_have_and_a_missing_section_describe_nothing():
    assert parameter_descriptions("Just a summary.") == {}
    assert parameter_descriptions(None) == {}
    tool = function_tool(numpy, revision="1", describe_from_docstring=True)
    assert set(tool.parameters["properties"]) == {"city", "days"}
    assert parameter_descriptions("Summary.\n\n    Args:\n        ghost: Not a parameter.\n        empty:\n") == {
        "ghost": "Not a parameter."}


def test_a_section_ends_at_the_next_heading_even_without_a_blank_line():
    google_style = "Summary.\n\n    Args:\n        city: The city.\n    Returns:\n        A forecast.\n"
    numpy_style = "Summary.\n\n    Parameters\n    ----------\n    city : str\n        The city.\n    Returns\n    -------\n    dict\n"
    for docstring in (google_style, numpy_style):
        assert parameter_descriptions(docstring) == {"city": "The city."}
        assert "Returns" in without_parameters(docstring) and "The city" not in without_parameters(docstring)
