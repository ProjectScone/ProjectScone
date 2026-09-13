"""Infer an application tool from a host-owned annotated Python callable."""
from __future__ import annotations

from collections.abc import Callable, Mapping
import ast
from copy import deepcopy
from dataclasses import dataclass
import functools
import inspect
import json
import tokenize
import types

from .custom_tools import AgentTool, ToolContext, _check_dispatch, _encoded
from .function_types import ParameterType, parameter_type, resolve_annotation
from ..realtime.output_schema import accepts_schema, compile_schema


def _source_annotations(function: types.FunctionType) -> dict[str, str]:
    # Python 3.14's STRING format can first execute the real __annotate__.
    # Source syntax supplies annotations without calling that machinery.
    try:
        with tokenize.open(function.__code__.co_filename) as stream:
            source = stream.read(1048577)
        if len(source.encode('utf-8')) > 1048576:
            raise ValueError('function source byte limit')
        tree = ast.parse(source)
        compiled = compile(source, function.__code__.co_filename, 'exec', dont_inherit=True)
    except (OSError, TypeError, SyntaxError, UnicodeError, RecursionError):
        raise ValueError('deferred function annotations require readable Python source') from None
    annotate = getattr(function, '__annotate__', None)
    if not inspect.isfunction(annotate):
        raise ValueError('deferred function annotations require original source')
    pending = [compiled]
    matched = False
    count = 0
    while pending:
        code = pending.pop()
        count += 1
        if count > 4096:
            raise ValueError('function source structure limit')
        if code.co_qualname == annotate.__code__.co_qualname and code == annotate.__code__:
            matched = True
            break
        pending.extend(value for value in code.co_consts if isinstance(value, types.CodeType))
    if not matched:
        raise ValueError('function source does not match its loaded annotations')
    definitions = [node for node in ast.walk(tree)
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and min([node.lineno, *(decorator.lineno for decorator in node.decorator_list)]) == function.__code__.co_firstlineno]
    if len(definitions) != 1:
        raise ValueError('deferred function annotations require a source definition')
    arguments = definitions[0].args
    parameters = [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]
    return {parameter.arg: ast.unparse(parameter.annotation)
            for parameter in parameters if parameter.annotation is not None}


def _signature_callable(function: object, depth: int = 0) -> object:
    if depth > 32:
        raise ValueError('function wrapper depth limit')
    if isinstance(function, functools.partial):
        target = _signature_callable(function.func, depth + 1)
        if not callable(target):
            raise ValueError('function tool requires a callable signature')
        return functools.partial(target, *function.args, **function.keywords)
    if inspect.ismethod(function):
        target = _signature_callable(function.__func__, depth + 1)
        if not callable(target):
            raise ValueError('function tool requires a callable signature')
        return types.MethodType(target, function.__self__)
    if inspect.isfunction(function):
        target = inspect.unwrap(function)
        if target is not function:
            return _signature_callable(target, depth + 1)
        if getattr(function, '__annotate__', None) is None:
            return function
        clone = types.FunctionType(function.__code__, function.__globals__, function.__name__,
                                   function.__defaults__, function.__closure__)
        clone.__kwdefaults__ = function.__kwdefaults__
        clone.__annotations__ = _source_annotations(function)
        return clone
    if callable(function) and not inspect.isclass(function):
        return _signature_callable(function.__call__, depth + 1)
    raise ValueError('function tool requires an inspectable Python callable')


def _signature(function: Callable[..., object]) -> inspect.Signature:
    target = _signature_callable(function)
    if not callable(target):
        raise ValueError('function tool requires an inspectable signature')
    try:
        return inspect.signature(target, eval_str=False)
    except (TypeError, ValueError):
        raise ValueError('function tool requires an inspectable signature') from None


def _namespace(function: Callable[..., object], supplied: Mapping[str, object] | None) -> dict[str, object]:
    target: object = function
    for _ in range(32):
        if isinstance(target, functools.partial):
            target = target.func
        elif inspect.ismethod(target):
            target = target.__func__
        elif inspect.isfunction(target):
            unwrapped = inspect.unwrap(target)
            if unwrapped is target:
                break
            target = unwrapped
        elif callable(target):
            target = target.__call__
        else:
            break
    namespace = dict(getattr(target, '__globals__', {}))
    if supplied is not None:
        if not isinstance(supplied, Mapping) or any(not isinstance(key, str) for key in supplied):
            raise ValueError('function annotation namespace requires string names')
        namespace.update(supplied)
    return namespace


@dataclass(frozen=True)
class _Parameter:
    name: str
    positional: bool
    contract: ParameterType | None


def _parameters(signature: inspect.Signature, namespace: Mapping[str, object],
                context_parameter: str | None) -> tuple[tuple[_Parameter, ...], dict[str, object], str]:
    if len(signature.parameters) > 32:
        raise ValueError('function tool parameter count limit')
    if context_parameter is not None and (not isinstance(context_parameter, str) or context_parameter not in signature.parameters):
        raise ValueError('function context parameter is not in the signature')
    parameters = []
    properties: dict[str, object] = {}
    required: list[str] = []
    defaults: dict[str, object] = {}
    for parameter in signature.parameters.values():
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise ValueError('function tools require explicit nonvariadic parameters')
        positional = parameter.kind == inspect.Parameter.POSITIONAL_ONLY
        if parameter.name == context_parameter:
            if (parameter.annotation is not inspect.Parameter.empty
                    and resolve_annotation(parameter.annotation, namespace) is not ToolContext):
                raise ValueError('injected function context must be annotated ToolContext')
            parameters.append(_Parameter(parameter.name, positional, None))
            continue
        if parameter.annotation is inspect.Parameter.empty:
            raise ValueError('function tool parameters require annotations')
        contract = parameter_type(parameter.annotation, namespace)
        schema = deepcopy(contract.schema)
        if parameter.default is inspect.Parameter.empty:
            required.append(parameter.name)
        else:
            encoded_default = contract.encode_default(parameter.default)
            encoded = _encoded({'value': encoded_default}, 32768)
            validator = compile_schema({'type': 'object', 'properties': {'value': schema},
                                        'required': ['value'], 'additionalProperties': False})
            if not accepts_schema(encoded, validator):
                raise ValueError('function default does not satisfy its annotation')
            defaults[parameter.name] = json.loads(encoded)['value']
            schema['default'] = defaults[parameter.name]
        properties[parameter.name] = schema
        parameters.append(_Parameter(parameter.name, positional, contract))
    result: dict[str, object] = {'type': 'object', 'properties': properties, 'additionalProperties': False}
    if required:
        result['required'] = required
    return tuple(parameters), result, _encoded(defaults, 32768)


def function_tool(function: Callable[..., object], *, revision: str,
                  name: str | None = None, description: str | None = None,
                  context_parameter: str | None = None, max_output_bytes: int = 16000,
                  annotation_namespace: Mapping[str, object] | None = None) -> AgentTool:
    """Adapt an annotated sync/async function without evaluating annotations.

    Names and descriptions default to the callable's name and docstring. Explicit
    revisions describe trusted handler/configuration changes; code is not hashed.
    Defaults are snapshotted and supplied on every call. Only an explicitly named
    context parameter receives the host ToolContext, outside the model schema.
    """
    if not callable(function) or inspect.isgeneratorfunction(function) or inspect.isasyncgenfunction(function):
        raise ValueError('function tools require a synchronous or asynchronous callable')
    signature = _signature(function)
    parameters, schema, defaults_json = _parameters(signature, _namespace(function, annotation_namespace), context_parameter)

    def invoke(arguments: dict[str, object], context: ToolContext) -> object:
        values: dict[str, object] = json.loads(defaults_json)
        values.update(arguments)
        positional: list[object] = []
        keywords: dict[str, object] = {}
        for parameter in parameters:
            value = context if parameter.contract is None else parameter.contract.decode(values[parameter.name])
            if parameter.positional:
                positional.append(value)
            else:
                keywords[parameter.name] = value
        _check_dispatch(context)
        return function(*positional, **keywords)

    selected_name = getattr(function, '__name__', '') if name is None else name
    selected_description = (inspect.getdoc(function) or '') if description is None else description
    return AgentTool(selected_name, selected_description, revision, schema, invoke, max_output_bytes)


__all__ = ['function_tool']
