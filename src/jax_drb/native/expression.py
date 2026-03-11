from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import jax.numpy as jnp

from ..config.boutinp import BoutConfig, ROOT_SECTION

_SAFE_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_SAFE_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_SAFE_CONSTANTS = {
    "e": jnp.e,
    "pi": jnp.pi,
}
_REFERENCE_PATTERN = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?::[A-Za-z_][A-Za-z0-9_]*)+\b")


def _max_fn(*args: Any) -> Any:
    result = args[0]
    for value in args[1:]:
        result = jnp.maximum(result, value)
    return result


def _min_fn(*args: Any) -> Any:
    result = args[0]
    for value in args[1:]:
        result = jnp.minimum(result, value)
    return result


_SAFE_FUNCTIONS = {
    "abs": jnp.abs,
    "cos": jnp.cos,
    "exp": jnp.exp,
    "log": jnp.log,
    "max": _max_fn,
    "min": _min_fn,
    "sin": jnp.sin,
    "sqrt": jnp.sqrt,
    "tan": jnp.tan,
}


@dataclass
class ArrayExpressionEvaluator:
    config: BoutConfig
    local_values: Mapping[str, Any]
    external_values: Mapping[str, Any] | None = None
    _cache: dict[tuple[str, str], Any] = field(default_factory=dict, init=False)

    def evaluate(self, expression: str, *, current_section: str = ROOT_SECTION) -> Any:
        return self._evaluate_expression(expression, current_section=current_section, seen=set())

    def resolve_option(self, section: str, key: str) -> Any:
        return self._resolve_option(section, key, seen=set())

    def _resolve_option(self, section: str, key: str, seen: set[tuple[str, str]]) -> Any:
        cache_key = (section, key)
        if cache_key in self._cache:
            return self._cache[cache_key]
        if cache_key in seen:
            cycle = " -> ".join(f"{entry_section}:{entry_key}" for entry_section, entry_key in (*seen, cache_key))
            raise ValueError(f"Cyclic array expression detected: {cycle}")

        if not self.config.has_option(section, key):
            raise KeyError(f"Missing option {section}:{key}")

        seen.add(cache_key)
        value = self.config.get(section, key)
        if isinstance(value.parsed, bool | int | float):
            result = value.parsed
        elif isinstance(value.parsed, tuple):
            raise TypeError(f"Option {section}:{key} is a list, not an array expression")
        else:
            result = self._evaluate_expression(value.raw, current_section=section, seen=seen)
        self._cache[cache_key] = result
        seen.remove(cache_key)
        return result

    def _evaluate_expression(self, expression: str, *, current_section: str, seen: set[tuple[str, str]]) -> Any:
        sanitized = expression.replace("π", "pi").replace("^", "**")
        sanitized = re.sub(r"(?<=\d)pi\b", "*pi", sanitized)
        references: dict[str, str] = {}

        def replace_reference(match: re.Match[str]) -> str:
            token = f"__ref_{len(references)}"
            references[token] = match.group(0)
            return token

        tree = ast.parse(_REFERENCE_PATTERN.sub(replace_reference, sanitized), mode="eval")
        return self._eval_node(tree.body, current_section=current_section, references=references, seen=seen)

    def _eval_node(
        self,
        node: ast.AST,
        *,
        current_section: str,
        references: Mapping[str, str],
        seen: set[tuple[str, str]],
    ) -> Any:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool | int | float):
                return node.value
            raise TypeError(f"Unsupported constant value: {node.value!r}")
        if isinstance(node, ast.Name):
            return self._resolve_name(node.id, current_section=current_section, references=references, seen=seen)
        if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_BINOPS:
            left = self._eval_node(node.left, current_section=current_section, references=references, seen=seen)
            right = self._eval_node(node.right, current_section=current_section, references=references, seen=seen)
            return _SAFE_BINOPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_UNARYOPS:
            operand = self._eval_node(node.operand, current_section=current_section, references=references, seen=seen)
            return _SAFE_UNARYOPS[type(node.op)](operand)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            function_name = node.func.id
            if function_name not in _SAFE_FUNCTIONS:
                raise TypeError(f"Unsupported array expression function {function_name!r}")
            args = [
                self._eval_node(argument, current_section=current_section, references=references, seen=seen)
                for argument in node.args
            ]
            return _SAFE_FUNCTIONS[function_name](*args)
        raise TypeError(f"Unsupported array expression node: {ast.dump(node, include_attributes=False)}")

    def _resolve_name(
        self,
        name: str,
        *,
        current_section: str,
        references: Mapping[str, str],
        seen: set[tuple[str, str]],
    ) -> Any:
        if name in references:
            section_name, key = references[name].rsplit(":", 1)
            return self._resolve_option(section_name, key, seen)
        if self.external_values is not None and name in self.external_values:
            return self.external_values[name]
        if name in self.local_values:
            return self.local_values[name]
        if name in _SAFE_CONSTANTS:
            return _SAFE_CONSTANTS[name]
        if current_section != ROOT_SECTION and self.config.has_option(current_section, name):
            return self._resolve_option(current_section, name, seen)
        if self.config.has_option(ROOT_SECTION, name):
            return self._resolve_option(ROOT_SECTION, name, seen)
        raise KeyError(f"Unknown array expression symbol {name!r} while evaluating section {current_section!r}")
