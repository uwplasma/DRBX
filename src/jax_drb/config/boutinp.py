from __future__ import annotations

import ast
import math
import operator
import re
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python < 3.11
    import tomli as tomllib

ROOT_SECTION = "__root__"

_MISSING = object()
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
_SAFE_FUNCTIONS = {
    "abs": abs,
    "cos": math.cos,
    "exp": math.exp,
    "log": math.log,
    "max": max,
    "min": min,
    "sin": math.sin,
    "sqrt": math.sqrt,
    "tan": math.tan,
}
_SAFE_CONSTANTS = {
    "e": math.e,
    "pi": math.pi,
}
_REFERENCE_PATTERN = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?::[A-Za-z_][A-Za-z0-9_]*)+\b")
_INTEGER_PATTERN = re.compile(r"[+-]?\d+")


@dataclass(frozen=True)
class OptionValue:
    raw: str
    parsed: bool | int | float | str | tuple[str, ...]
    kind: str


@dataclass(frozen=True)
class OptionEntry:
    key: str
    value: OptionValue
    line: int


@dataclass(frozen=True)
class OptionSection:
    name: str
    entries: "OrderedDict[str, OptionEntry]"

    def __contains__(self, key: str) -> bool:
        return key in self.entries

    def __getitem__(self, key: str) -> OptionEntry:
        return self.entries[key]

    def get(self, key: str, default: Any = None) -> OptionEntry | Any:
        return self.entries.get(key, default)

    def keys(self) -> Iterable[str]:
        return self.entries.keys()

    def items(self) -> Iterable[tuple[str, OptionEntry]]:
        return self.entries.items()

    def values(self) -> Iterable[OptionEntry]:
        return self.entries.values()


@dataclass(frozen=True)
class BoutConfig:
    sections: "OrderedDict[str, OptionSection]"

    def section_names(self, *, include_root: bool = False) -> tuple[str, ...]:
        names = tuple(self.sections)
        if include_root:
            return names
        return tuple(name for name in names if name != ROOT_SECTION)

    def has_section(self, name: str) -> bool:
        return name in self.sections

    def section(self, name: str = ROOT_SECTION) -> OptionSection:
        if name not in self.sections:
            raise KeyError(f"Unknown section {name!r}")
        return self.sections[name]

    def has_option(self, section: str, key: str) -> bool:
        return self.has_section(section) and key in self.sections[section]

    def entry(self, section: str, key: str) -> OptionEntry:
        return self.section(section)[key]

    def get(self, section: str, key: str, default: Any = _MISSING) -> OptionValue | Any:
        if self.has_option(section, key):
            return self.entry(section, key).value
        if default is not _MISSING:
            return default
        raise KeyError(f"Missing option {section}:{key}")

    def raw(self, section: str, key: str) -> str:
        return self.entry(section, key).value.raw

    def parsed(self, section: str, key: str) -> bool | int | float | str | tuple[str, ...]:
        return self.entry(section, key).value.parsed


def apply_bout_overrides(config: BoutConfig, overrides: Iterable[str]) -> BoutConfig:
    sections: "OrderedDict[str, OrderedDict[str, OptionEntry]]" = OrderedDict(
        (name, OrderedDict(section.entries)) for name, section in config.sections.items()
    )
    for index, override in enumerate(overrides, start=1):
        if "=" not in override:
            raise ValueError(f"Override must be an assignment: {override!r}")
        raw_key, raw_value = override.split("=", 1)
        key = raw_key.strip()
        value = raw_value.strip()
        if ":" in key:
            section_name, option_name = key.split(":", 1)
        else:
            section_name, option_name = ROOT_SECTION, key
        sections.setdefault(section_name, OrderedDict())
        existing = sections[section_name].get(option_name)
        line = existing.line if existing is not None else -index
        sections[section_name][option_name] = OptionEntry(
            key=option_name,
            value=_parse_value(value),
            line=line,
        )
    return BoutConfig(
        OrderedDict(
            (name, replace(config.sections.get(name, OptionSection(name=name, entries=OrderedDict())), entries=entries))
            for name, entries in sections.items()
        )
    )


def load_bout_input(path: str | Path) -> BoutConfig:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".toml":
        return parse_toml_input(text)
    return parse_bout_input(text)


def parse_toml_input(text: str) -> BoutConfig:
    parsed = tomllib.loads(text)
    raw_sections: "OrderedDict[str, OrderedDict[str, OptionEntry]]" = OrderedDict()
    raw_sections[ROOT_SECTION] = OrderedDict()
    next_line = -1

    def add_entry(section_name: str, key: str, value: Any) -> None:
        nonlocal next_line
        raw_sections.setdefault(section_name, OrderedDict())
        raw_sections[section_name][key] = OptionEntry(
            key=key,
            value=_parse_value(_serialize_toml_value(value)),
            line=next_line,
        )
        next_line -= 1

    def visit_table(prefix: tuple[str, ...], table: Mapping[str, Any]) -> None:
        for key, value in table.items():
            path = (*prefix, key)
            if isinstance(value, Mapping):
                if _looks_like_scalar_wrapper(value):
                    add_entry(_toml_section_name(prefix), key, value)
                    continue
                visit_table(path, value)
                continue
            add_entry(_toml_section_name(prefix), key, value)

    for key, value in parsed.items():
        if isinstance(value, Mapping):
            visit_table((key,), value)
            continue
        add_entry(ROOT_SECTION, key, value)

    sections: "OrderedDict[str, OptionSection]" = OrderedDict(
        (name, OptionSection(name=name, entries=entries)) for name, entries in raw_sections.items()
    )
    return BoutConfig(sections=sections)


def parse_bout_input(text: str) -> BoutConfig:
    raw_sections: "OrderedDict[str, OrderedDict[str, OptionEntry]]" = OrderedDict()
    raw_sections[ROOT_SECTION] = OrderedDict()
    current_section = ROOT_SECTION
    pending_key: str | None = None
    pending_line = 0
    pending_chunks: list[str] = []
    pending_balance = 0

    def commit(section_name: str, key: str, raw_value: str, line: int) -> None:
        raw_sections.setdefault(section_name, OrderedDict())
        raw_sections[section_name][key] = OptionEntry(
            key=key,
            value=_parse_value(raw_value),
            line=line,
        )

    for line_number, physical_line in enumerate(text.splitlines(), start=1):
        logical_line = _strip_inline_comment(physical_line).strip()
        if not logical_line:
            continue

        if pending_key is not None:
            pending_chunks.append(logical_line)
            pending_balance += _structural_balance(logical_line)
            if pending_balance <= 0:
                commit(current_section, pending_key, " ".join(pending_chunks), pending_line)
                pending_key = None
                pending_chunks = []
                pending_balance = 0
            continue

        if logical_line.startswith("[") and logical_line.endswith("]"):
            current_section = logical_line[1:-1].strip()
            raw_sections.setdefault(current_section, OrderedDict())
            continue

        if "=" not in logical_line:
            raise ValueError(f"Line {line_number} is not a section header or assignment: {physical_line!r}")

        key, raw_value = logical_line.split("=", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        balance = _structural_balance(raw_value)
        if balance > 0:
            pending_key = key
            pending_line = line_number
            pending_chunks = [raw_value]
            pending_balance = balance
            continue

        commit(current_section, key, raw_value, line_number)

    if pending_key is not None:
        raise ValueError(f"Unclosed multiline value for {current_section}:{pending_key}")

    sections: "OrderedDict[str, OptionSection]" = OrderedDict(
        (name, OptionSection(name=name, entries=entries)) for name, entries in raw_sections.items()
    )
    return BoutConfig(sections=sections)


class NumericResolver:
    def __init__(self, config: BoutConfig, external_values: Mapping[str, float] | None = None):
        self.config = config
        self.external_values = dict(external_values or {})
        self._cache: dict[tuple[str, str], float] = {}

    def resolve(self, section: str, key: str) -> float:
        return self._resolve_option(section, key, seen=set())

    def evaluate(self, expression: str, *, current_section: str = ROOT_SECTION) -> float:
        return self._evaluate_expression(expression, current_section=current_section, seen=set())

    def _resolve_option(self, section: str, key: str, seen: set[tuple[str, str]]) -> float:
        cache_key = (section, key)
        if cache_key in self._cache:
            return self._cache[cache_key]
        if cache_key in seen:
            cycle = " -> ".join(f"{sec}:{name}" for sec, name in (*seen, cache_key))
            raise ValueError(f"Cyclic numeric reference detected: {cycle}")
        seen.add(cache_key)
        value = self.config.get(section, key)

        if isinstance(value.parsed, bool):
            result = float(value.parsed)
        elif isinstance(value.parsed, int | float):
            result = float(value.parsed)
        elif isinstance(value.parsed, tuple):
            raise TypeError(f"Option {section}:{key} is a list, not a scalar numeric expression")
        else:
            result = self._evaluate_expression(value.raw, current_section=section, seen=seen)

        self._cache[cache_key] = result
        seen.remove(cache_key)
        return result

    def _evaluate_expression(self, expression: str, *, current_section: str, seen: set[tuple[str, str]]) -> float:
        sanitized = expression.replace("π", "pi").replace("^", "**")
        sanitized = re.sub(r"(?<=\d)pi\b", "*pi", sanitized)
        references: dict[str, str] = {}

        def replace_reference(match: re.Match[str]) -> str:
            token = f"__ref_{len(references)}"
            references[token] = match.group(0)
            return token

        sanitized = _REFERENCE_PATTERN.sub(replace_reference, sanitized)
        tree = ast.parse(sanitized, mode="eval")
        return float(self._eval_node(tree.body, current_section=current_section, references=references, seen=seen))

    def _eval_node(
        self,
        node: ast.AST,
        *,
        current_section: str,
        references: Mapping[str, str],
        seen: set[tuple[str, str]],
    ) -> float:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, int | float):
                return float(node.value)
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
                raise TypeError(f"Unsupported numeric function {function_name!r}")
            args = [
                self._eval_node(argument, current_section=current_section, references=references, seen=seen)
                for argument in node.args
            ]
            return float(_SAFE_FUNCTIONS[function_name](*args))
        raise TypeError(f"Unsupported numeric expression node: {ast.dump(node, include_attributes=False)}")

    def _resolve_name(
        self,
        name: str,
        *,
        current_section: str,
        references: Mapping[str, str],
        seen: set[tuple[str, str]],
    ) -> float:
        if name in references:
            reference = references[name]
            section_name, key = reference.rsplit(":", 1)
            return self._resolve_option(section_name, key, seen)
        if name in self.external_values:
            return float(self.external_values[name])
        if name in _SAFE_CONSTANTS:
            return _SAFE_CONSTANTS[name]
        if current_section != ROOT_SECTION and self.config.has_option(current_section, name):
            return self._resolve_option(current_section, name, seen)
        if self.config.has_option(ROOT_SECTION, name):
            return self._resolve_option(ROOT_SECTION, name, seen)
        raise KeyError(f"Unknown numeric symbol {name!r} while evaluating section {current_section!r}")


def _parse_value(raw: str) -> OptionValue:
    value = raw.strip()
    if not value:
        return OptionValue(raw="", parsed="", kind="empty")

    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return OptionValue(raw=value, parsed=value[1:-1], kind="string")

    lowercase = value.lower()
    if lowercase == "true":
        return OptionValue(raw=value, parsed=True, kind="bool")
    if lowercase == "false":
        return OptionValue(raw=value, parsed=False, kind="bool")

    items = _parse_sequence(value)
    if items is not None:
        return OptionValue(raw=value, parsed=items, kind="list")

    if _INTEGER_PATTERN.fullmatch(value):
        return OptionValue(raw=value, parsed=int(value), kind="int")

    try:
        parsed_float = float(value)
    except ValueError:
        return OptionValue(raw=value, parsed=value, kind="expression")
    return OptionValue(raw=value, parsed=parsed_float, kind="float")


def _looks_like_scalar_wrapper(value: Mapping[str, Any]) -> bool:
    return any(key in value for key in ("expr", "raw", "ref", "value", "string"))


def _toml_section_name(path: tuple[str, ...]) -> str:
    if not path:
        return ROOT_SECTION
    head, *tail = path
    if head in {"root", "time"}:
        return ROOT_SECTION
    if head in {"species", "fields"} and tail:
        return tail[0]
    if head == "runtime":
        return "runtime" if not tail else ":".join((head, *tail))
    return ":".join(path)


def _serialize_toml_value(value: Any) -> str:
    if isinstance(value, Mapping):
        if "expr" in value:
            return str(value["expr"])
        if "raw" in value:
            return str(value["raw"])
        if "ref" in value:
            return str(value["ref"])
        if "value" in value:
            return _serialize_toml_value(value["value"])
        if "string" in value:
            return _quote_toml_string(str(value["string"]))
        raise TypeError(f"Unsupported TOML scalar wrapper keys: {tuple(value)}")
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format(value, ".16g")
    if isinstance(value, str):
        return _quote_toml_string(value)
    if isinstance(value, tuple | list):
        items = ", ".join(_serialize_toml_sequence_item(item) for item in value)
        if len(value) == 1:
            items = f"{items},"
        return f"({items})"
    raise TypeError(f"Unsupported TOML input value type: {type(value)!r}")


def _serialize_toml_sequence_item(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format(value, ".16g")
    if isinstance(value, Mapping):
        return _serialize_toml_value(value)
    raise TypeError(f"Unsupported TOML sequence item type: {type(value)!r}")


def _quote_toml_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _parse_sequence(value: str) -> tuple[str, ...] | None:
    candidate = value
    if value.startswith("(") and value.endswith(")"):
        candidate = value[1:-1].strip()
    parts = tuple(part.strip() for part in _split_top_level_commas(candidate))
    if len(parts) <= 1:
        return None
    return tuple(part for part in parts if part)


def _split_top_level_commas(value: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    paren_depth = 0
    bracket_depth = 0
    brace_depth = 0
    in_single = False
    in_double = False

    for index, char in enumerate(value):
        previous = value[index - 1] if index else ""
        if char == "'" and not in_double and previous != "\\":
            in_single = not in_single
        elif char == '"' and not in_single and previous != "\\":
            in_double = not in_double
        elif not in_single and not in_double:
            if char == "(":
                paren_depth += 1
            elif char == ")":
                paren_depth -= 1
            elif char == "[":
                bracket_depth += 1
            elif char == "]":
                bracket_depth -= 1
            elif char == "{":
                brace_depth += 1
            elif char == "}":
                brace_depth -= 1
            elif char == "," and paren_depth == 0 and bracket_depth == 0 and brace_depth == 0:
                parts.append("".join(current).strip())
                current = []
                continue
        current.append(char)

    parts.append("".join(current).strip())
    return parts


def _strip_inline_comment(line: str) -> str:
    in_single = False
    in_double = False
    for index, char in enumerate(line):
        previous = line[index - 1] if index else ""
        if char == "'" and not in_double and previous != "\\":
            in_single = not in_single
        elif char == '"' and not in_single and previous != "\\":
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return line[:index]
    return line


def _structural_balance(value: str) -> int:
    balance = 0
    in_single = False
    in_double = False
    for index, char in enumerate(value):
        previous = value[index - 1] if index else ""
        if char == "'" and not in_double and previous != "\\":
            in_single = not in_single
        elif char == '"' and not in_single and previous != "\\":
            in_double = not in_double
        elif not in_single and not in_double:
            if char in "([{":
                balance += 1
            elif char in ")]}":
                balance -= 1
    return balance
