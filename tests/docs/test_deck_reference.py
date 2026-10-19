"""docs/decks.md documents exactly the keys, types and defaults of the deck schema."""

import re
from pathlib import Path

import pytest

from drbx.deck import BACKENDS, schema_rows, table_specs

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "docs" / "decks.md"


def documented_tables() -> dict[tuple[str, str], list[tuple[str, str, str]]]:
    """``{(backend, table): [(key, type, default)]}`` read from the page's key tables."""

    tables: dict[tuple[str, str], list[tuple[str, str, str]]] = {}
    backend = table = None
    for line in PAGE.read_text(encoding="utf-8").splitlines():
        if match := re.match(r'`\[model\] backend = "(\w+)"`', line):
            backend, table = match.group(1), None
        elif match := re.match(r"#### `\[([\w.]+)\]`", line):
            table = match.group(1)
            tables[(backend, table)] = []
        elif line.startswith("### "):
            table = None
        elif table and (match := re.match(r"\| `(\w+)` \| `([^`]+)` \| `([^`]*)` \|", line)):
            tables[(backend, table)].append(match.groups())
    return tables


def documented_outputs(backend: str) -> tuple[set[str], set[str]]:
    """Array names and summary ``results`` keys documented for ``backend``."""

    text = PAGE.read_text(encoding="utf-8")
    section = text.split(f'`[model] backend = "{backend}"`')[1].split("\n## ")[0]
    outputs = section.split("### Outputs")[1]
    results = outputs.split("`results` keys:")[1] if "`results` keys:" in outputs else ""
    arrays = outputs.split("`results` keys:")[0]
    names = set()
    for row in re.findall(r"^\| ([^|]+) \|", arrays, flags=re.MULTILINE):
        names.update(re.findall(r"`(\w+)`", row))
    return names, set(re.findall(r"`(\w+)`", results))


def test_every_table_key_type_and_default_is_documented():
    documented = documented_tables()
    expected = {(b, t): schema_rows(c) for b in BACKENDS for t, c in table_specs(b).items()}
    assert set(documented) == set(expected), set(documented) ^ set(expected)
    for table, rows in expected.items():
        assert documented[table] == [tuple(row) for row in rows], table


def test_every_backend_has_a_section_and_nav_entry():
    text = PAGE.read_text(encoding="utf-8")
    for backend in BACKENDS:
        assert f'`[model] backend = "{backend}"`' in text
    assert "decks.md" in (ROOT / "mkdocs.yml").read_text(encoding="utf-8")


@pytest.mark.parametrize("backend", sorted(BACKENDS))
def test_outputs_are_documented(backend):
    arrays, results = documented_outputs(backend)
    assert arrays and (results or backend == "fci_braginskii")
