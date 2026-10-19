"""The tutorial page only references files and pages that exist."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "docs" / "tutorial.md"


def test_tutorial_links_and_paths_exist() -> None:
    text = PAGE.read_text(encoding="utf-8")
    for target in re.findall(r"\]\(([^)#]+?)(?:#[^)]*)?\)", text):
        assert (PAGE.parent / target).exists(), target
    for path in re.findall(r"(?:examples|generate_hsx|simulate_hsx)[\w/.\-]*\.(?:py|toml|hsx)", text):
        assert (ROOT / path).is_file(), path


def test_tutorial_is_in_the_nav() -> None:
    assert "tutorial.md" in (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
