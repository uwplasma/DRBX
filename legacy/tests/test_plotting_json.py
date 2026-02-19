from __future__ import annotations

import json
from pathlib import Path

from jaxdrb.analysis.plotting import save_json
from jaxdrb.models.params import DRBParams


def test_save_json_handles_drbparams_with_line_bcs(tmp_path: Path) -> None:
    p = DRBParams()
    out = tmp_path / "params.json"
    save_json(out, {"params": p.__dict__})
    data = json.loads(out.read_text())
    assert isinstance(data, dict)
    assert "params" in data
    assert isinstance(data["params"], dict)
