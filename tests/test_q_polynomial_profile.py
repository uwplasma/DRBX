"""Profile packaging and completion refusal; no GPU requirement for these checks."""
import importlib.util
import json
from pathlib import Path
import pytest

PATH=Path(__file__).resolve().parents[1]/'scripts/q08_polynomial_profile/profile.py'
spec=importlib.util.spec_from_file_location('q_polynomial_profile',PATH)
profile=importlib.util.module_from_spec(spec);spec.loader.exec_module(profile)


def test_overlay_is_current_package_source():
    manifest=profile.check_source()
    root=PATH.parents[2]
    for rel in manifest['files']:
        if rel.startswith('overlay/'):
            assert (PATH.parent/rel).read_bytes()==(root/'src/drbx/native'/Path(rel).name).read_bytes()


def test_completion_rejects_missing_and_failed(tmp_path):
    with pytest.raises(FileNotFoundError):profile.validate(tmp_path)
    (tmp_path/'completion.json').write_text(json.dumps(dict(passed=False)))
    with pytest.raises(ValueError,match='incomplete'):profile.validate(tmp_path)


def test_completion_checks_payload_bytes(tmp_path):
    payload=tmp_path/'results.json';payload.write_text('{}')
    receipt=dict(passed=True,files={'results.json':profile.sha(payload)},candidate_identity='test')
    (tmp_path/'completion.json').write_text(json.dumps(receipt))
    payload.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='checksum'):profile.validate(tmp_path)
