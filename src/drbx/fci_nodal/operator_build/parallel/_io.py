"""Local products retain the bundle's relative array and receipt names."""

import json
from pathlib import Path

import numpy as np


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def save(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.tmp')
    with pending.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    pending.replace(path)


def _value(item):
    return item.tolist() if isinstance(item, np.ndarray) else item.item()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.tmp')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False, default=_value) + '\n')
    pending.replace(path)
