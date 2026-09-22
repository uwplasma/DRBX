import hashlib
import json
import os
from pathlib import Path
import numpy as np

SCHEMA = "q03-direct-cubic-v1"
FIELDS = ("radial_eta", "angular_x", "mixed_y_eta", "constant")


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True,
                              default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item()) + '\n')
    os.replace(tmp, path)


def source_identity():
    return {p.name: sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def load_inputs(path, verify=True):
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    if manifest['schema'] != SCHEMA:
        raise RuntimeError('incompatible input schema')
    if verify:
        for name, digest in manifest['files'].items():
            if sha(path / name) != digest:
                raise RuntimeError(f'corrupt input: {name}')
    arrays = {Path(name).stem: np.load(path / name, mmap_mode='r', allow_pickle=False)
              for name in manifest['files'] if name.endswith('.npy')}
    if tuple(manifest['fields']) != FIELDS or arrays['target'].shape != (len(arrays['face.ids']), 20):
        raise RuntimeError('incompatible field catalogue or target basis')
    from .numerics import POLICY
    if manifest['identity']['policy'] != POLICY:
        raise RuntimeError('incompatible numerical input policy')
    return arrays, manifest
