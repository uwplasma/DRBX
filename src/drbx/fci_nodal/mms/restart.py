"""Two committed generations of accepted manufactured-run state."""

from dataclasses import dataclass
import json
from math import isfinite
import os
from pathlib import Path

import numpy as np

from .._hashing import sha256_file as digest
from ..case._receipts import code_identity, read_json, verify_code, worker_code_files, write_json

SEGMENT_KEYS = ("segment_wall_seconds", "segment_steps", "checkpoint_interval_seconds")
CHECKPOINT_DIR = "checkpoint"
_PAYLOADS = ("owner0000.npy", "progress.json")


@dataclass(frozen=True)
class Paused:
    time: float
    steps: int


def segment_controls(options):
    controls = {key: options[key] for key in SEGMENT_KEYS if options.get(key) is not None}
    for key, value in controls.items():
        valid = type(value) is int or (key != "segment_steps" and type(value) is float)
        if not valid or value <= 0 or not isfinite(value):
            raise ValueError(f"[mms] {key} must be positive and finite")
    return controls


class Checkpoint:
    def __init__(self, root, identity):
        self.root = Path(root)
        self.identity = json.loads(json.dumps(identity, default=str, allow_nan=False))
        self.generation = -1

    def load(self):
        committed = [(path, read_json(path)) for path in self.root.glob("*/manifest.json")]
        if not committed:
            return None
        path, manifest = max(committed, key=lambda item: item[1]["generation"])
        self.generation = manifest["generation"]
        if manifest["schema"] != 1:
            raise ValueError("Unsupported MMS checkpoint schema")
        directory = path.parent
        files = manifest["files"]
        if set(files) != set(_PAYLOADS) or any(
            not (directory/name).is_file() or digest(directory/name) != files[name] for name in _PAYLOADS
        ):
            raise ValueError(f"MMS checkpoint file hash differs: {directory}")
        progress = read_json(directory/"progress.json")
        recorded = progress.pop("identity")
        try:
            verify_code("mms_run", progress)
        except (OSError, ValueError):
            changed = "code"
        else:
            changed = next((key for key in self.identity if recorded.get(key) != self.identity[key]), None)
        if changed is not None:
            print(f"MMS checkpoint identity differs: {changed}; starting from time zero", flush=True)
            return None
        del progress["code_files"], progress["code_sha256"]
        return np.load(directory/"owner0000.npy", allow_pickle=False), progress

    def save(self, state, progress):
        self.generation += 1
        directory = self.root/f"slot-{self.generation % 2}"
        directory.mkdir(parents=True, exist_ok=True)
        manifest = directory/"manifest.json"
        # Invalidate the overwritten slot before touching its files. The other
        # slot remains committed throughout publication.
        manifest.unlink(missing_ok=True)
        sources = worker_code_files("mms_run")
        progress = dict(progress, identity=self.identity, code_files=sources,
                        code_sha256=code_identity(sources))
        np.save(directory/"owner0000.npy", np.asarray(state), allow_pickle=False)
        write_json(directory/"progress.json", progress)
        pending = directory/"manifest.pending.json"
        write_json(pending, dict(schema=1, generation=self.generation,
                                files={name: digest(directory/name) for name in _PAYLOADS}))
        pending.replace(manifest)
