"""Names, identities and checkpoint readers shared by the JAX stage and the reduction (no JAX import)."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from p_shared import runner                                                       # noqa: E402
from p08_step5_combined import references                                         # noqa: E402

ARMS = ("presc", "solved")


def arm_key(arm: str, field: str, term: str) -> str:
    """Key of an arm's ``(n_owners,)`` array in a variant checkpoint."""
    return f"{arm}__{field}__{term}"


def jsonable(obj):
    """JSON-safe copy (numpy to Python; non-finite floats to ``None``, which ``runner.write_json`` would refuse)."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return jsonable(obj.item())
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def jax_identity(identity: str, n: int, artifact_identity_sha256: str | None) -> str:
    """Identity of the JAX-stage checkpoints: the campaign, the grid and the step-4 artifact."""
    return runner.digest({"campaign": identity, "stage": "jax", "n": int(n),
                          "artifact_identity_sha256": artifact_identity_sha256})


def artifact_sha(inputs: dict, n: int) -> str | None:
    return (inputs.get("grids", {}).get(str(n)) or {}).get("artifact_identity_sha256")


def variant_unit(n: int, variant: str) -> dict:
    """The ``runner`` unit under which a variant's arrays are checkpointed."""
    return {"stage": f"jax_{variant}", "n": int(n), "start": 0, "stop": 1}


def load_variant(output, n: int, variant: str, jid: str) -> tuple[dict, dict]:
    """``(arrays, info)`` of a checkpointed variant (checksum verified by ``runner.valid_unit``)."""
    work = references.work_dir(output)
    unit = variant_unit(n, variant)
    if not runner.valid_unit(work, unit, jid):
        raise ValueError(f"no valid JAX checkpoint for N{n} {variant}; run the jax stage first")
    receipt = json.loads(runner.receipt_path(work, unit).read_text())
    with np.load(runner.unit_path(work, unit), allow_pickle=False) as z:
        arrays = {name: z[name].copy() for name in z.files}
    return arrays, receipt["info"]
