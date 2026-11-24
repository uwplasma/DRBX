"""Content and layout identity used in owner P construction."""

import hashlib
import json

import numpy as np


def sha256_arrays(arrays: dict, extra: str = "") -> str:
    """sha256 over ``(name, dtype, shape, bytes)`` of every array (sorted by name) and an extra string."""
    h = hashlib.sha256()
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name])
        h.update(f"{name}|{a.dtype}|{a.shape}|".encode())
        h.update(a.tobytes())
    h.update(extra.encode())
    return h.hexdigest()


def layout_identity(layout) -> str:
    """sha256 over the Zernike core-ring layout layout (``n``, planes, node coordinates, ``H`` weights). Platform dependent (see the module
    docstring)."""
    if layout.shape is not None:
        return sha256_arrays({"node_u": layout.node_u, "node_theta": layout.node_theta, "wxy": layout.wxy,
                              "node_ring": layout.node_ring}, json.dumps(layout.shape.to_dict(), sort_keys=True))
    return sha256_arrays({"node_u": layout.node_u, "node_theta": layout.node_theta, "wxy": layout.wxy,
                          "node_ring": np.asarray(layout.node_ring)},
                         extra=json.dumps(dict(n=layout.n, E=layout.n_eta, P=layout.P, deta=float(layout.deta),
                                               du=float(layout.du), layout_kind="zernike_core_ring"), sort_keys=True))


def jsonable(o):
    """``o`` with numpy scalars / arrays converted for ``json`` (unknown objects become ``str``)."""
    if o is None or isinstance(o, (str, int, float, bool)):
        return o
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
