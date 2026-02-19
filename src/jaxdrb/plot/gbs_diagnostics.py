from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt


def plot_0d_time_traces(
    h5_path: str | Path,
    *,
    fields: Iterable[str] = ("globtheta", "globtemperature", "globomega"),
    output: str | Path | None = None,
) -> plt.Figure:
    import h5py  # type: ignore

    path = Path(h5_path)
    if not path.exists():
        raise FileNotFoundError(path)

    with h5py.File(path, "r") as f:
        if "data/var0d/time" not in f:
            raise KeyError("/data/var0d/time not found")
        t = np.asarray(f["data/var0d/time"][...])
        series = {}
        for name in fields:
            key = f"data/var0d/{name}"
            if key in f:
                series[name] = np.asarray(f[key][...])

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    for name, arr in series.items():
        ax.plot(t, arr, label=name)
    ax.set_xlabel("time")
    ax.set_ylabel("value")
    ax.legend(loc="best", fontsize=8)
    ax.set_title("GBS 0D diagnostics")
    fig.tight_layout()
    if output is not None:
        fig.savefig(str(output), dpi=150)
    return fig
