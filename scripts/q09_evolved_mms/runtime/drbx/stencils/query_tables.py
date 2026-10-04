"""Shared exact-bit coordinate deduplication; request roles stay in caller schemas."""
import numpy as np

class ExactQueryTable:
    """Deduplicates float64 (3,) query points by exact bit pattern.

    Every stored point is a verbatim copy of the array it was given, so a
    later lookup returns the identical bits (never a recomputed value) —
    required for the "donor/target trace-query ids" to expand bitwise.
    """

    def __init__(self):
        self._index: dict[bytes, int] = {}
        self._points: list[np.ndarray] = []

    def add(self, point) -> int:
        point = np.array(point, dtype=np.float64, copy=True)
        if point.shape != (3,):
            raise ValueError("a query point must have shape (3,)")
        key = point.tobytes()
        idx = self._index.get(key)
        if idx is None:
            idx = len(self._points)
            self._index[key] = idx
            self._points.append(point)
        return idx

    def add_many(self, points) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("query points must have shape (count, 3)")
        return np.array([self.add(p) for p in points], dtype=np.int32)

    def array(self) -> np.ndarray:
        if not self._points:
            return np.zeros((0, 3), dtype=np.float64)
        return np.stack(self._points, axis=0)


_QueryTable = ExactQueryTable
