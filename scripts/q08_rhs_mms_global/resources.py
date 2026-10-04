"""Empirical host guard for the Q08 scientific merge, not a universal bound."""
import math

GIB = 2**30


def host_guard(estimate, host_gib):
    """Keep the pre-existing 3x + 8 GiB policy, with explicit provenance.

    The older byte census missed Python/library and retained-runtime overhead.
    Replay RSS/census ratios were 2.07/1.38/1.42 when including its BC cache,
    or 2.34/1.56/1.60 relative to the base census used by this guard.
    This guard is calibrated for that workflow. It is not a production payload
    estimate or a proof for different grids, donor policies or runtimes.
    """
    base = estimate['estimated_host_peak_upper_bytes']
    if (not math.isfinite(host_gib) or host_gib <= 0 or
            isinstance(base, bool) or not math.isfinite(base) or base <= 0):
        raise ValueError('positive finite host budget and byte census required')
    budget = int(host_gib*GIB)
    observed = estimate.get('observed_host_available_bytes', budget)
    if not math.isfinite(observed) or observed <= 0:
        raise ValueError('positive finite observed available memory required')
    required = math.ceil(3*base)+8*GIB
    available = min(budget, int(observed))
    if required > available:
        raise MemoryError(f'calibrated host guard needs {required/GIB:.2f} GiB; '
                          f'budget/available {available/GIB:.2f} GiB')
    return dict(policy='q08-scientific-host-3x-plus-8gib-v1',
                array_census_bytes=int(base), required_bytes=required,
                budget_bytes=budget, available_bytes=available,
                calibration='N32/N48/N64 A100 replay and six-field MMS, 3 October 2026',
                scope='empirical admission guard; not a hard RSS upper bound')
