"""Exclusive wall-clock phases and deduplicated per-record shared timings."""
import math
import time


class PhaseLedger:
    def __init__(self, clock=time.perf_counter):
        self.clock = clock
        self.start = self.last = clock()
        self.phases = {}

    def lap(self, name):
        now = self.clock()
        delta = now - self.last
        self.phases[name] = self.phases.get(name, 0.) + delta
        self.last = now
        return delta

    def finish(self):
        self.lap('finalize')
        elapsed = self.last - self.start
        return dict(elapsed_seconds=elapsed, phases_seconds=self.phases.copy(),
            unaccounted_seconds=elapsed-sum(self.phases.values()),
            scope='This invocation only, through receipt construction; excludes final summary writes and independent completion validation. Phases are exclusive, never sum nested per-record timings into this total.')


def record_totals(records):
    """The one/four GPU records share one BC and CPU reference evaluation."""
    shared = {}
    for r in records:
        key = r['shared_work_id']
        value = (r['boundary_host_seconds'], r['cpu_reference_seconds'])
        if any(not math.isfinite(v) or v < 0 for v in value):
            raise ValueError('invalid shared timing')
        if key in shared and shared[key] != value:
            raise ValueError('inconsistent shared-work timings')
        shared[key] = value
    return dict(shared_evaluations=len(shared),
        boundary_host_seconds=sum(v[0] for v in shared.values()),
        cpu_reference_seconds=sum(v[1] for v in shared.values()),
        first_synchronized_seconds=sum(r['first_synchronized_seconds'] for r in records),
        warm_seconds=sum(sum(r['warm_seconds']) for r in records),
        case_transfer_seconds=sum(r['case_transfer_seconds'] for r in records),
        scope='Persisted record timings, possibly across restarts; shared evaluations counted once. Not a complete wall-clock ledger.')
