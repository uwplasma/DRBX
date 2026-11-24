"""Per-plane dense requests in a spawned geometry lane (sharded geometry scheduling and canonical collection geompart.tasks.ctx)."""
from .rays import Rays


class LaneRays(Rays):
    traced = 0

    def read(self, seeds, delta, label=''):
        if not self.dbs:
            self.traced += len(seeds)
        return super().read(seeds, delta, label)


def geometry_rays(options, evaluator):
    return LaneRays(evaluator, options.trace_substeps)
