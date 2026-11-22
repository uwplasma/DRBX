"""bounded indexed parallel assembly 64-key fit tasks and the gathered-width compatibility bank."""
import numpy as np

from .job_storage import mapped, read
from .weights import WallExpandedBlockFit
from ...operators.parallel.weights import FitTable, POWERS


def fit_batch(folder, layout, options, keys, trace):
    allowed = np.broadcast_to(np.ones(layout.P, bool), (layout.symmetry_period_planes, layout.P))
    owners = [(int(k) // layout.P, int(k) % layout.P) for k in keys[trace[keys]]]
    rec = WallExpandedBlockFit(layout, allowed, options.support_widths, options.expand_widths, trace_owners=owners)
    table = FitTable(rec, keys)
    for name in ('keys', 'ids', 'ncol', 'C', 'Vt', 'center'):
        np.save(folder / (name + '.npy'), getattr(table, name), allow_pickle=False)
    return dict(width=table.width, count=len(keys), links=int(table.ncol.sum()))


def fit_export(folder, batches, layout):
    width = max(item['width'] for _, item in batches)
    count = sum(item['count'] for _, item in batches)
    links = sum(item['links'] for _, item in batches)
    basis = len(POWERS)
    shapes = dict(keys=(count,), ids=(count, width), ncol=(count,), C=(count, basis, width-1),
                  Vt=(count, width-1, basis), center=(count, 2), index=(layout.period_nodes,),
                  indptr=(count+1,), indices=(links,))
    data = {key: mapped(folder / (key + '.npy'), shape,
                        np.float64 if key in ('C', 'Vt', 'center') else np.int64)
            for key, shape in shapes.items()}
    data['index'][:] = -1
    data['indptr'][0] = 0
    offset = link = 0
    for path, item in batches:
        n = item['count']
        local_width = item['width']
        rows = slice(offset, offset+n)
        keys = read(path / 'keys.npy')
        for key in ('keys', 'ncol', 'center'):
            data[key][rows] = read(path / (key + '.npy'))
        data['ids'][rows, :local_width] = read(path / 'ids.npy')
        data['C'][rows, :, :local_width-1] = read(path / 'C.npy')
        data['Vt'][rows, :local_width-1] = read(path / 'Vt.npy')
        data['index'][keys] = np.arange(offset, offset+n)
        for j, ncol in enumerate(data['ncol'][rows], offset):
            data['indices'][link:link+ncol] = data['ids'][j, :ncol]
            link += int(ncol)
            data['indptr'][j+1] = link
        offset += n
    return dict(width=width)


class MappedFits(FitTable):
    """The placed evaluator reads only its indexed fit rows from disk."""
    def __init__(self, layout, folder):
        self.folder = folder
        self.layout, self.N, self.P = layout, layout.N, layout.P
        for name in ('keys', 'ids', 'ncol', 'C', 'Vt', 'center', 'index'):
            setattr(self, name, read(folder / (name + '.npy')))
        self.width = self.ids.shape[1]
        self.poly_error = 0.

