import numpy as np
from ..operators.nodal.nodal_layout_specs import build_zernike_core_ring_layout


def build(N, layout_builder=build_zernike_core_ring_layout):
    if layout_builder is None:
        layout_builder = build_zernike_core_ring_layout
    l = layout_builder(N, n_eta=N)
    core = l.blocks[0]
    # Core quadrature is Gauss in s=r^2. Cumulative weights give a positive
    # dual annulus per Gaussian radius, containing its node, with area=wxy.
    nt = len(core.u) // len(core.radii)
    areas = core.wxy.reshape(-1, nt).sum(axis=1)
    rb = np.sqrt(np.r_[0., np.cumsum(areas)] / np.pi)
    rb[-1] = core.R_c
    cells = []
    for i in range(core.n_nodes):
        r = i // nt
        cells.append((rb[r], rb[r+1], core.theta[i]-np.pi/nt, core.theta[i]+np.pi/nt))
    for b in l.blocks[1:]:
        for i in range(b.n_nodes):
            r = b.i0 + i // b.N
            cells.append((r/l.n, (r+1)/l.n, b.theta[i]-np.pi/b.N, b.theta[i]+np.pi/b.N))
    cells = np.asarray(cells)
    area = (cells[:,1]**2-cells[:,0]**2)*(cells[:,3]-cells[:,2])/2
    polar_measure = (cells[:,1]-cells[:,0])*(cells[:,3]-cells[:,2])
    assert np.all((l.node_u >= cells[:,0]) & (l.node_u <= cells[:,1]))
    assert np.all((l.node_theta >= cells[:,2]) & (l.node_theta <= cells[:,3]))
    # Split the core/ring interface at the union of the two angular meshes.
    # This preserves identical geometric edges on both sides of that seam.
    angles = np.unique(np.round(np.r_[cells[:core.n_nodes,2] % (2*np.pi), cells[core.n_nodes:,2] % (2*np.pi)],14))
    corners, edge_midpoints = [], []
    for r0,r1,t0,t1 in cells:
        for r in [r0,r1]:
            aa = [t0,t1]
            if abs(r-core.R_c) < 1e-14:
                aa += [float(t) for t in np.r_[angles-2*np.pi,angles,angles+2*np.pi] if t0 < t < t1]
            aa = np.unique(np.round(aa,14))
            corners.extend((r,t) for t in aa)
            edge_midpoints.extend((r,(ta+tb)/2) for ta,tb in zip(aa[:-1],aa[1:]))
        edge_midpoints.extend(((r0+r1)/2,t) for t in [t0,t1])
    def canonical(points):
        return sorted(set((round(r,14),0. if r == 0 else round(t%(2*np.pi),13)) for r,t in points))
    vertices = np.array(canonical(corners))
    midpoint = np.array(canonical(edge_midpoints))
    seeds = np.array(canonical(list(vertices)+list(midpoint)))
    return l, cells, area, polar_measure, vertices, midpoint, seeds
