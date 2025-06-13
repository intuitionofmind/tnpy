"""
Fast full update of 2D fermion TPS on bipartite square lattice
(Physical Review B 92, 035142)
"""

from itertools import product
import gtensor as gt
from phys_models.init_gate import init_gate
import ctmrg.ctm_io as cio
import ctmrg.tps_io as tio
from ctmrg import update
from ctmrg.measure import doublet
import ctmrg.update_full as ffu
from copy import deepcopy


# load initialization
tps_dir = "update_ftps/results_2d/tJ-su_dwave-D4/mu-4.20/conv1-sample0/"
ctm_dir = tps_dir + "ctm-16/"
N1, N2 = 2, 2
bipartite = True
# load iPEPS
ts, ts1 = tio.load_tps_wt(tps_dir, N1, N2, normalize=True)
ms = [[
    doublet(ts[y][x], ts1[y][x], None) for x in range(N2)
] for y in range(N1)]
# normalize and round tensors
for y, x in product(range(N1), range(N2)):
    ms[y][x] /= gt.maxabs(ms[y][x])
# load CTMRG
ctms = cio.load_ctm(ctm_dir, N1, N2)
# check bipartiion
if bipartite:
    update.assert_bipartite(ts)
    update.assert_bipartite(ms)
    for cs in ctms: update.assert_bipartite(cs)

param = dict(
    model = "tJ", t = 3, J = 1, mu = 4.2, 
    tJ_convention = 1, nbond = 4, dt = 1e-3,
    eps = 5e-8, Dmax = 4, bipartite = True, cheap=True
)
gate = init_gate(param, expo=True)
evolstep = 100
for count in range(evolstep):
    ts0 = deepcopy(ts)
    # update all columns
    for x in range(N2):
        ffu.update_column(x, gate, ts, ms, ctms, **param)
        ffu.leftright_move(x, ms, ctms, **param)
        if bipartite:
            # no need to update x = 1 from scratch
            for y in range(N1):
                ts[y-1][x-1] = ts[y][x]
                ms[y-1][x-1] = ms[y][x]
                for n, cs in enumerate(ctms):
                    ctms[n][y-1][x-1] = cs[y][x]
            break
    # update all rows
    for y in range(N1):
        ffu.update_row(y, gate, ts, ms, ctms, **param)
        ffu.updown_move(y, ms, ctms, **param)
        if bipartite:
            # no need to update y = 1 from scratch
            for x in range(N2):
                ts[y-1][x-1] = ts[y][x]
                ms[y-1][x-1] = ms[y][x]
                for n, cs in enumerate(ctms):
                    ctms[n][y-1][x-1] = cs[y][x]
            break
    diff = sum(
        gt.norm(ts[y][x] - ts0[y][x])
        for y, x in product(range(N1), range(N2))
    )
    print("---- iPEPS tensor diff: {:.4e} ----".format(diff))
    if bipartite:
        update.assert_bipartite(ts)
        update.assert_bipartite(ms)
        for cs in ctms: update.assert_bipartite(cs)

