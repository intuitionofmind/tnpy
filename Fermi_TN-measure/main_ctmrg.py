import os
import argparse
from copy import deepcopy
from ctmrg import ctm_io
from ctmrg import tps_io
from ctmrg import update
from ctmrg.measure import doublet
from ctmrg.examples import ising
import gtensor as gt
import torch
torch.set_default_dtype(torch.float64)
from itertools import product
from time import time
from datetime import timedelta


def get_args():
    parser = argparse.ArgumentParser(
        description="CTMRG program for fermionic 2D tensor network", 
    )
    parser.add_argument("N1", type=int, 
        help="Number of rows in the unit cell")
    parser.add_argument("N2", type=int, 
        help="Number of columns in the unit cell")
    parser.add_argument("Dcut", type=int, 
        help="Virtual bond dimension (total) of CTMs")
    parser.add_argument("tps_dir", type=str,
        help="Folder containing the TPS")
    parser.add_argument("--ctm_dir", type=str,
        help="Folder containing initialization of CTMs")
    parser.add_argument("--Dce", type=int, 
        help="Virtual bond dimension (even sector) of CTMs")
    parser.add_argument("--eps", type=float, default=5e-8, 
        help="singular value cutoff")
    parser.add_argument("--min_iter", type=int, default=25,
        help="Minimum number of CTMRG iterations")
    parser.add_argument("--max_iter", type=int, default=1000,
        help="Maximum number of CTMRG iterations")
    parser.add_argument("--diff_max", type=float, default=1e-12,
        help="Folder containing the TPS")
    parser.add_argument("--seed", type=int, 
        help="Random seed for CTM initialization")
    parser.add_argument("--cheap", action="store_true",
        help="Use cheap method to find projectors")
    parser.add_argument("--bipartite", action="store_true",
        help="The input iPEPS is bipartite")
    parser.add_argument("--check_int", type=int, default=5, 
        help="Step interval to check convergence")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    print()
    time_start = time()
    args = get_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
    chi: int = args.Dcut
    chie: int|None = args.Dce
    if chie is None:
        assert chi % 2 == 0
        chie = chi // 2
    cheap: bool = args.cheap
    bipartite: bool = args.bipartite
    eps: float = args.eps
    diff_max: float = args.diff_max
    min_iter: int = args.min_iter
    max_iter: int = args.max_iter
    assert max_iter >= min_iter >= 1
    check_int: int = args.check_int
    tps_dir: str = args.tps_dir
    ctm_dir0: str|None = args.ctm_dir
    N1: int = args.N1
    N2: int = args.N2
    assert tps_dir[-1] == "/"
    ts0, ts1 = tps_io.load_tps_wt(tps_dir, N1, N2)
    if bipartite:
        update.assert_bipartite(ts0)
    # combine T and T into M
    ms = [[
        doublet(ts0[y][x], ts1[y][x], None) for x in range(N2)
    ] for y in range(N1)]
    # normalize and round tensors
    for y, x in product(range(N1), range(N2)):
        ms[y][x] /= gt.maxabs(ms[y][x])

    # chi, chie = 8, 8
    # ms = ising.ising(torch.tensor(0.2, dtype=torch.float64))

    if ctm_dir0 is None:
        ctms = ctm_io.rand_ctm(ms, chi, chie, complex_init=False)
    else:
        ctms = ctm_io.load_ctm(ctm_dir0, N1, N2)
    print("{:<5s}{:>13s}{:>13s}{:>10s}".format(
        "iter", "diff", "diff2", "time/s"
    ))
    for count in range(max_iter):
        ctms0 = deepcopy(ctms)
        time0 = time()
        for direction in ("left", "right", "up", "down"):
            update.ctmrg_move(
                direction, ms, ctms, chi, chie, 
                eps=eps, cheap=cheap, bipartite=bipartite
            )
        time1 = time()
        try:
            diff = update.compare_ctms(ctms, ctms0)
        except ValueError:
            diff = float("nan")
        
        if count % check_int == 0:
            facs = update.converge_test(0, 0, ms, ctms)
            facs0 = update.converge_test(0, 0, ms, ctms0)
            diff2 = abs(facs[0] / facs0[0] - 1)
            print("{:<5d}{:>13.4e}{:>13.4e}{:>10.2f}".format(
                count, diff, diff2, time1 - time0
            ))
            if diff2 <= diff_max and count >= min_iter:
                break
    if bipartite:
        for cs in ctms: update.assert_bipartite(cs)
    ctm_dir = tps_dir + f"ctm-{chi}/"
    os.makedirs(ctm_dir, exist_ok=True)
    ctm_io.save_ctm(ctm_dir, ctms)
    time_end = time()
    print("\nTotal calculation time: {}\n".format(
        str(timedelta(seconds=time_end-time_start))
    ))
