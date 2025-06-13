"""
Measure correlation function of
2nd neighbor (diagonal bonds) on square lattice
"""

import argparse
import os
from glob import glob
import vumps.files_par   as ffile
from vumps.fixed_point2 import cal_fGL2s, cal_fGR2s
from vumps.cal_rhoss import cal_rhodss, cal_rho4ss


def get_args():
    parser = argparse.ArgumentParser(description="VUMPS measurement on 2nd neighbor bonds")
    parser.add_argument("tps_type", type=str, 
        help="Unit cell type (AB or MN)")
    parser.add_argument("N1", type=int, 
        help="Number of rows in the unit cell")
    parser.add_argument("N2", type=int, 
        help="Number of columns in the unit cell")
    parser.add_argument("Dcut", type=int, 
        help="Dimension of the boundary MPS")
    parser.add_argument("--tps_dir", type=str,
        help="Folder containing the TPS")
    parser.add_argument("--rhod", action="store_true",
        help="Only calculate 2-site (diagonal) rho. Otherwise, calculate 2x2 rho")
    parser.add_argument("--plot_fig", action="store_true",
        help="Show the plot of measurement results")
    parser.add_argument("--noskip", action="store_true",
        help="Do not skip fixed point and rho calculation even if previously calculated result is detected.")
    args = parser.parse_args()
    return args


def vumps_fixedpoint2(
    N1: int, N2: int, tps_dir: str, vumps_dir: str, 
    tps_type="MN"
): 
    """
    Main program of calculating the 2-row fixed points
    """
    # load PEPS
    fG0ss, fG1ss = ffile.load_peps(N1, N2, tps_dir, tps_type)
    # load boundary MPS
    assert vumps_dir.endswith(os.sep)
    # only AL tensors are needed
    fUss = ffile.load_bMPScanon(N1, N2, vumps_dir + "up/")[1]
    fDss = ffile.load_bMPScanon(N1, N2, vumps_dir + "down/")[1]
    precision = 1e-6
    # calculate 2-row fixed point
    fGL2ss = [None] * N1
    fGR2ss = [None] * N1
    print(f"--- Calculating left 2-row fixed point ---")
    for i in range(N1):
        fGL2ss[i] = cal_fGL2s(fUss, fDss, fG1ss, fG0ss, i, precision)
    print(f"--- Calculating right 2-row fixed point ---")
    for i in range(N1):
        fGR2ss[i] = cal_fGR2s(fUss, fDss, fG1ss, fG0ss, i, precision)
    ffile.save_fixedpoint2(
        fGL2ss, fGR2ss, vumps_dir + "fixed_point2/"
    )


if __name__ == "__main__":
    args = get_args()
    N1: int = args.N1
    N2: int = args.N2
    chi: int = args.Dcut
    tps_type: str = args.tps_type
    assert tps_type in ("AB", "MN")
    
    tps_dir: str = args.tps_dir
    vumps_dir = tps_dir + f"contract-{chi}/"
    rhoss_dir = tps_dir + ("rhodss" if args.rhod else "rho4ss") + f"-{chi}/"
    fp_dir = vumps_dir + "fixed_point2/"
    noskip: bool = args.noskip
    # if rho is already calculated, directly skip to measurement
    if (
        os.path.exists(rhoss_dir) and 
        len(glob(rhoss_dir + "*.npz")) == N1*N2*(2 if args.rhod else 1)
    ) and (noskip is False):
        print("---- Skip fixed point and rho calculation ----")
    else:
        # calculate left/right fixed point 
        # if it has not been calculated yet
        if (
            os.path.exists(fp_dir) and len(glob(fp_dir + "*.npz")) == 2*N1*N2
        ) and (noskip is False):
            print("---- Skip 2-row fixed point calculation ----")
        else:
            vumps_fixedpoint2(N1, N2, tps_dir, vumps_dir, tps_type)
        if args.rhod:
            # diagonal (2nd neighbor) density operators
            cal_rhodss(N1, N2, tps_dir, vumps_dir, rhoss_dir, tps_type)
        else:
            # 2x2 site density operators
            cal_rho4ss(N1, N2, tps_dir, vumps_dir, rhoss_dir, tps_type)
    