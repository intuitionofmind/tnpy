import os
import argparse
import numpy as np
from glob import glob
import vumps.boundary    as fbond
import vumps.fixed_point as ffp
import vumps.files_par   as ffile
from vumps.cal_rhoss import cal_rhoss


def get_args():
    parser = argparse.ArgumentParser(
        description="VUMPS algorithm and measurement.", 
    )
    subparsers = parser.add_subparsers(dest="command")
    # options used by both bMPS calculation and measurement
    parser_ucell = argparse.ArgumentParser(add_help=False)
    parser_ucell.add_argument("-tps_type", type=str, 
        help="Unit cell type (AB or MN)")
    parser_ucell.add_argument("-N1", type=int, 
        help="Number of rows in the unit cell")
    parser_ucell.add_argument("-N2", type=int, 
        help="Number of columns in the unit cell")
    parser_ucell.add_argument("-Dcut", type=int, 
        help="Virtual bond dimension (total) of boundary MPS")
    parser_ucell.add_argument("--Dce", type=int, 
        help="Virtual bond dimension (even sector) of boundary MPS")
    parser_ucell.add_argument("--tps_dir", type=str,
        help="Folder containing the TPS")
    # boundary MPS calculation
    parser_calc = subparsers.add_parser(
        "calc", parents=[parser_ucell], help="calculate boundary MPS")
    parser_calc.add_argument("--seed", type=int, 
        help="Random seed for initialization")
    parser_calc.add_argument("--smallD_dir", type=str,
        help="Folder containing the boundary MPS with smaller dcut as initialization")
    parser_calc.add_argument("--direction", type=str, default="all",
        help="Boundary MPSs to be calculated (all, ud, lr)")
    parser_calc.add_argument("--iternum1", type=int, default=30, 
        help="max iternum from row i -> i + 1 -> i + 2 -> ...  (default 30)")
    parser_calc.add_argument("--iternum2", type=int, default=1, 
        help="max iternum within i -> i + 1  (default 1)")
    # measurement (calculate rho)
    parser_meas = subparsers.add_parser(
        "meas", parents=[parser_ucell], help="measure boundary MPS")
    parser_meas.add_argument("--noskip", action="store_true",
        help="Do not skip fixed point and rho calculation even if previously calculated result is detected.")
    args = parser.parse_args()
    return args


def vumps_contract(
    tps_type: str, N1: int, N2: int, direction: str,
    Dcut: int, Dce: int, tps_dir: str, 
    iternum0 = 6, iternum1 = 30, iternum2 = 1, seed=None, 
    smallD_dir: None|str = None
):
    """
    VUMPS algorithm: obtain uniform boundary MPSs

    Parameters
    ----
    N1, N2: int
        unit cell size (number of rows and columns)
    Dcut, Dce: int
        total/even bond dimension for VUMPS
    tps_dir: str
        Folder containing the fermion PEPS
    save_dir: str
        Folder to save VUMPS tensors
    seed: None or int
        Random seed for VUMPS algorithm
    """
    # load PEPS
    assert tps_type in ("AB", "MN")
    if tps_type == "AB":
        assert N1 == N2 == 2
    fG0ss, fG1ss = ffile.load_peps(N1, N2, tps_dir, tps_type)
    if smallD_dir is None:
        fAss = None
    else:
        assert smallD_dir.endswith(os.sep)
        fAss = ffile.load_bMPScanon(N1, N2, smallD_dir + f"{direction}/")
    vumps_params = dict(
        tps_type=tps_type, iternum0=iternum0, 
        iternum1=iternum1, iternum2=iternum2, 
        tolerance=1e-5, icheck=2
    )
    # use VUMPS to obtain the up MPS
    if seed is not None:
        np.random.seed(seed)
    print(f"------ Calculating {direction} MPS ------")
    fCss, fALss, fARss, info = fbond.fixed_boundary(
        Dcut, Dce, direction, fG1ss, fG0ss, 
        fAss=fAss, **vumps_params
    )
    save_dir = tps_dir + f"contract-{Dcut}/"
    ffile.save_bMPScanon([fCss, fALss, fARss], save_dir + f"{direction}/")


def vumps_fixedpoint(
    N1: int, N2: int, tps_dir: str, vumps_dir: str, 
): 
    """
    Main program of calculating the fixed points
    """
    # load PEPS
    try:
        fG0ss, fG1ss = ffile.load_peps(N1, N2, tps_dir, "MN")
    except AssertionError:
        fG0ss, fG1ss = ffile.load_peps(N1, N2, tps_dir, "AB")
    assert vumps_dir.endswith(os.sep)
    fRss = ffile.load_bMPScanon(N1, N2, vumps_dir + "right/")
    fLss = ffile.load_bMPScanon(N1, N2, vumps_dir + "left/")
    fUss = ffile.load_bMPScanon(N1, N2, vumps_dir + "up/")
    fDss = ffile.load_bMPScanon(N1, N2, vumps_dir + "down/")
    precision = 1e-5 # for fixed point
    # only AL tensors are needed (use left canonical form) 
    # to calculate fixed point
    print(f"------ Calculating up fixed point ------")
    fGUss = ffp.fixed_point(
        fRss[1], fLss[1], fG1ss, fG0ss, "up",   precision=precision)
    print(f"------ Calculating down fixed point ------")
    fGDss = ffp.fixed_point(
        fRss[1], fLss[1], fG1ss, fG0ss, "down", precision=precision)
    print(f"------ Calculating left fixed point ------")
    fGLss = ffp.fixed_point(
        fUss[1], fDss[1], fG1ss, fG0ss, "left",  precision=precision)
    print(f"------ Calculating right fixed point ------")
    fGRss = ffp.fixed_point(
        fUss[1], fDss[1], fG1ss, fG0ss, "right", precision=precision)
    ffile.save_fixedpoint(
        [fGUss, fGDss, fGLss, fGRss], vumps_dir + "fixed_point/")


if __name__ == "__main__":
    args = get_args()
    tps_type: str = args.tps_type
    assert tps_type in ("AB", "MN")
    N1: int = args.N1
    N2: int = args.N2
    if tps_type == "AB": assert (N1, N2) == (2,2)
    Dcut: int = args.Dcut
    Dce: int | None = args.Dce
    if Dce is None: Dce = Dcut // 2
    tps_dir: str = args.tps_dir
    assert tps_dir.endswith(os.sep)

    if args.command == "calc":
        vumps_contract(
            tps_type, N1, N2, args.direction, Dcut, Dce, 
            tps_dir, iternum0=6, 
            iternum1=args.iternum1, iternum2=args.iternum2, 
            seed=args.seed, smallD_dir=args.smallD_dir, 
        )
    elif args.command == "meas":
        vumps_dir = tps_dir + f"contract-{Dcut}/"
        fp_dir = vumps_dir + "fixed_point/"
        rhoss_dir = tps_dir + f"rhoss-{Dcut}/"
        noskip: bool = args.noskip
        # if rho is already calculated, directly skip to measurement
        if (
            os.path.exists(rhoss_dir) and 
            len(glob(rhoss_dir + "*.npz")) == 4*N1*N2
        ) and (noskip is False):
            print("---- Skip fixed point and rho calculation ----")
        else:
            # calculate fixed point if it has not been calculated yet
            if (
                os.path.exists(fp_dir) and 
                len(glob(fp_dir + "*.npz")) == 4*N1*N2
            ) and (noskip is False):
                print("---- Skip fixed point calculation ----")
            else:
                vumps_fixedpoint(N1, N2, tps_dir, vumps_dir)
            # calculate density operators
            cal_rhoss(N1, N2, tps_dir, vumps_dir, rhoss_dir, tps_type)
    else:
        raise ValueError("Unrecognized command argument.")
    
