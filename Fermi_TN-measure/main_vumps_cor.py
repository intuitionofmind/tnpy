import os
import argparse
from time import time
import numpy as np
import vumps.correlation as cor
import vumps.files_par as ffile
from phys_models.onesiteop import get_tJconv
from phys_models.init_gate import makeops_tJft


def get_args():
    parser = argparse.ArgumentParser(description=(
        "VUMPS measurement on 2-site correlation function "
        + "<Oa_{i,0} Ob_{i,j}> and <Oa_{0,j} Ob_{i,j}>"
    ))
    parser.add_argument("-Oa", type=str, 
        help="Name of operator Oa")
    parser.add_argument("-Ob", type=str, 
        help="Name of operator Ob")
    parser.add_argument("-tps_type", type=str, 
        help="Unit cell type (AB or MN)")
    parser.add_argument("-N1", type=int, 
        help="Number of rows in the unit cell")
    parser.add_argument("-N2", type=int, 
        help="Number of columns in the unit cell")
    parser.add_argument("-Dcut", type=int, 
        help="Dimension of boundary MPS")
    parser.add_argument("--num", type=int, default=10,
        help="Max separation between two sites")
    parser.add_argument("--tps_dir", type=str,
        help="Folder containing the TPS")
    parser.add_argument("--print", action="store_true",
        help="Print measurement results")
    args = parser.parse_args()
    return args


def cal_cor(
    op1name: str, op2name: str, N1: int, N2: int, 
    tps_dir: str, vumps_dir: str, 
    tps_type = "MN", num = 20
):
    """
    Calculate 2-site correlation functions
    between any site Oa = (i,j) and 
    
    - Hotizontal: (i,j') with j' = j, j+1, j+num-1
    - Vertical:   (i',j) with i' = i, i+1, i+num-1
    """
    # load PEPS
    fG0ss, fG1ss = ffile.load_peps(N1, N2, tps_dir, tps_type)
    # load boundary MPS and fixed points
    fGXss = ffile.load_fixedpoint(N1, N2, vumps_dir + "fixed_point/")
    fXss = [
        # only AL tensors are needed
        ffile.load_bMPScanon(N1, N2, vumps_dir + f"{d}/")[1]
        for d in ("up", "down", "left", "right")
    ]
    Dps, Dpe = fG0ss[0][0].DS[0], fG0ss[0][0].DE[0]
    tJ_conv = get_tJconv((Dpe, Dps-Dpe))
    op1 = makeops_tJft(op1name, tJ_conv)
    op2 = makeops_tJft(op2name, tJ_conv)
    corH = cor.cor_2site_hor(op1, op2, fXss, fGXss, fG1ss, fG0ss, num=num)
    corV = cor.cor_2site_ver(op1, op2, fXss, fGXss, fG1ss, fG0ss, num=num)
    return corH, corV


if __name__ == "__main__":
    args = get_args()
    N1: int = args.N1
    N2: int = args.N2
    chi: int = args.Dcut
    tps_dir: str = args.tps_dir
    vumps_dir = tps_dir + f"contract-{chi}/"
    assert tps_dir[-1] == '/' and vumps_dir[-1] == '/'
    tps_type: str = args.tps_type
    assert tps_type in ("AB", "MN")

    cor_dir = tps_dir + f"corXY-{chi}/"
    os.makedirs(cor_dir, exist_ok=True)

    op1name: str = args.Oa
    op2name: str = args.Ob
    time0 = time()
    corH, corV = cal_cor(
        op1name, op2name, N1, N2, tps_dir, vumps_dir, 
        tps_type, num=20
    )
    if args.print:
        print(corH, corV, sep='\n')
    np.savez(cor_dir + f"{op1name}{op2name}.npz", corH=corH, corV=corV)
    print(f"Used time = {(time() - time0):.4f} s")
