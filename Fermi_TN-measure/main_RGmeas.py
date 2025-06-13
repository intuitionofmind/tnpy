"""
Measure a 2D TPS on square lattice
using tensor network renormalization

Tensor names on the square loop (loop convention)
```
      ↑   ↓
    → W ← Z →
      ↓   ↑
    ← X → Y ←
      ↑   ↓
```

On PEPS with 2 x 2 unit cells, 
there are 4 types of square loops
```
    A ---x1---- B ---x2_--- A
    |           |           |
    y1_   4     y2_   2     y1_
    |           |           |
    D ---x2---- C ---x1_--- D
    |           |           |
    y2    1     y1    3     y2
    |           |           |
    A ---x1---- B ---x2_--- A
```

On PEPS with only AB,
there are 2 types of square loops
(1 = 2 and 3 = 4)
``
    A ---x1---- B ---x2---- A
    |           |           |
    y1    3     y2    1     y1
    |           |           |
    B ---x2---- A ---x1---- B
    |           |           |
    y2    1     y1    3     y2
    |           |           |
    A ---x1---- B ---x2---- A
```
"""

import os
import argparse
from time import time
import logs
import gtensor as gt
from gtensor import GTensor
from utils import split_measkey
from phys_models.onesiteop import makeops, get_tJconv
from rg_fermion.rgstep import rgstep, rgstep_impure, normalize
from rg_fermion.coarsegrain import coarsegrain
from rg_fermion.double_tensor import build_dbts, build_dbts2
from rg_fermion.meas_utils import gen_meas_file
from update_ftps.sutools import absorb_wts
from update_ftps.tps_io import load_tps

def get_args():
    parser_model = argparse.ArgumentParser(add_help=False)
    parser_model.add_argument("model", type=str, 
        help="Statistical model")
    parser_model.add_argument("in_tps", type=str, 
        help="Path to the folder containing the TPS. Must end with \"/\".")
    # RG and loop optimization options
    parser_model.add_argument("--dcut", type=int, default=32, 
        help="Maximal bond dimension (even+odd) for RG; default 32.")
    parser_model.add_argument("--rgstep", type=int, default=10, 
        help="Number of RG steps; default 10.")
    parser_model.add_argument("--flt", action="store_true", 
        help="Turn on entanglement filtering.")
    parser_model.add_argument("--maxloop", type=int, default=0, 
        help="Number of loop optimization sweep rounds; default 0 (turned off).")

    parser = argparse.ArgumentParser(
        description="Measure physical quantities on 2D GTPS by GTNR")
    subparsers = parser.add_subparsers(dest="command")
    
    parser_unif = subparsers.add_parser(
        "unif", parents=[parser_model], help="calculate uniform part RG")
    parser_unif.add_argument("--plq", type=int, required=True, 
        help="(Required) Loop ID (1 to 4 for ABCD network; 1 and 3 for ABAB network) to be calculated.")
    parser_unif.add_argument("--save_unif", action="store_true", 
        help="Save uniform part RG tensors and filtering projectors after each RG step (before normalization).")

    parser_meas = subparsers.add_parser(
        "meas", parents=[parser_model], help="measure with RG")
    parser_meas.add_argument("measkey", type=str, 
        help="Two-site term to be measured.")
    parser_meas.add_argument("--usegate", action="store_true", 
        help="Use 2-site gate to get double tensors")
    parser_meas.add_argument("--load_unif", action="store_true", 
        help="Load previously calculated tensors of the uniform part. If not, the uniform tensors will be calculated altogether.")
    parser_meas.add_argument("--save_unif", action="store_true", 
        help="Save uniform part RG tensors and filtering projectors after each RG step (before normalization).")
    parser_meas.add_argument("--save_xyzw", action="store_true", 
        help="Save measured part RG tensors after each RG step (before normalization).")
    args = parser.parse_args()
    return args

def zero_check(t: GTensor):
    """Check if a GTensor is close to 0"""
    return gt.allclose(t, gt.zeros_like(t))

def save_rgstep(
    folder: str, step: int, Pas: None | list[GTensor], 
    Pbs: None | list[GTensor], Sapp: list[GTensor]
):
    """
    Save RG results (projectors and octagon tensors) 
    from uniform part of the network
    """
    assert folder.endswith(os.sep)
    try:
        assert len(Pas) == len(Pbs) == 4
        for i, (Pa, Pb) in enumerate(zip(Pas, Pbs)):
            gt.save(folder + f"projs/Pa-{step}-{i}.npz", Pa)
            gt.save(folder + f"projs/Pb-{step}-{i}.npz", Pb)
    except TypeError:
        assert Pas is None and Pbs is None
    assert len(Sapp) == 8
    for i, s in enumerate(Sapp):
        gt.save(folder + f"sapp/S-{step}-{i}.npz", s)

def load_rgstep(folder: str, step: int):
    """
    Load RG results from uniform part of the network
    """
    assert folder.endswith(os.sep)
    try:
        Pas = [gt.load(folder + f"projs/Pa-{step}-{i}.npz") for i in range(4)]
        Pbs = [gt.load(folder + f"projs/Pb-{step}-{i}.npz") for i in range(4)]
    except:
        Pas, Pbs = None, None
    Sapp = [gt.load(folder + f"sapp/S-{step}-{i}.npz") for i in range(8)]
    return Pas, Pbs, Sapp

if __name__ == "__main__":
    args = get_args()
    command = args.command
    assert command in ("unif", "meas")
    if command == "meas":
        if args.save_unif: assert args.load_unif is False 
    sites = ("x", "y", "z", "w")
    d_cutRG = args.dcut
    model = args.model
    # RG parameters
    param = {
        "model": model, "d_cutRG": d_cutRG, 
        "maxloop": args.maxloop, "flt": args.flt, 
        "maxRGstep": args.rgstep
    }

    parent_dir: str = args.in_tps
    assert parent_dir.endswith(os.sep)
    if command == "meas":
        measkey = args.measkey
    workdir = parent_dir + "flt-{}_loop-{}_dRG-{}/".format(
        param["flt"], param["maxloop"], param["d_cutRG"],
    )
    os.makedirs(workdir, exist_ok=True)
    if command == "meas":
        os.makedirs(workdir + f"{measkey}/", exist_ok=True)

    # load TPS, absorb square root of weights 
    # and convert to loop convention
    tensors, weights = load_tps(parent_dir)
    tensors = absorb_wts(tensors, weights)
    if 'tJ' in model:
        tJ_conv = get_tJconv((tensors["Ta"].DE[0], tensors["Ta"].DO[0]))
    else:
        tJ_conv = None
    t4 = (len(tensors) == 4)
    # total virtual dimension of the TPS tensors
    d_vir = tensors["Ta"].DS[-1]
    param["Dvir"] = d_vir

    # analyze measkey
    if command == "meas":
        (site1, site2), name1, name2, plq = split_measkey(measkey)
        plq = int(plq)
        op1 = makeops(name1, model, tJ_conv=tJ_conv)
        op2 = makeops(name2, model, tJ_conv=tJ_conv)
    else:
        plq = int(args.plq)
    if t4:
        assert plq in (1,2,3,4)
    else:
        # in ABAB network, loops (2,4) are equivalent to (1,3) respectively
        assert plq in (1,3)
    
    # create folders to save RG tensors
    # projectors and octagon tensors on uniform part
    if args.save_unif:
        os.makedirs(workdir + f"uniform{plq}/", exist_ok=True)
        os.makedirs(workdir + f"uniform{plq}/projs/", exist_ok=True)
        os.makedirs(workdir + f"uniform{plq}/sapp/", exist_ok=True)
    # octagon tensors on the impurity center
    if command == "meas": 
        if args.save_xyzw: 
            os.makedirs(workdir + f"{measkey}/sapp_impure/", exist_ok=True)
    
    # save RG parameters to file
    logs.init_files(
        workdir + f"{measkey}/" if command == "meas"
        else workdir + f"uniform{plq}/"
    )
    for key in param:
        logs.error.write("{:16}  {}\n".format(key, param[key]))
    logs.error.write("\n")

    # path to previously calculated tensors (uniform part)
    if command == "meas":
        input_dir = (
            None if args.load_unif is False else
            workdir + f"uniform{plq}/"
        )
    else:
        input_dir = None
    
    # get tensors on the center square loop
    Ta, Tb, Tc, Td = [
        tensors[f"T{s}"] for s in (
            ("abcd" if t4 else "abab") if plq == 1 else
            ("cdab" if t4 else "abab") if plq == 2 else
            ("badc" if t4 else "baba") if plq == 3 else
            ("dcba" if t4 else "baba")
        )
    ]
    # change dual arrow to counter-clockwise on loop 3 and 4
    if plq in (3,4):
        # minus signs are absorbed into sub-lattice A
        Ta = Ta.flip_dual([1,2,3,4], minus=True)
        Tc = Tc.flip_dual([1,2,3,4], minus=True)
        Tb = Tb.flip_dual([1,2,3,4], minus=False)
        Td = Td.flip_dual([1,2,3,4], minus=False)
    if t4 is False:
        Tc, Td = None, None
    
    # double layer tensors
    # we do not save them, since they can be easily generated
    if command == "meas":
        if args.usegate is True:
            gate = gt.outer(op1, op2).transpose(0,2,1,3)
            dTx, dTy, dTz, dTw, dTa, dTb, dTc, dTd \
                = build_dbts2(gate, site1+site2, Ta, Tb, Tc, Td)
        else:
            ops = dict((site, (
                op1 if site == site1 else
                op2 if site == site2 else None
            )) for site in sites)
            dTx, dTy, dTz, dTw, dTa, dTb, dTc, dTd \
                = build_dbts(list(ops[site] for site in sites), Ta, Tb, Tc, Td)
    else:
        dTa, dTb, dTc, dTd = build_dbts([None]*4, Ta, Tb, Tc, Td)[4::]

    # RG process
    for i in range(1, param["maxRGstep"] + 1):
        timestart = time()
        print("---- RG step {} ----".format(i))
        logs.error.write("---- RG step {} ----\n".format(i))
        
        # check if any of the 4 impure tensors is already 0
        if command == "meas":
            if any(zero_check(t) for t in (dTx, dTy, dTz, dTw)):
                logs.error.write("One of Ta, Tb, Tc, Td is zero. No need to perform RG. \n")
                for iterRG2 in range(1, param["maxRGstep"] + 1):
                    logs.info.write("{:4d} {:23.16G} {:4d}\n".format(iterRG2, 0., 1))
                    logs.info.write("{:4d} {:23.16G} {:4d}\n".format(iterRG2, 1., 1))
                break
        
        # after one step of RG, 
        # there will be no C and D in the network
        if input_dir is None:
            dTa, dTb, Pas, Pbs, Sapp_u = rgstep(
                param, dTa, dTb, Tc = (dTc if i == 1 else None), 
                Td = (dTd if i == 1 else None), clockwise = False
            )
            if args.save_unif:
                save_rgstep(workdir + f"uniform{plq}/", i, Pas, Pbs, Sapp_u)
        else:
            Pas, Pbs, Sapp_u = load_rgstep(workdir + f"uniform{plq}/", i)
            # reconstruct dTa, dTb from Sapp_u
            dTa, dTb = coarsegrain(Sapp_u)
        
        if command == "meas":
            dTx, dTy, dTz, dTw, Sapp = rgstep_impure(
                param, dTx, dTy, dTz, dTw, 
                Pas, Pbs, Sapp_u, clockwise = False
            )
            # we save the octagon S tensors to save disk space
            # dT's can be reconstructed by `coarsegrain_impure`
            if args.save_xyzw:
                for n, s in enumerate(Sapp):
                    gt.save(workdir + f"{measkey}/sapp_impure/S-{i}-{n}.npz", s)
        
        # normalize
        if command == "meas":
            fac, sgn, dTx, dTy, dTz, dTw = normalize(dTx, dTy, dTz, dTw)
            logs.info.write("{:4d} {:23.16G} {:4d}\n".format(i, fac, sgn))
        fac, sgn, dTa, dTb = normalize(dTa, dTb)
        logs.info.write("{:4d} {:23.16G} {:4d}\n".format(i, fac, sgn))
        
        timeend = time()
        logs.error.write("Used time : {:.3f} s\n\n".format(timeend - timestart))

    logs.closeall()

    # generate measurement file
    if command == "meas":
        normfile = workdir + f"{measkey}/Tnorm.txt"
        gen_meas_file(normfile, maxRGstep=param["maxRGstep"])
    