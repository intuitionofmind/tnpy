"""
Main program for Loop-GTNR calculation

Except CFT calculations from transfer matrix, 
we adopt the loop convention of tensors
```
        1           1    
        ↓           ↑
    2 ← A → 0   2 → B ← 0
        ↑           ↓
        3           3
```
"""

import os
import argparse
import datetime
from time import time
import logs
from gtensor import save
from utils import get_precision

from phys_models.init_gate import init_gate
from rg_fermion import norm
from rg_fermion.userparams import set_paramRG
from rg_fermion.shrink import shrink_all as shrink4
from rg_fermion import cft
from rg_fermion.rgstep import rgstep, normalize

def get_args():
    parser = argparse.ArgumentParser(description="Start main program for Loop-GTNR")
    parser.add_argument("model", type=str, 
        help="Statistical model")
    parser.add_argument("dcut", type=int, 
        help="Maximal bond dimension for RG")
    # adjustable parameter
    parser.add_argument("--name", "-n", nargs="+", type=str,
        help="Store parameter name")
    parser.add_argument("--value", "-v", nargs="+", type=float,
        help="Store parameter value (input using the same order as --name)")
    parser.add_argument("--tJ_conv", type=str, default="1",
        help="t-J model physical convention")
    # options
    parser.add_argument("--flt", action="store_true",
        help="Turn on entanglement filtering")
    parser.add_argument("--maxloop", type=int, default=0,
        help="Maximum iteration of loop optimization")
    parser.add_argument("--maxRGstep", type=int, default=21,
        help="Maximum iteration of loop optimization")
    parser.add_argument("--nocft4", action="store_false",
        help="Skip calculation of cft4")
    parser.add_argument("--shrink1by1", action="store_true",
        help="Compress network layer by layer")
    parser.add_argument("--savetensor", action="store_true",
        help="Save tensors")
    parser.add_argument("--task", "-t", action="store",
        help="Specify a task name. If not set, date and time are used as task name.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    # calcft4 = args.nocft4
    model = args.model
    shrink1by1 = args.shrink1by1
    # convert input parameters to dictionary
    assert len(args.name) == len(args.value)
    input_param = {
        "d_cutRG": args.dcut, "shrink1by1": shrink1by1, 
        "calcft4": args.nocft4,
        "task": args.task, "savetensor": args.savetensor, 
        "flt": args.flt, "maxloop": args.maxloop, 
        "nev": 80, "maxRGstep": args.maxRGstep
    }
    if shrink1by1 is True:
        input_param["eps"] = 0.01

    # set parent directory
    if args.task is None:
        nowtime = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M")
        parent_dir = "rg_fermion/results/{}-{}/".format(model, nowtime)
    else:
        parent_dir = "rg_fermion/results/{}-{}/".format(model, args.task)
    os.makedirs(parent_dir, exist_ok=True)

    # set sub directory for current parameters
    subname = ""
    for name, val in zip(args.name, args.value):
        # different parameters are separated by "_"
        # since "-" will appear in negative parameters
        prec = get_precision(val)
        subname += f"{name}-{val:.{prec}f}_"
        input_param[name] = val
    subname = subname[:-1] + os.sep
    sub_dir = parent_dir + subname
    os.makedirs(sub_dir, exist_ok=True)

    # combine two dict of parameters
    param = set_paramRG(model, additions=input_param)

    # set directory to save tensors
    if args.savetensor is True:
        os.makedirs(sub_dir + "tensors/", exist_ok=True)

    # run main program
    calcft4 = param["calcft4"]
    savetensor = param["savetensor"]
    # for 1D system
    param["nbond"] = 2
    logs.init_files(sub_dir, init_cftfile=True, init_cft4=calcft4)
    for key, value in param.items():
        logs.error.write("{:<22s}{}\n".format(key, value))
    logs.error.write('\n')
    
    # initialize tensor network
    gate = init_gate(param)
    Ta, Tb = shrink4(gate, param)
    # save initial tensor after compress
    if savetensor is True:
        save(sub_dir + "tensors/Ta-{}.npz".format(0), Ta)
        save(sub_dir + "tensors/Tb-{}.npz".format(0), Tb)
    
    # coarse graining
    for i in range(1, param["maxRGstep"] + 1):
        timestart = time()
        print(i)
        logs.error.write("RG round {:d}\n".format(i))
        
        # Here we are still using the tensors from last (i - 1) RG step
        # Calculate when i - 1 = 2, 4, 6, ... 
        if i >= param["stepCFT"] and i % 2 == 1:
            # eigenvalues in vertical direction
            tmEigVal2V = cft.tm2_eigs(Ta, Tb, param["nev"], "v", "loop")
            # eigenvalues in horizontal direction (the correct ones)
            tmEigVal2H = cft.tm2_eigs(Ta, Tb, param["nev"], "h", "loop")
            cft.cal_cft(2, tmEigVal2V, tmEigVal2H, logs.cftFiles, i-1)
            print("scaling dimension (w = 2) extraction done")   

        # one RG step
        Ta, Tb = rgstep(param, Ta, Tb, clockwise=False)[0:2]
        print("coarsegrain done")

        # Here we are using the tensors from the current (i) step
        # to approximate CFT results at step (i - 1)
        # Calculate when i - 1 = 2, 4, 6, ... 
        if i >= param["stepCFT"] and i % 2 == 1:
            if calcft4 is True:
                tmEigVal4V = cft.tm4a_eigs(Ta, Tb, param["nev"], "v", "loop")
                tmEigVal4H = cft.tm4a_eigs(Ta, Tb, param["nev"], "h", "loop")
                cft.cal_cft(4, tmEigVal4V, tmEigVal4H, logs.cftFiles, i-1)
                print("scaling dimension (w = 4) extraction done")
        
        # rotate the network back to the original direction after two RG steps
        if i % 2 == 0:
            Ta, Tb = Tb.transpose(3,0,1,2), Ta.transpose(3,0,1,2)
            print("Rotated network back")

        # save NOT normalized tensor
        if savetensor is True:
            save(sub_dir + "tensors/Ta-{}.npz".format(i), Ta)
            save(sub_dir + "tensors/Tb-{}.npz".format(i), Tb)
        
        # normalize using the 2 x 2 norm
        fac, sgn, Ta, Tb = normalize(Ta, Tb)
        logs.info.write("{:<5d}{:<22.16G}{:4d}\n".format(i, fac, sgn))

        timeend = time()
        logs.error.write("Used time : {:.3f} s\n".format(timeend - timestart))
        # finish log of current step
        logs.error.write("\n")
    
    logs.closeall()
