"""
PEPS uses natural convention
(weight axis order: 0 left/down, 1 right/up)
```
            :       :
            ↑       ↑
            y10     y11
            ↑       ↑
    ..→ x →T10→x10→T11→x11→..
            ↑       ↑
            y00     y01
            ↑       ↑
    ..→ x →T00→x00→T01→x01→..
            ↑       ↑
            y       y
            ↑       ↑
            :       :
```

Tensor axis order
```
        2  0
        ↑ /
    3 → A → 1
        ↑
        4
```
"""

import os
import argparse
from pprint import pprint
from time import time
from datetime import timedelta
import torch
from utils import get_precision
from phys_models.init_gate import init_gate
from update_ftps.userparams import set_paramSU
from update_ftps_mn.update_tps import update_tps

torch.set_default_dtype(torch.float64)

def get_args():
    parser = argparse.ArgumentParser(
        description="Fermion TPS (PEPS) update with N1 x N2 unit cell", 
    )
    parser.add_argument("model", type=str, 
        help="Statistical model")
    parser.add_argument("N1", type=int, 
        help="Number of rows in the unit cell")
    parser.add_argument("N2", type=int, 
        help="Number of columns in the unit cell")
    parser.add_argument("Dcut", type=int, 
        help="Maximal virtual bond dimension (total) of TPS")
    parser.add_argument("--De", type=int, 
        help="Initial bond dimension (even) of TPS. Only applies to random initialization.")
    
    # model parameters
    parser.add_argument("--name", "-n", nargs="+", type=str,
        help="Store parameter name")
    parser.add_argument("--value", "-v", nargs="+", type=float,
        help="Store parameter value (input using the same order as --name)")
    ## tJ model specific
    parser.add_argument("--tJ_conv", type=int, default=1,
        help="Choose tJ model physical axis convention.")
    
    # initialization parameters
    parser.add_argument("--mode", type=str, default="simple",
        help="Choose update scheme. Default simple.")
    
    # TPS parameters
    parser.add_argument("--seed", type=int,
        help="Seed for random TPS initialization")
    parser.add_argument("--in_tps", "-i", type=str,
        help="Folder containing initialization TPS")
    parser.add_argument("--cplxinit", action="store_true", 
        help="Use complex (instead of real) initialization the TPS.")
    
    # evolution parameters
    parser.add_argument("--avgwt", action="store_true", 
        help="Average all weights after each round of update.")
    parser.add_argument("--avgwt_mode", default="par", type=str,
        help="Method of averaging all weights (par or max)")
    parser.add_argument("--evolstep", default=800000, type=int,
        help="Max update steps. Default 800000. Will be ignored if dts_mode is `step`")
    parser.add_argument("--initdt", default=0, type=int,
        help="Initial dt for time evolution. ")
    parser.add_argument("--wterror", default=1e-10, type=float,
        help="Stop evolution when weight difference between two steps becomes smaller than wterror.")

    # result saving parameters
    parser.add_argument("--task", "-t", default="test", type=str,
        help="Specify a task name. Default is \"test\".")
    parser.add_argument("--sampleid", default="0", type=str,
        help="Sample ID to distinguish runs of similar parameters.")
    parser.add_argument("--debug", action="store_true",
        help="Debug mode: print more info; initialization is not saved.")
    parser.add_argument("--overwrite", action="store_true", 
        help="Overwrite old log file.")
    parser.add_argument("--saveinit", action="store_true",
        help="Save initialization TPS. If --debug, this flag will be ignored. ")
    parser.add_argument("--saveevol", action="store_true",
        help="Save intermediate TPS during evolution")
    parser.add_argument("--savedir", default="results_2d/", type=str, 
        help="Root folder to save results. Default \"results_2d/\"")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    # set environment variables
    tJ_conv: int = args.tJ_conv
    # set unit cell size
    N1: int = args.N1
    N2: int = args.N2
    args = get_args()
    # set environment variables
    tJ_conv: int = args.tJ_conv
    model: str = args.model
    even_only = (tJ_conv == 0)
    # time evolution parameters
    param = {
        "N1": N1, "N2": N2, "nbond": 4,
        "model": model, "tJ_convention": tJ_conv, 
        "evolStep": args.evolstep, "initdt": args.initdt,
        "avgwt": args.avgwt, "avgwt_mode": args.avgwt_mode, 
        "cplx_init": args.cplxinit,
        "wterror": args.wterror, "min_weight": 1e-10,
    }
    # virtual dimension
    Dcut, De = args.Dcut, args.De
    param["Dcut"] = Dcut
    param["De"] = De

    # get model parameters from input
    assert isinstance(args.task, str)
    task = args.task
    folder = f"update_ftps_mn/{args.savedir}/"
    if args.debug:
        folder += "debug/"
    folder += f"{model}-su{N1}{N2}_{task}-D{Dcut}/"
    gate_param = {}
    for name, value in zip(args.name, args.value):
        gate_param[name] = value
        # get number of digits (assumed to be < 8) after decimal place
        precision = get_precision(value)
        folder += f"{name}-{value:.{precision}f}_"
    folder = folder[:-1] + os.sep
    os.makedirs(folder, exist_ok=True)
    folder += "conv{}-sample{}/".format(tJ_conv, args.sampleid)
    os.makedirs(folder, exist_ok=True)
    param.update(gate_param)
    # set default parameters
    set_paramSU(param)

    # get the list of gates to be applied
    # in one round of update
    gate = init_gate(param, expo=False)
    Dphy = (gate.DE[0], gate.DO[0])
    param["seed"] = args.seed
    param["Dphy"] = Dphy

    # save parameters to log file
    # mode x: file should not already exist when not debugging
    # to avoid accidentally overwriting old results
    overwrite: bool = args.overwrite
    debug: bool = args.debug
    info = open(
        folder + "tpsinfo.txt", buffering=1, 
        mode = ("w" if (overwrite or debug) else "x")
    )
    for key, value in param.items():
        info.write(f"{key:<22s}{value}\n")
    info.write("\n")

    time0 = time()
    ts, wxs, wys = update_tps(
        folder, info, gate, debug=debug, 
        input_tps=args.in_tps, save_init=args.saveinit, 
        save_evol=args.saveevol, **param
    )
    # local measure
    info.write("\n")
    if "tJ" in model:
        from localmeas_mn.tJ_localmeas import tJ_localmeas
        from localmeas_mn.tJ_measprocess import process_measure
        measures = tJ_localmeas(ts, wxs, wys)
        results = process_measure(measures, param, nb2=False)
    elif model == "tV":
        raise NotImplementedError
    else:
        raise ValueError("Unsupported model")
    for key, val in results.items():
        print(key + "-" * 20, file=info)
        pprint(val, stream=info, width=60, sort_dicts=False)
    time1 = time()
    info.write("\nTotal evolution time: {}\n".format(
        str(timedelta(seconds=time1-time0))
    ))
    info.close()
