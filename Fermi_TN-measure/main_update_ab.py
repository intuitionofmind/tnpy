"""
Simple update of 2D fermion TPS on honeycomb lattice
"""

import os
import argparse
from pprint import pprint
from time import time
from datetime import timedelta
from utils import get_precision
import phys_models.init_gate as ig
from update_ftps.userparams import set_paramSU
from update_ftps_ab.update_tps import update_tps


def get_args():
    parser = argparse.ArgumentParser(
        description="Fermion TPS (PEPS) update", 
    )
    parser.add_argument("model", type=str, 
        help="Statistical model")
    parser.add_argument("lattice", type=str, 
        help="Bipartite lattice (honeycomb or square)")
    parser.add_argument("Dcut", type=int, 
        help="Maximal virtual bond dimension (total) of TPS")
    parser.add_argument("--De", type=int, 
        help="Initial bond dimension (even) of TPS. Only applies to random initialization.")
    parser.add_argument("--dualconv", type=str, default="sym",
        help="Dual convention of TPS (sym or atob).")
    # model parameters
    parser.add_argument("--name", "-n", nargs="+", type=str,
        help="Store parameter name")
    parser.add_argument("--value", "-v", nargs="+", type=float,
        help="Store parameter value (input using the same order as --name)")
    ## tJ model specific
    parser.add_argument("--tJ_conv", type=int, default=1,
        help="Choose tJ model convention.")
    # TPS parameters
    parser.add_argument("--seed", type=int,
        help="Seed for random TPS initialization")
    parser.add_argument("--in_tps", "-i", type=str,
        help="Folder containing initialization TPS")
    parser.add_argument("--cplxinit", action="store_true", 
        help="Use complex (instead of real) initialization the TPS.")
    # weight averaging parameters
    parser.add_argument("--avgwt_group", default="all", type=str,
        help="Weights to be averages. Accepted values: all (default), none")
    parser.add_argument("--avgwt_mode", default="par", type=str,
        help="Weight averaging method. Accepted values: par, max")
    # evolution parameters
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

# this script should be executed as main program
if __name__ == "__main__":
    args = get_args()
    # set environment variables
    tJ_conv: int = args.tJ_conv
    lattice: str = args.lattice
    assert lattice in ("honeycomb", "square")
    model: str = args.model
    even_only = (tJ_conv == 0)
    # time evolution parameters
    param = {
        "mode": "simple", "lattice": lattice, 
        "model": model, "tJ_convention": tJ_conv, 
        "evolStep": args.evolstep, "initdt": args.initdt,
        "avgwt_group": args.avgwt_group, 
        "avgwt_mode": args.avgwt_mode, 
        "cplx_init": args.cplxinit,
        "wterror": args.wterror, "min_weight": 1e-10,
    }

    # physical dimension (model dependent)
    if model == "tV":
        Dphy = (1, 1)
    elif model in ("tJ", "tJ2"):
        if tJ_conv == 0:    Dphy = (2, 0)
        elif tJ_conv == 1:  Dphy = (2, 1)
        elif tJ_conv == 2:  Dphy = (1, 2)
        elif tJ_conv == 3:  Dphy = (2, 2)
        else: raise ValueError("Unrecognized tJ convention")
    else:
        raise NotImplementedError
    param["Dphy"] = Dphy
    # virtual dimension
    Dcut, De = args.Dcut, args.De
    param["Dcut"] = Dcut
    param["De"] = De

    # get model parameters from input
    assert isinstance(args.task, str)
    task = args.task
    folder = f"update_ftps_ab/{args.savedir}/"
    if args.debug:
        folder += "debug/"
    folder += "{}-su{}_{}-D{}/".format(
        model, "hc" if lattice == "honeycomb" else "sq", task, Dcut
    )
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
    param["nbond"] = 3 if lattice == "honeycomb" else 4
    # set default parameters
    set_paramSU(param)

    # get the list of gates to be applied
    # in one round of update
    gate = ig.init_gate(param, expo=False)
    param["seed"] = args.seed

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
    tensors, wts = update_tps(
        folder, info, gate, 
        dualconv=args.dualconv, debug=debug, 
        input_tps=args.in_tps, save_init=args.saveinit, 
        save_evol=args.saveevol, **param
    )
    # local measure
    info.write("\n")
    if "tJ" in model:
        from localmeas_ab.tJ_localmeas import tJ_localmeas
        from localmeas_ab.tJ_measprocess import process_measure
        measures = tJ_localmeas(tensors, wts)
        results = process_measure(measures, param)
    elif model == "tV":
        from localmeas_ab.tV_localmeas import tV_localmeas
        from localmeas_ab.tV_measprocess import process_measure
        measures = tV_localmeas(tensors, wts)
        results = process_measure(measures, param)
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
