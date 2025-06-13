"""
Approximate local measurement of
2D TPS on square lattice

benchmark: AFM Heisenberg model 
(with nearest neighbor J = 1)

- without shift by -1/4 n_i n_j
    energy per link: -0.3346
    per site: (per link) * 2 = -0.6692
- with shift by -1/4 n_i n_j
    energy per link: -0.3346 - J*0.25 = -0.5846
    per site: (per link) * 2 = -1.1692

benchmark: J1-J2 model 
(with J = 1, J2 = 0.5 and shifted by -1/4 n_i n_j)

- without shift by -1/4 n_i n_j
    energy per link: -0.2481
    per site: (per link) * 2 = -0.4962
- with shift by -1/4 n_i n_j
    energy per link: -0.2481 - (J*0.25 + J2*0.25) = -0.6231
    per site: (per link) * 2 = -1.2462

D = 7 iPEPS (PRB 96 014414)
- without shift by -1/4 n_i n_j
    energy per link: -0.2475
    per site: (per link) * 2 = -0.4950
- with shift by -1/4 n_i n_j
    energy per link: -0.2475 - (J*0.25 + J2*0.25) = -0.6225
    per site: (per link) * 2 = -1.2450
"""

import os
import argparse
from utils import dict_loadtxt
from pprint import pprint
import pickle
import numpy as np
from fermiT.conversion import ft2gt
from update_ftps.tps_io import load_tps
from update_ftps.convert_ft2gt import load_tps_FermiT


def get_args():
    parser = argparse.ArgumentParser(
        description="Approximate local measurement")
    # input options
    parser.add_argument("model", type=str, 
        help="Physical model (supported: tJ, tV).")
    parser.add_argument("folder", type=str, 
        help="Folder containing the tensors and weights")
    parser.add_argument("--nb2", action="store_true",
        help="Measure 2nd neighbor quantities")
    parser.add_argument("--meas_mode", type=str, default="bond",
        help="Measurement mode for 1st neighbor bonds (bond or loop)")
    parser.add_argument("--print", action="store_true", 
        help="Print measured terms.")
    parser.add_argument("--save", action="store_true", 
        help="Save measured terms and configuration plot.")
    parser.add_argument("--make_plot", action="store_true", 
        help="Plot configuration")
    args = parser.parse_args()
    return args


def main_localmeas(
    model: str, folder: str, nb2=True, meas_mode="bond"
):
    assert folder.endswith(os.sep)
    param = dict_loadtxt(folder + "tpsinfo.txt")
    assert meas_mode in ("bond", "loop")
    try:
        # native GTensor format
        tensors, weights = load_tps(folder)
    except KeyError:
        # FermiT format
        tensors, weights = load_tps_FermiT(folder)
        for key, val in tensors.items():
            tensors[key] = ft2gt(val.transpose(0,4,3,2,1))
        for key, val in weights.items():
            weights[key] = ft2gt(val)
    if model == "tJ":
        from localmeas.tJ_localmeas import tJ_localmeas
        from localmeas.tJ_measprocess import process_measure
        measures = tJ_localmeas(tensors, weights, nb2, meas_mode)
        results = process_measure(measures, param, nb2)
    elif model == "tV":
        from localmeas.tV_localmeas import tV_localmeas
        from localmeas.tV_measprocess import process_measure
        measures = tV_localmeas(tensors, weights, nb2, meas_mode)
        results = process_measure(measures, param, nb2)
        # use zero arrays as placeholder of magnetization
        n_sites = len(results["dope"])
        results.update({"mag": np.zeros((3,n_sites)), "mag_norm": np.zeros(n_sites)})
    else:
        raise ValueError("Unrecognized model.")
    results["param"] = param
    return results, measures


if __name__ == "__main__":
    args = get_args()
    model: str = args.model
    folder: str = args.folder
    make_plot: bool = args.make_plot
    nb2: bool = args.nb2
    meas_mode: str = args.meas_mode
    results, measures = main_localmeas(model, folder, nb2, meas_mode)
    if args.print:
        for key, val in results.items():
            if key == "param": continue
            print(key + "-" * 20)
            pprint(val, sort_dicts=False)
    if args.save:
        save_dir = folder + "measures_local0/"
        os.makedirs(save_dir, exist_ok=True)
        with open(save_dir + "results.pkl", "wb") as f:
            pickle.dump(results, f)
        with open(save_dir + "terms.pkl", "wb") as f:
            pickle.dump(measures, f)

    if make_plot:
        from localmeas.plot_config import plt, plot_config
        fig, ax = plot_config(**results)
        if args.save:
            fig.savefig(save_dir + "config_local.png", dpi=144)
            plt.close()
        else:
            plt.show()
