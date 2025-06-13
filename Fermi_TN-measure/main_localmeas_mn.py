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
"""

import os
import argparse
from utils import dict_loadtxt
from pprint import pprint
import pickle
from update_ftps_mn.tps_io import load_tps

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
    model: str, folder: str, nb2: bool, meas_mode="bond"
):
    assert folder.endswith(os.sep)
    param = dict_loadtxt(folder + "tpsinfo.txt")
    assert meas_mode in ("bond", "loop")
    try:
        N1: int = param["N1"]
        N2: int = param["N2"]
    except KeyError:
        N1, N2 = 2, 2
    ts, wxs, wys = load_tps(folder, N1, N2)
    if model == "tJ":
        from localmeas_mn.tJ_localmeas import tJ_localmeas
        from localmeas_mn.tJ_measprocess import process_measure
        measures = tJ_localmeas(ts, wxs, wys, nb2, meas_mode)
        results = process_measure(measures, param, nb2)
    elif model == "tV":
        raise NotImplementedError
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
        save_dir = folder + "measures_local/"
        os.makedirs(save_dir, exist_ok=True)
        with open(save_dir + "results.pkl", "wb") as f:
            pickle.dump(results, f)
        with open(save_dir + "terms.pkl", "wb") as f:
            pickle.dump(measures, f)

    if make_plot:
        from plottools.imshow import plt, imshow_config
        param = dict_loadtxt(folder + "tpsinfo.txt")
        try:
            N1: int = param["N1"]
            N2: int = param["N2"]
        except KeyError:
            N1, N2 = 2, 2
        fig, ax = imshow_config(
            N1, N2, results["e_site"], 
            results["dope"], results["mag"], 
            results["scorder"][1], results["scorder"][0],
            figname=None
        )
        if args.save:
            fig.savefig(save_dir + "config_local.png", dpi=144)
            plt.close()
        else:
            plt.show()
