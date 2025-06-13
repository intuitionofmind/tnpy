"""
Load TNR measurement and get physical quantities
"""

import os
import argparse
from pprint import pprint
from utils import dict_loadtxt

from update_ftps.tJ_loadRGmeas import load_measure
from localmeas.tJ_measprocess import process_measure
from localmeas.plot_config import plt, plot_config

def get_args():
    parser = argparse.ArgumentParser(
        description="Load and process RG measurement files")
    # input options
    parser.add_argument("folder", type=str, 
        help="Folder containing the tensors and weights")
    parser.add_argument("info", type=str, 
        help="Path to tpsinfo.txt file")
    parser.add_argument("--tJ_conv", default=None,
        help="Set tJ convention when it is not provided in tpsinfo.txt")
    parser.add_argument("--make_plot", action="store_true", 
        help="Plot configuration")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    param = dict_loadtxt(args.info)

    folder: str = args.folder
    assert folder.endswith(os.sep)
    make_plot = args.make_plot

    measures = load_measure(folder, rgstep=6, auto_complete=False)
    results = process_measure(measures, param)
    pprint(results)

    if make_plot:
        fig, ax = plot_config(**results)
    plt.show()
    