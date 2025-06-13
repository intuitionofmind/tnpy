import os
from glob import glob
from pprint import pprint
from update_ftps.sutools import get_tpstJconv
from update_ftps.tps_io import load_tps
from vumps.files_par import load_peps
from phys_models.onesiteop import get_tJconv
from tqdm import tqdm

folders = glob(
    "update_ftps/results_2d-ZW/tJ-sut4_*-D12/mu-*/conv3-sample*/"
)
for f in tqdm(folders):
    # get value of mu
    mu = float(f.split("mu-", 1)[-1].split(os.sep)[0])
    filename = f + "tpsinfo.txt"
    with open(filename, "w") as info:
        try:
            tensors, wts = load_tps(f)
            tJ_conv = get_tpstJconv(tensors)
            Dphy = (tensors["Ta"].DE[0], tensors["Ta"].DO[0])
            Dcut, De = (tensors["Ta"].DS[1], tensors["Ta"].DE[1])
        except KeyError:
            fG0ss, fG1ss = load_peps(2, 2, f, "MN")
            Dphy = (fG0ss[0][0].DE[0], fG0ss[0][0].DO[0])
            Dcut, De = fG0ss[0][0].DS[1], fG0ss[0][0].DE[1]
            tJ_conv = get_tJconv(Dphy)
        param = {
            "t": 3.0, "J": 1.0, "mu": mu,
            "Dphy": Dphy, "Dcut": Dcut, "De": De,
            "tJ_convention": tJ_conv
        }
        for key, value in param.items():
            info.write("{:<22s}{}\n".format(key, value))
        info.write("\n")
        # measures = tJ_localmeas(
        #     tensors, wts, tJ_conv, 
        #     mode1="site", mode2="bond", 
        #     nnn_t = ("t2" in param), nnn_J = ("J2" in param)
        # )
        # results = process_measure(measures, param)[0]
        # pprint(results, stream=info, width=60)
