import numpy as np
from argparse import Namespace
from utils import get_precision

def get_command(n: Namespace, param: dict[str]):
    """Get simple update options from preset"""
    command = "{} {} --task {} ".format(
        n.model, n.dcut, n.task
    )
    if n.flt is True:
        command += "--flt "
    command += "--maxloop {} ".format(n.maxloop)
    command += "--maxRGstep {} ".format(n.maxRGstep)
    # set variables
    command += "-n "
    for name in param.keys():
        command += name + " "
    command += "-v "
    for value in param.values():
        prec = get_precision(value)
        command += f"{value:.{prec}f} "
    return command

preset_1d = {
    "tV": Namespace(
        model = "tV", dcut = 48, task = "bm2", 
        params = [
            {"g": g} for g in np.arange(0.0, 0.981, 0.02)
            # {"g": 0.2}
        ], 
        calcft4=True, flt=True, maxloop=150, 
        maxRGstep = 19
    )
}

# model = "kitaev"
# task = "PT1"
# dcut = 24
# names = ["beta", "V", "mu"]
# start = [0.6, 0.0, 2.0]
# end   = [1.4, 0.0, 2.0]
# step  = [0.2, 0.1, 0.1]

# model = "kitaev"
# task = "tV"
# dcut = 24
# names = ["beta", "D", "V", "mu"]
# start = [1.0, 0.0, 0.0, 1.6]
# end   = [1.0, 0.0, 0.0, 2.4]
# step  = [1.0, 0.1, 0.1, 0.2]
