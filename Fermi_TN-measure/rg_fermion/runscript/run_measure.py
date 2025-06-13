"""
Generate job submission scripts
(1+1D quantum system CFT data from RG)
"""

import os
import sys
import numpy as np
pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(pwd.rsplit(os.sep, 2)[0]+os.sep)
from utils.runscript import write_script
from utils import get_precision
from itertools import product, chain

script = "main_RGmeas.py"
account = "cent"
queue = "doom"

if account == "mine":
    workdir = "~/Fermion_TN/"
    python_path = "~/miniconda3/envs/py310/bin/python"
elif account == "cent":
    workdir = "/mnt/doom/zhengyuanyue/Fermion_TN/"
    python_path = "python"

if account == "cent":
    preamble = "source /home/gzcgu/zhengyuanyue/.zshrc\n"
    preamble += "num_threads=$SLURM_JOB_CPUS_PER_NODE\n"
    preamble += "if ! [ \"$SLURM_NODELIST\" = \"doom\" ]; then \n"
    preamble += "    num_threads=$(($num_threads / 2))\n"
    preamble += "fi\n"
    preamble += "export MKL_NUM_THREADS=$num_threads\n"
    preamble += "export OPENBLAS_NUM_THREADS=$num_threads\n"
else:
    preamble = ""

memory = "12GB"
allot_time = "72:00:00"
envvars = {}
flt, maxloop, dRG = True, 100, 32
task = "tJ-sut4_pdw-D4"
scriptdir = "rg_fermion/runscript/sub_job/"
scriptdir += f"{task}/flt-{flt}_loop-{maxloop}_dRG-{dRG}/"
os.makedirs(scriptdir, exist_ok=True)
mus = np.concatenate([
    # np.array([5.10, 5.15, 5.20, 5.24]),
    # np.arange(5.28, 5.441, 0.02)
    np.array([5.20, 5.25]),
    np.arange(5.28, 5.421, 0.02),
    np.arange(5.44, 5.481, 0.01)
])
unif, meas = True, True
for mu in [4.00]:
    prec = get_precision(mu)
    tps_dir = f"update_ftps/results_2d-ZW/{task}/mu-{mu:.{prec}f}/conv3-sample5/"
    subdir = f"{scriptdir}mu-{mu:.{prec}f}/"
    os.makedirs(subdir, exist_ok=True)
    # RG of uniform tensors
    if unif:
        cpunum = 8
        for plq in (1,2,3,4):
            with open(subdir + f"measRG-unif{plq}.sh", "w") as w:
                command = ""
                # boundary mps calculation
                command += f"{python_path} -u {script} "
                command += f"unif tJ {tps_dir} --plq {plq} "
                if flt: command += "--flt "
                command += f"--dcut {dRG} "
                command += f"--maxloop {maxloop} "
                command += "--rgstep 14 "
                command += "--save_unif "
                write_script(
                    w, command, preamble, memory, cpunum, workdir, queue, 
                    allot_time=allot_time, envvars=envvars
                )
    # measurements
    if meas:
        cpunum = 4
        plqs = [str(i) for i in (1,2,3,4)]
        measkeys = ["".join(term) for term in chain(
            # doping
            product(["xy", "wz"], ["NhId", "IdNh"], ["1", "3"]),
            # # 1st neighbor hopping, spin correlation and pairing
            # product(["xy", "yz", "wz", "xw"], [
            #     "CpuCmu", "CpdCmd", 
            #     "SpSm", "SzSz", "NudNud",
            #     "CmuCmu", "CmdCmd"
            # ], ["1"]),
            # 2nd neighbor hopping, spin correlation and pairing
            product(["xz", "wy"], [
                # "CpuCmu", "CpdCmd", 
                "SpSm", "SzSz", 
                # "NudNud",
                # "CmuCmu", "CmdCmd"
            ], plqs)
        )]
        for i, measkey in enumerate(measkeys):
            with open(subdir + f"measRG-{i}.sh", "w") as w:
                command = ""
                # boundary mps calculation
                command += f"{python_path} -u {script} "
                command += f"meas tJ {tps_dir} {measkey} "
                if flt: command += "--flt "
                command += f"--dcut {dRG} "
                command += f"--maxloop {maxloop} "
                command += "--rgstep 14 "
                command += "--load_unif "
                write_script(
                    w, command, preamble, memory, cpunum, workdir, queue, 
                    allot_time=allot_time, envvars=envvars
                )
