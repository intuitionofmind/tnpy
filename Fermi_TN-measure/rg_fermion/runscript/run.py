"""
Generate job submission scripts
(1+1D quantum system CFT data from RG)
"""
import os
import sys
pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(pwd)
sys.path.append(os.getcwd())
from run_presets import preset_1d, get_command
from utils.runscript import write_script

queue = "normal"
script = "main_RGcft.py"
memory = 16
cpunum = 7
account = "mine"

if account == "mine":
    workdir = "~/Fermion_TN/"
    python_path = "~/miniconda3/envs/py310/bin/python"
elif account == "cent":
    workdir = "/mnt/doom/zhengyuanyue/Fermion_TN/"
    python_path = "~/miniconda3/bin/python"
    # overwrite queue and cpu setting
    queue = None
    if cpunum > 8:
        cpunum = 8

key = "tV"
n = preset_1d[key]
scriptdir = pwd + '/sub_job/{}-{}'.format(n.model, n.task)
if account == "mine":
    scriptdir += os.sep
elif account == "cent":
    scriptdir += "-cent/"
    queue, cpunum = "cent", 4
os.makedirs(scriptdir, exist_ok=True)

preamble = ""
for i, param in enumerate(n.params):
    with open(scriptdir + '{}-{}-{}.sh'.format(n.model, n.task, i), 'w') as w:
        # command to run main program
        command = "{} -u {} ".format(python_path, script)
        command += get_command(n, param)
        write_script(
            w, command, preamble, memory, cpunum, 
            workdir, queue, allot_time=None
        )
