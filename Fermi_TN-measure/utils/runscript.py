from typing import TextIO

def write_script(
    w: TextIO, command: str, preamble: str,
    memory: str, cpunum: int, workdir: str, 
    queue: str, allot_time=None, 
    envvars: dict[str] = {}, shell="/bin/zsh"
):
    """Function for script generation"""
    assert workdir[-1] == "/"
    w.write(f"#!{shell} \n\n")
    w.write(f"#PBS -S {shell} \n")
    if allot_time is None:
        if queue is None:
            allot_time = "48:00:00"
        elif queue == "debug":
            allot_time = "00:30:00"
        elif queue == "long":
            allot_time = "168:00:00"
        else:
            allot_time = "60:00:00"
    w.write("#PBS -l walltime={}\n".format(allot_time))
    if queue is not None:
        w.write("#PBS -q {}\n".format(queue))
    
    # memory request
    w.write("#PBS -l mem={}GB\n".format(memory))
    # cpu usage
    w.write("#PBS -l nodes=1:ppn={}\n\n".format(cpunum))

    # environmental variables
    for key, val in envvars.items():
        w.write(f"export {key}={val}\n")

    # working directory
    w.write("PBS_O_WORKDIR={}\n".format(workdir))
    # preamble
    # activate conda env, set env variables, etc
    w.write(preamble + "\n")
    # go to working directory
    w.write("cd $PBS_O_WORKDIR\n")
    # record start time
    w.write("echo \"Start at $(date)\"\n\n")
    # command to be executed
    w.write(command + "\n\n")
    # record finish time
    w.write("echo \"End at $(date)\"\n")
    
