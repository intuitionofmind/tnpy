"""
Auxiliary functions for RG measurement
"""

import os
import numpy as np

def expval(normfile: str, rgstep=None):
    """
    Calculate expectation value from TNR generated norm file
    """
    try:
        nrm = np.loadtxt(normfile, dtype=float)
    except:
        nrm = np.loadtxt(normfile, dtype=complex)
        # if np.allclose(nrm, nrm.real):
        #     nrm = nrm.real
    assert nrm.shape[0] % 2 == 0
    if rgstep is None:
        rgstep = nrm.shape[0] // 2
    nrm = nrm[0 : 2*rgstep]
    nume_nrm = nrm[np.arange(0, 2 * rgstep, 2, dtype=int)][:,1] ** 4
    deno_nrm = nrm[np.arange(1, 2 * rgstep, 2, dtype=int)][:,1] ** 4
    final_nume = nrm[-2, -1]
    final_deno = nrm[-1, -1]
    return np.prod(nume_nrm / deno_nrm) * (final_nume / final_deno)

def gen_meas_file(normfile: str, maxRGstep=8, dry_run=False):
    """
    Create measurement result file 
    (including results at each RG step)

    normfile format:
        any_path/measure_key/Tnorm.txt
    """
    # stored in the same folder as the norm file
    workdir = os.path.dirname(normfile) + os.sep
    key = workdir.rsplit(os.sep, 2)[-2]
    if dry_run is True:
        print(key)
        return
    with open(workdir + "measure.txt", mode="w", buffering=1) as f:
        for rgstep in range(1, maxRGstep+1):
            try:
                result = expval(normfile, rgstep)
                f.write("{:10s} {:23.16G} {:4d}\n".format(key, result, rgstep))
            except IndexError:
                break
