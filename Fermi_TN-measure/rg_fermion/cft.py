"""
Calculate CFT data for Grassmann tensor network 
(using natural dual convention)
```
        1           1    
        ↓           ↓
    2 → A → 0   2 → B → 0
        ↓           ↓
        3           3
```
"""

import torch
torch.set_default_dtype(torch.float64)
import numpy as np
import gtensor as gt
from gtensor import GTensor
import gtensor.legacy as gt0
import gtensor.tools as gtools
from scipy.sparse.linalg import eigs, LinearOperator
from itertools import product
from typing import IO


def _get_conv(a: GTensor, b: GTensor):
    if a.dual == (1,0,1,0) and b.dual == (0,1,0,1):
        return "loop"
    elif a.dual == b.dual == (1,0,0,1):
        return "nat"
    else:
        raise ValueError("unrecognized tensor dual convention")


def _change_convention(a: GTensor, b: GTensor):
    """Conversion between natural and loop convention"""
    a = a.flip_dual((2,3))
    b = b.flip_dual((0,1), minus=False)
    return a, b


def rotate90_ccw(a: GTensor, b: GTensor):
    """
    counter-clockwise rotation of tensor network 
    by 90 degrees to change vertical TM to horizontal TM
    """
    return (
        b.flip_dual((0,2)).transpose(3,0,1,2), 
        a.flip_dual((0,2), minus=False).transpose(3,0,1,2)
    )


def build_row(
    a: GTensor, b: GTensor, vec: GTensor, bc: str
):
    """
    Apply a row of ABAB... onto the vector, 
    whose length must be an even number
    
    PBC
    ```
            0   1   2   3
            ↓   ↓   ↓   ↓
        ..→ A → B → A → B →..
            ↓   ↓   ↓   ↓
            0   1   2   3
            ↓___↓___↓___↓
                vec
    ```
    """
    assert _get_conv(a, b) == "nat"
    assert vec.ndim % 2 == 0
    assert bc in ('p', 'a')
    n = vec.ndim
    for i in range(n):
        t = a if i % 2 == 0 else b
        if i == 0:
            vec = gt0.tensordot(t, vec, ((3,), (0,), (1,)), False)
        elif i == n - 1:
            g = 1 if bc == "p" else -1
            vec = gt0.tensordot(t, vec, ((0,2,3), (n,0,n+1),(g,-1,1)), False)
        else:
            vec = gt0.tensordot(t, vec, ((2,3), (0,i+2), (-1,1)), False)
    perm = list(reversed(range(n)))
    vec = vec.transpose(*perm)
    return vec


def tm2_eigs(
    a: GTensor, b: GTensor, nev=80, mode="h", 
    input_conv="nat", return_eigenvectors=False
):
    """
    find vertical eigenvalues of 2 x 2 transfer matrix
    using the natural convention 

    Parameters
    ----------
    a, b: the input tensors (in natural convention)
    nev : number of eigenvalues to be kept
    mode: "v" (vertical) or "h" (horizontal)
    """
    if return_eigenvectors is True:
        raise NotImplementedError
    assert input_conv in ("loop", "nat")
    if input_conv == "loop":
        a, b = _change_convention(a, b)
    if mode == "h":
        a, b = rotate90_ccw(a, b)
    assert _get_conv(a, b) == "nat"
    # construct transfer matrix operator
    shapeV = tuple((a.shape[par][3], b.shape[par][3]) for par in range(2))
    shapeV_merged = gtools.merge_shape(shapeV, (len(shapeV[0]),))
    mDim = tuple(dim[0] for dim in shapeV_merged)
    
    def mat2(v: np.ndarray):
        # v with two axes merged
        vf = gt.unflatten(shapeV, (0,)*2, vpar, torch.from_numpy(v))
        # the tensor m is defined below
        vf = build_row(a, b, vf, bc)
        vf = build_row(b, a, vf, bc)
        return vf.flatten().numpy()
    
    if np.any(np.array(a.shape) < 2) or np.any(np.array(b.shape) < 2):
        return None
    w = dict()
    for bc, vpar in product(["p","a"], range(2)):
        label = bc + str(vpar)
        op = LinearOperator(matvec=mat2, shape=(mDim[vpar], mDim[vpar]), dtype=complex)
        wTmp = eigs(op, k=min(nev, mDim[vpar]-2), which="LM", return_eigenvectors=False)
        w[label] = wTmp[ np.argsort(np.abs(wTmp)) ][::-1][:nev]
    return w


def tm4_eigs(
    a: GTensor, b: GTensor, nev=80, mode="h", 
    input_conv="nat", return_eigenvectors=False
):
    """
    find vertical/horizontal eigenvalues of 4 x 2 / 2 x 4 transfer matrix

    Axes order convention
    ```
            1
            |
        0 - A - 2
            |
            3
    ```

    Transfer matrix (mode = "v")
    ```
            0   1   2   3
            |   |   |   |
        ..- B - A - B - A -..
            |   |   |   |
        ..- A - B - A - B -..
            |   |   |   |
            4   5   6   7
            0   1   2   3
            |___|___|___|
            vec
    ```
    
    Parameters
    ----------
    a, b: the input tensors   
    nev : number of eigenvalues to be kept
    mode: "h" or "v"
    """
    if return_eigenvectors is True:
        raise NotImplementedError
    assert input_conv in ("loop", "nat")
    if input_conv == "loop":
        a, b = _change_convention(a, b)
    if mode == "h":
        a, b = rotate90_ccw(a, b)
    assert _get_conv(a, b) == "nat"
    # construct transfer matrix operator
    shapeV = tuple((a.shape[par][3], b.shape[par][3], a.shape[par][3], b.shape[par][3]) for par in range(2))
    shapeV_merged = gtools.merge_shape(shapeV, (len(shapeV[0]),))
    mDim = tuple(dim[0] for dim in shapeV_merged)

    def mat4(v: np.ndarray):
        # v with two axes merged
        vf = gt.unflatten(shapeV, (0,)*4, vpar, torch.from_numpy(v))
        # the tensor m is defined below
        vf = build_row(a, b, vf, bc)
        vf = build_row(b, a, vf, bc)
        return vf.flatten().numpy()
    
    if np.any(np.array(a.shape) < 2) or np.any(np.array(b.shape) < 2):
        return None
    w = dict()
    for bc, vpar in product(["p","a"], range(2)):
        label = bc + str(vpar)
        op = LinearOperator(matvec=mat4, shape=(mDim[vpar], mDim[vpar]), dtype=complex)
        wTmp = eigs(op, k=min(nev, mDim[vpar]-2), which="LM", return_eigenvectors=False)
        w[label] = wTmp[ np.argsort(np.abs(wTmp)) ][::-1][:nev]
    return w


def tm4a_eigs(
    a: GTensor, b: GTensor, nev=80, mode="h", 
    input_conv="nat", return_eigenvectors=False
):
    """
    find vertical eigenvalues of the approximate
    4 x 2 transfer matrix (in natural convention)

    natural convention
    (rotated counter-clockwise by 45 deg)
    ```
        1       0       1       0
          ↘   ↗           ↘   ↗
            A               B
          ↗   ↘           ↗   ↘
        2       3       2       3
    ```

    This orientation is the result of one step of RG (with no adjusting rotation)
    """
    if return_eigenvectors is True:
        raise NotImplementedError
    assert input_conv in ("loop", "nat")
    if input_conv == "loop":
        a, b = _change_convention(a, b)
    if mode == "h":
        a, b = rotate90_ccw(a, b)
    assert _get_conv(a, b) == "nat"
    shapeV = tuple((b.shape[par][3], b.shape[par][2], b.shape[par][3], b.shape[par][2]) for par in range(2))
    shapeV_merged = gtools.merge_shape(shapeV, (len(shapeV[0]),))
    mDim = tuple(dim[0] for dim in shapeV_merged)

    def mat4(v: np.ndarray):
        # v with four axes merged
        vf = gt.unflatten(shapeV, (0,)*4, vpar, torch.from_numpy(v))
        if bc == "p":
            """
            PBC
                0   3   2   1
                ↘   ↗   ↘   ↗
                  A       A     :
                ↗   ↘   ↗   ↘   ↗
                :     B       B
                    ↗   ↘   ↗   ↘
                    :   :   :   :
                    3   2   1   0
                    ↑___↓___↑___↓
                        vec
            """
            vf = gt0.tensordot(b, vf, [(3,2),(0,1),(1,-1)], False)
            vf = gt0.tensordot(b, vf, [(3,2),(2,3),(1,-1)], False)
            vf = gt0.tensordot(a, vf, [(3,2),(3,0),(1,-1)], False)
            vf = gt0.tensordot(a, vf, [(3,2),(2,3),(1,-1)], False)
        # Anti-PBC
        elif bc == "a":
            """
            Anti-PBC
                0   3   2   1
                ↘   ↗   ↘   ↗
                  A       A     :
                ↗   ↖   ↗   ↘   ↗
                :     B       B
                    ↙   ↘   ↗   ↘
                    :   :   :   :
                    3   2   1   0
                    ↓___↓___↑___↓
                        vec
            """
            vf = gt0.tensordot(b, vf, [(3,2),(0,1),(1,-1)], False)
            vf = gt0.tensordot(b.flip_dual((1,2)), vf, [(3,2),(2,3),(1,-1)], False)
            vf = gt0.tensordot(a, vf, [(3,2),(3,0),(1,-1)], False)
            vf = gt0.tensordot(a, vf, [(3,2),(2,3),(1,-1)], False)
        else:
            raise ValueError
        
        vf = vf.transpose(1,2,3,0)
        return vf.flatten().numpy()
    
    if np.any(np.array(a.shape) < 2) or np.any(np.array(b.shape) < 2):
        return None
    w = dict()
    for bc, vpar in product(["p","a"], range(2)):
        label = bc + str(vpar)
        op = LinearOperator(matvec=mat4, shape=(mDim[vpar], mDim[vpar]), dtype=complex)
        wTmp = eigs(op, k=min(nev, mDim[vpar]-2), which="LM", return_eigenvectors=False)
        w[label] = wTmp[ np.argsort(np.abs(wTmp)) ][::-1][:nev]
    return w


def cal_dim(ev: dict, tau: complex):
    """
    CFT scaling dimension from eigenvalues of 
    transfer matrix with modular parameter `tau`
    """
    # normalize by the eigenvalue of largest magnitude from all sectors
    max_absev = max([abs(value[0]) for value in ev.values()])
    prefac = - 1 / (2 * np.pi) / np.imag(tau)
    scdim = dict()
    for key in ev:
        scdim[key] = prefac * np.log(np.abs(ev[key]) / max_absev)
    return scdim


def cal_tau(
    evh: dict, evv: dict, evld=None, evrd=None
):
    """
        Use the eigenvalues of the four
        2 x 2 transfer matrices (h, v, ld, ud)
        to determine the modular parameter
    """
    # aspect ratio (vh/w) = v
    xi1 = np.sqrt(
        np.log(np.abs(evh["a0"][1::]) / np.abs(evh["a0"][0])) / 
        np.log(np.abs(evv["a0"][1::]) / np.abs(evv["a0"][0]))
    )
    v = 1 / xi1
    if (not evld is None) or (not evrd is None):
        # ratio of ld and rd eigenvalues
        xi2 = np.sqrt(
            np.log(np.abs(evrd[1::]) / np.abs(evrd[0])) / 
            np.log(np.abs(evld[1::]) / np.abs(evld[0]))
        )
        # inclination angle of w2 
        # (theta, expected value pi/2 = 1.57)
        theta = np.arccos(
            (1 - xi2**2) * (1 + xi1**2) / (
                2 * (1 + xi2**2) * xi1
            )
        )
        return v, theta
    else:
        return v


def cal_cft(
    w: int, tmEigValV, tmEigValH, 
    cftFiles, iterMain: int
):
    """CFT scaling dimension from w x (h = 2) transfer matrix"""
    # aspect ratio
    assert w == 2 or w == 4
    asp = w // 2
    prefix = "d" + str(w)   # "d2" or "d4"
    mattype_list = ["p0","p1","a0","a1"]
    # Transfer in vertical direction (contract horizontal bonds)
    # max eigval in all sectors
    expGsV = max([abs(value[0]) for value in tmEigValV.values()])
    for mattype in mattype_list:
        # "d2v-p0", "d2v-p1", "d2v-a0", "d2v-a1"
        # or "d4..." etc
        label = prefix + "v-" + mattype
        cal_scdim(tmEigValV[mattype], expGsV, asp, cftFiles[label], iterMain)

    # Transfer in horizontal direction (contract vertial bonds)
    # max eigval in all sectors
    expGsH = max([abs(value[0]) for value in tmEigValH.values()])
    for mattype in mattype_list:
        # "d2h-p0", "d2h-p1", "d2h-a0", "d2h-a1"
        # or "d4..." etc
        label = prefix + "h-" + mattype
        cal_scdim(tmEigValH[mattype], expGsH, asp, cftFiles[label], iterMain)


def cal_scdim(
    eigVal, expGs, 
    asp: float, f: IO, it: int
):
    """
    Calculate scaling dimension from eigenvalues of transfer matrix
    
    Parameters
    ----------
    eigVal: eigenvalues of the trsf mat   
    expGs : max eigenvalue from all sectors
    asp   : aspect ratio of the trsf mat (w / h)   
    f     : file to save calculation result   
    it    : iteration step of RG   

    Returns
    -------
    Calculated scaling dimensions
    """
    f.write("{:4d}".format(it))
    for val in eigVal:
        if np.abs(val) > 0:
            f.write("  {:14.8G}".format( -asp*np.log(np.abs(val/expGs))/(2*np.pi) ))
    f.write("\n")
    dims = - asp * np.log(np.abs(val / expGs)) / (2 * np.pi)
    return dims
