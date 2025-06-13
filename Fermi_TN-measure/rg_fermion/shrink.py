"""
Build tensor network representation of the
partition function of 1+1D quantum models
from imaginary time evolution gate exp(-beta * H) 
"""

import numpy as np
# from tqdm import tqdm
from cmath import isclose
import logs
from . import norm
import gtensor as gt
from gtensor import GTensor
import gtensor.linalg as gla
import gtensor.legacy as gt0
import mps.pbc as mps

def normalize(
    Ta: GTensor, Tb = None, Tc = None
) -> tuple[GTensor, ...]:
    """
    Normalize by 1 x 1 norm
        nrm = T[h,v,h,v]
    or 2 x 1 norm
        nrm = Ta[r,x,l,x] * Tb[l,y,r,y]
    or (rarely used)
        nrm = Ta[1,x,3,x] * Tb[2,y,1,y] * Tc[3,z,2,z]
    """
    if Tb is None:
        # Tensor A only
        assert Tc is None, "incorrect use of 'normalize'"
        nrm = gt0.trace(Ta, axis1=(0,1), axis2=(2,3)).item()
        fac = nrm
        # round the number to avoid division underflow
        fac = np.around(fac, decimals=10)
        logs.info.write('{:<20.16f} {:.3f}\n'.format(fac.real, fac.imag))
        return (Ta / fac,)
    elif Tc is None:
        assert isinstance(Tb, GTensor)
        # Network A - B
        A = gt0.trace(Ta, axis1=1, axis2=3)
        B = gt0.trace(Tb, axis1=1, axis2=3)
        nrm = gt0.tensordot(A, B, ([0,1],[1,0],[1,-1])).item()
        fac = np.sqrt(nrm)
        # round the number to avoid division underflow
        fac = np.around(fac, decimals=10)
        logs.info.write('{:<20.16f} {:.3f}\n'.format(fac.real, fac.imag))
        return (Ta / fac, Tb / fac)
    else:
        assert isinstance(Tc, GTensor)
        # Network A - B - C
        A = gt0.trace(Ta, axis1=1, axis2=3)
        B = gt0.trace(Tb, axis1=1, axis2=3)
        C = gt0.trace(Tc, axis1=1, axis2=3)
        AB = gt0.tensordot(A, B, (0,1))
        nrm = gt0.tensordot(AB, C, ([0,1],[0,1])).item()
        fac = nrm**(1/3)
        # round the number to avoid division underflow
        fac = np.around(fac, decimals=10)
        logs.info.write('{:<20.16f} {:.3f}\n'.format(fac.real, fac.imag))
        return (Ta / fac, Tb / fac, Tc / fac)

def build_tensor2(gate: GTensor):
    """
    Construct tensor from 2-body Trotter gate

    Input gate axis order
    ```
        0      1
          ↘︎  ↙︎
          Gate
          ↙︎  ↘︎
        2      3
    ```
    """
    assert len(gate.shape[0]) == 4
    """
        1      3            1                  2
          ↘︎  ↙︎                ↘              ↙
          Gate      -->        S1 → 2   0 → S3
          ↙︎  ↘︎                ↙              ↘
        0      2            0                  1
    """
    S1, _, S3 = gla.svd(gate.transpose(2,0,3,1), 2, absorb_s=True)
    """
                            1       0
                              ↘   ↙
                                S4
        1      0                ↓
          ↘︎  ↙︎                  2
          Gate      -->
          ↙︎  ↘︎                  0
        3      2                ↓
                                S2
                              ↙   ↘
                            2       1
    """
    S4, _, S2 = gla.svd(gate.transpose(1,0,3,2), 2, absorb_s=True)
    """
    Combine SVD results of the Trotter gate network

                    0
                    ↓
                    S2
                  ↙   ↘
                2       1                    1
              ↙           ↘                  ↓
        0 → S3            S1 → 2  --->  2 → T → 0
              ↘           ↙                  ↓
                1       0                    3
                  ↘   ↙
                    S4
                    ↓
                    2
    """
    S12 = gt.tensordot(S1, S2, (1,1))
    S34 = gt.tensordot(S3, S4, (1,1))
    return gt.tensordot(S12, S34, ((0,3),(2,1)))

def splith_3body(t: GTensor):
    """
    Split a 3-body gate horizontally into three pieces
    ```
    Input:          Output:
    0   1   2       1         2         2
    ↓   ↓   ↓       ↓         ↓         ↓
    ----T----       L → 2 0 → C → 3 0 → R
    ↓   ↓   ↓       ↓         ↓         ↓
    3   4   5       0         1         1
    ```
    """
    l, _, cr = gla.svd(t.transpose(3,0,4,1,5,2), 2, absorb_s=True)
    c, _, r = gla.svd(cr, 3, absorb_s=True)
    for p in range(2):
        assert l.shape[p][2] == c.shape[p][0] \
            and c.shape[p][3] == r.shape[p][0], \
                "inconsistent output dimensions"
    return l, c, r

def mergev_lcr(l: GTensor, c: GTensor, r: GTensor):
    """
    Merge l, c, r in vertical direction 
    into 3 types of tensors (t1, t2, t3)
    ```
        t1      t2      t3
          ↓       ↓       ↓
        → c →   → r       l →       1
          ↓       ↓       ↓         ↓
        → r       l →   → c →   2 → T → 0
          ↓       ↓       ↓         ↓
          l →   → c →   → r         3
          ↓       ↓       ↓
    ```
    """
    # get t1
    t1 = gt.tensordot(
        gt.tensordot(c, r, (1,2)), l, (4,1)
    ).transpose(0,3,1,4,2,5).merge_axes((2,1,1,2), (1,1,1,-1)).transpose(3,1,0,2)
    # get t2
    t2 = gt.tensordot(
        gt.tensordot(r, l, (1,1)), c, (2,2)
    ).transpose(0,3,1,4,2,5).merge_axes((2,1,1,2), (1,1,1,-1)).transpose(3,1,0,2)
    # get t3
    t3 = gt.tensordot(
        gt.tensordot(l, c, (0,2)), r, (3,2)
    ).transpose(2,4,0,5,1,3).merge_axes((2,1,1,2), (1,1,1,-1)).transpose(3,1,0,2)
    return t1, t2, t3

def mergeh_3t(t1: GTensor, t2: GTensor, t3: GTensor):
    """
    Merge t1, t2, t3 in hotizontal direction 
    into a single tensor t
    """
    t = gt.tensordot(
        gt.tensordot(t1, t2, (0,2)), t3, (3,2)
    ).transpose(0,3,6,1,5,2,4,7).merge_axes((3,1,1,3), (1,1,1,-1)).transpose(2,0,1,3)
    return t

def build_tensor3(gate: GTensor):
    """Construct tensor from 3-body Trotter gate"""
    assert len(gate.shape[0]) == 6
    l, c, r = splith_3body(gate)
    t1, t2, t3 = mergev_lcr(l, c, r)
    t = mergeh_3t(t1, t2, t3)
    return t

def build_tensor(gate: GTensor):
    """Construct tensor from 2-body or 3-body Trotter gate"""
    if len(gate.shape[0]) == 4:
        return build_tensor2(gate)
    elif len(gate.shape[0]) == 6:
        return build_tensor3(gate)

def shrink_once(
    ts: list[GTensor], dm, it=None, no_combine=False
):
    """
    Perform one shrink step (2^N) 
    of imag-time evolution layers

    Parameters
    -----------
    ts: sequence of ndarray's
            1                1
            ↓                ↓
        2 → ts[0] → 0 .. 2 → ts[i] → 0 
            ↓                ↓
            3                3
    dm: int
        SVD bond truncation cutoff dimension
    it: None or int
        Iteration step of shrinking
    no_combine: dafault `False`
        False: multiplying two rows of tensors, combine & compress virtual legs
        True: only normalize and canonicalize one row of tensors
    """
    if it is not None:
        logs.error.write('shrink step = {:d}\n'.format(it))
    logs.error.write('shape of T\'s = ')
    for t in ts:
        logs.error.write('{} '.format(t.shape))
    logs.error.write('\n')
    # usually ts contains only one tensor
    # normalize by 1 x 1 norm
    ts = normalize(*ts)
    mat = []
    for t in ts:
        if no_combine:
            pass
        else:
            t = gt.tensordot(t, t, [3,1]).transpose(0,3,1,2,4,5)\
                .merge_axes((2,1,2,1), order=(1,1,-1,1))
        mat.append(t)
    mat, _, svdErrs, qrErr = mps.canonicalize(
        [t.transpose(2,1,3,0) for t in mat], dm
    )
    mps.print_qrErr(qrErr, logs.error)
    mps.print_svdErrs(svdErrs, logs.error)
    logs.error.write('\n')
    return [t.transpose(3,1,0,2) for t in mat]


def shrink_all(gate: GTensor, params: dict):
    """
    Construct tensor network from 
    imag-time evolution gate with 4 axes
    
    Input gate axis order
    ----
    Site    i     i+1
            0      1
              ↘︎  ↙︎
              Gate
              ↙︎  ↘︎
            2      3

    Parameters
    ----------
    gate: 4-axis GTensor
        imag-time evolution gate
    params: dict
        parameters of compression of evolution layers

    Returns
    -------
    Normalized network tensor Ta, Tb
    """
    if params['3body']:
        raise NotImplementedError
    dm = params['d_cutRG']
    assert np.all( np.array(gate.shape)**2 <= dm ), \
           'dm must be at least di^2 for quantum models'
    logs.error.write('shrink on time direction\n\n')
    print("Compressing tau direction:")
    # construct tensor from time evolution gates
    T = build_tensor(gate)
    # normalize and canonicalize tensor
    T = shrink_once([T], dm, no_combine=True)[0]

    # shrink time evolution gate layers (vertical direction)
    # for i in tqdm(range(params['verticalStep'])):
    for i in range(params['verticalStep']):
        T = shrink_once([T], dm, it=i)[0]
    Ta = T
    Tb = Ta.copy()
    logs.error.write('\n')
    """
    convert from natural to loop convention
            1         1               1         1
            ↓         ↓               ↑         ↓
        2 → B → 0 2 → A → 0  ==>  2 → B ← 0 2 ← A → 0
            ↓         ↓               ↓         ↑
            3         3               3         3
            1         1               1         1
            ↓         ↓               ↓         ↑
        2 → A → 0 2 → B → 0  ==>  2 ← A → 0 2 → B ← 0
            ↓         ↓               ↑         ↓
            3         3               3         3
    """
    Ta = gt.flip_dual(Ta, [2,3])
    Tb = gt.flip_dual(Tb, [0,1], minus=False)
    # normalize using the 2 x 2 norm
    fac = (norm.norm4(Ta, Tb))**0.25
    if isclose(fac, fac.real):
        fac = fac.real
    Ta /= fac
    Tb /= fac
    logs.info.write('{:<20.16f} {:.3f}\n'.format(fac.real, fac.imag))
    return Ta, Tb
