"""
mps_pbc.py
======
PBC-MPS algorithms
"""

import numpy as np
import gtensor as gt
from math import inf
from gtensor import GTensor
import gtensor.linalg as gla
from .obc import EPSS_DEFAULT, _check_mps
from .obc import proj_from_RL, _normalize_rl
from .obc import print_svdErrs
from copy import deepcopy
from warnings import warn
from typing import IO

# import atexit
# import line_profiler as lp
# profile = lp.LineProfiler()
# atexit.register(profile.print_stats)

EPS = np.finfo(float).eps
TOL = 1.e-8
MIN_ITER = 10
MAX_ITER = 50
np.seterr(all='raise')

def _check_qr_convergence(
    Rs: list[GTensor], Ls: list[GTensor], 
    Rs_old: list[GTensor], Ls_old: list[GTensor], 
    abs_mode=True
):
    """
    Compare new `Rs, Ls` with old `Rs, Ls`
    (`Rs, Ls` must be normalized on input)

    Returns
    -------
    The norm difference between each pair of 
    `Rs`/`Rs_old` and `Ls`/`Ls_old`
    """
    assert len(Rs) == len(Ls) == len(Rs_old) == len(Ls_old), \
        'inconsistent number of Rs/Ls matrices'
    diff_Rs = tuple(
        gt.norm(gt.absolute(R) - gt.absolute(R_old)) / gt.norm(R_old) 
        if abs_mode else gt.norm(R - R_old) / gt.norm(R_old) 
        for R, R_old in zip(Rs, Rs_old)
    )
    diff_Ls = tuple(
        gt.norm(gt.absolute(L) - gt.absolute(L_old)) / gt.norm(L_old)
        if abs_mode else gt.norm(L - L_old) / gt.norm(L_old) 
        for L, L_old in zip(Ls, Ls_old)
    )
    return diff_Rs + diff_Ls


def loop_qr1(
    Ms: list[GTensor], i0: int, normalize=True
):
    """
    Perform one round of QR/LQ on a PBC-MPS
    to find R/L matrix on bond `i0`
    (connecting `Ms[i0]` and `Ms[i0+1]`)

    Example
    -------
    Suppose `len(Ms) = 3` and `i0 = 1`
    ```
            |           |           |     
        -→- 0 ----→---- 1 → R → L → 2 -→-
        |---------------←---------------|
    ```
    """
    # length of the PBC-MPS
    n = len(Ms)
    # one round of QR
    R = gt.eye(Ms[i0].DE[-1], Ms[i0].DO[-1])
    shape_R = R.shape
    for i in range(i0+1, i0+1+n, 1):
        RM = (
            Ms[i % n] if i == i0+1 else
            gt.tensordot_keepform(R, Ms[i%n], [1,0], anchor='b')
        )
        R = gla.qr(RM, [-1], return_q=False)
        if normalize: R = _normalize_rl(R)
    # one round of LQ
    L = gt.eye(Ms[i0].DE[-1], Ms[i0].DO[-1])
    for i in range(i0, i0-n, -1):
        ML = (
            Ms[i % n] if i == i0 else
            gt.tensordot_keepform(Ms[i%n], L, [-1,0], anchor='a')
        )
        L = gla.lq(ML, [0], return_q=False)
        if normalize: L = _normalize_rl(L)
    assert R.shape == L.shape == shape_R
    return R, L


def loop_qr(
    Ms: list[GTensor], 
    minIter=MIN_ITER, maxIter=MAX_ITER, normalize=True
) -> tuple[list[GTensor], list[GTensor], tuple[int, float]]:
    """
    Perform QR/LQ by looping over PBC-MPS
    until convergence

    Virtual bonds in `Ms` should all be `<bra| --> |ket>`
    
    Returns
    -------
    Lists of matrices `Rs` and `Ls` on all bonds. 

    - `Rs[i]` is on the right of `M[i]`
    - `Ls[i]` is on the left of `M[i]`

    Example
    -------
    Suppose `len(Ms) = 3`
    ```
            |           |           |     
    --→ L0→ 0 →R0 → L1→ 1 →R1 → L2→ 2 →R2 --→
    |-------------------←-------------------|
    ```
    """
    assert minIter <= maxIter
    # assert all(M.parity == 0 for M in Ms)
    # initialize variables
    n = len(Ms)
    # create a series of identities
    Rs = [gt.eye(Ms[i].DE[-1], Ms[i].DO[-1]) for i in range(n)]
    Ls = [gt.eye(Ms[i].DE[0], Ms[i].DO[0]) for i in range(n)]
    
    diff_old = inf
    Rs_old, Ls_old = deepcopy(Rs), deepcopy(Ls)
    for itr in range(maxIter):
        for i1, i2 in zip(range(n), reversed(range(n))):
            # i1: 0, 1, ..., n-2, n-1   use QR
            # i2: n-1, n-2, ..., 1, 0   use LQ
            RM = gt.tensordot_keepform(
                Rs[i1-1], Ms[i1], [1, 0], anchor='b'
            )
            ML = gt.tensordot_keepform(
                Ms[i2], Ls[(i2+1)%n], [-1, 0], anchor='a'
            )
            Rs[i1] = gla.qr(RM, [-1], return_q=False)
            Ls[i2] = gla.lq(ML, [0], return_q=False)
            if normalize:
                Rs[i1] = _normalize_rl(Rs[i1])
                Ls[i2] = _normalize_rl(Ls[i2])
        # calculate difference and update Rs, Ls
        relDiffs = _check_qr_convergence(Rs, Ls, Rs_old, Ls_old, abs_mode=True)
        maxRelDiff = max(relDiffs)
        # update R, L
        Rs_old[:], Ls_old[:] = Rs[:], Ls[:]
        # Already converged
        if itr > minIter and maxRelDiff < EPS: break
        #
        if itr > minIter and maxRelDiff < TOL and maxRelDiff / diff_old > 0.5: break
        # update difference if not converged yet
        diff_old = maxRelDiff
    # when the for loop is not terminated by "break"
    # i.e. QR/LQ loop exceeds max iteration
    else:
        if maxIter > 10:
            warn("\nQR/LQ exceeds max iteration {}".format(maxIter))
    return Rs, Ls, (itr, maxRelDiff)


def get_projs(
    Ms: list[GTensor], Dmax: None | int = None, 
    eps=EPSS_DEFAULT, absorb_s=True, 
    minIter=MIN_ITER, maxIter=MAX_ITER, normalize=True
) -> tuple[
    list[GTensor], list[GTensor], 
    list[GTensor], list[float], tuple[int, float]
]:
    """
    Find projectors and singular value spectrum
    on each bond of an PBC-MPS

    Returns
    -------
    Pas, Pbs: list[GTensor]
        Projectors A and B on each bond
    s_cuts: list[GTensor]
        Truncated singular value spectrum on each bond
    relEs: list[float]
        SVD truncation error on each bond
    """
    assert minIter <= maxIter
    # error checking and regularize Dsmax
    _check_mps(Ms, bc="p")
    # actual computation
    Rs, Ls, qrlog = loop_qr(Ms, minIter, maxIter, normalize)
    n = len(Rs)
    assert len(Ms) == len(Ls) == n
    Pas    = [None] * n
    Pbs    = [None] * n
    s_cuts = [None] * n
    relEs  = [None] * n
    # on the ith bond between `M[i]` and `M[i+1]`
    for i in range(n):
        Pas[i], Pbs[(i+1)%n], s_cuts[i], relEs[i] \
        = proj_from_RL(
            Rs[i], Ls[(i+1)%n], Dmax, eps, absorb_s
        )
    # assert all projectors have even parity
    for t in Pas: assert t.parity == 0
    for t in Pbs: assert t.parity == 0
    return Pas, Pbs, s_cuts, relEs, qrlog


def canonicalize(
    Ms: list[GTensor], Dmax: None | int = None, 
    eps=EPSS_DEFAULT, absorb_s=True, 
    minIter=MIN_ITER, maxIter=MAX_ITER, normalize=True
):
    """
    Put the PBC-MPS `Ms` into "canonical" form 
    (although it is only well-defined for OBC-MPS). 
    Optionally truncate bond dimensions and change bond dual.

    For each site `i`, start from
    ```
    |       |
    i ---- i+1
    ```
    Find projectors `A`, `B` and apply to `M`:
    ```
    |                            |
    i → A[i]) - gNew - (B[i+1] → i+1
    ```
    
    Parameters
    -----------
    Ms: list[GTensor]
        MPS tensors. For each M, first/last axis 
        connects to the last/next tensor
    
    Dmax: int or None
        SVD max bond dimension
        (when None, no truncation will be made)
    """
    assert minIter <= maxIter
    # error checking and initialize Dsmax
    _check_mps(Ms, bc="p")
    # compute proper projectors
    Pas, Pbs, s_cuts, relEs, qrlog = get_projs(
        Ms, Dmax, eps, absorb_s, 
        minIter, maxIter, normalize
    )
    # apply projectors to Ms's
    Ms_new = [
        gt.tensordot_keepform(
            gt.tensordot_keepform(
                Pbs[i], M, [1, 0], anchor='b'
            ), Pas[i], [-1, 0], anchor='a'
        ) for i, M in enumerate(Ms)
    ]
    return Ms_new, s_cuts, relEs, qrlog


# @profile
def truncate(
    Ms: list[GTensor], bonds: list[int],
    Dmax: int|None, eps=EPSS_DEFAULT, absorb_s=True
):
    """
    Truncate only specified `bonds` on a PBC-MPS

    Parameters
    ----
    Ms: list[GTensor]
        the PBC-MPS (modified in place)
    bonds: list[int]
        bonds to be truncated. 
        contains integers from 0 to len(Ms)-1
        specifying after which M the bond is.
    """
    for b in bonds:
        assert 0 <= b < len(Ms), "Bond index out of range"
    # find projectors on the bonds to be truncated
    n = len(Ms)
    Pas: list[GTensor] = []
    Pbs: list[GTensor] = []
    scuts: list[GTensor] = []
    svdErrs: list[float] = []
    for b in bonds:
        # perform one round of QR/LQ starting from this bond
        R, L = loop_qr1(Ms, b, normalize=True)
        # find projector
        Pa, Pb, scut, svdErr = proj_from_RL(R, L, Dmax, eps, absorb_s)
        Pas.append(Pa)
        Pbs.append(Pb)
        scuts.append(scut)
        svdErrs.append(svdErr)
    # apply the projectors
    for b, Pa, Pb in zip(bonds, Pas, Pbs):
        Ms[b] = gt.tensordot_keepform(
            Ms[b], Pa, [-1, 0], anchor='a'
        )
        Ms[(b+1) % n] = gt.tensordot_keepform(
            Pb, Ms[(b+1) % n], [1, 0], anchor='b'
        )
    return scuts, svdErrs


def print_qrErr(qrlog: tuple[int, float], io: IO):
    itr, maxRelDiff = qrlog
    io.write('iter, maxRelDiff = {:d}, {:.2E}\n'.format(itr, maxRelDiff))
