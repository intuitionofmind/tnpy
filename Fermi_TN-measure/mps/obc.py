"""
mps_obc.py
======
OBC-MPS algoritms
"""

import numpy as np
import gtensor as gt
from gtensor import GTensor
import gtensor.linalg as gla
from typing import IO

EPSS_DEFAULT = gt.EPSS_DEFAULT
np.seterr(all='raise')


def _check_mps(Ms: list[GTensor], bc="p"):
    """
    Check if the MPS (`Ms`) is well defined

    - Virtual axis dimension of connected tensors should match
    - Virtual bonds in `Ms` should all be `<bra| --> |ket>`

    Parameters
    ----
    bc: str ('p' or 'o')
        specify MPS boundary condition (periodic or open)
    """
    assert bc == "p" or bc == "o"
    assert len(Ms) >= (1 if bc == "p" else 2)
    # axes
    for i in range(
        len(Ms) if bc == "p" else len(Ms) - 1
    ):
        m1, m2 = Ms[i], Ms[(i+1) % len(Ms)]
        message = f"unmatch after site {i}"
        assert m1.DS[-1] == m2.DS[0], f"MPS total dimension {message}"
        assert m1.DE[-1] == m2.DE[0], f"MPS even dimension {message}"
        assert m1.dual[-1] == 1, f"MPS dual {message}"
        assert m2.dual[0] == 0, f"MPS dual {message}"


def _normalize_rl(rl: GTensor) -> GTensor:
    """
    normalize R/L matrix obtained from QR/LQ decomposition

    Paramaters
    ----
    rl: GTensor
        the dual of `rl` is required to satisfy
        `|ket> -- R -- (any)` and `(any) -- L -- <bra|`
    simple: bool
        scheme of normalization
    """
    assert rl.ndim == 2 and rl.parity == 0
    assert rl.dual == (0,1)
    if rl.size == 0:
        return rl
    diagMax = gt.maxabs(rl)
    # round the number to avoid division underflow
    diagMax = round(diagMax, ndigits=10)
    return rl / diagMax


def open_qr(
    Ms: list[GTensor], normalize=False
) -> tuple[list[GTensor], list[GTensor]]:
    """
    Perform QR/LQ from two ends of OBC-MPS 
    to obtain matrices `R` and `L` on the each bond

    Parameters
    ----
    normalize: bool
        controls whether to normalize the obtained R, L matrices
        (by setting their maximum diagonal element to 1)

    Example
    -------
    Suppose `len(Ms) = 4`, and
    ```
        ↓       ↓       ↓       ↓
        0 --→-- 1 --→-- 2 --→-- 3
    ```
    
    The R/L tensor are labelled as
    ```
        ↓       ↓       ↓       ↓
        0 →R0 → 1 →R1 → 2 →R2 → 3
    
        ↓       ↓       ↓       ↓
        0 → L1→ 1 → L2→ 2 → L3→ 3
    ```
    (actually L1, L2, ... are saved as L0, L1, ...)
    """
    n = len(Ms)
    Rs, Ls = [None]*(n-1), [None]*n
    # QR
    for i1, i2 in zip(range(n-1), reversed(range(1,n))):
        RM = (
            Ms[i1] if i1 == 0 else
            gt.tensordot_keepform(Rs[i1-1], Ms[i1], [1,0], anchor='b')
        )
        Rs[i1] = gla.qr(RM, [-1], return_q=False)
        ML = (
            Ms[i2] if i2 == len(Ms)-1 else
            gt.tensordot_keepform(
                Ms[i2], Ls[i2+1], [-1, 0], anchor='a'
            )
        )
        Ls[i2] = gla.lq(ML, [0], return_q=False)
        if normalize:
            Rs[i1] = _normalize_rl(Rs[i1])
            Ls[i2] = _normalize_rl(Ls[i2])
    # remove None placeholder in Ls
    Ls.pop(0)
    return Rs, Ls


def proj_from_RL(
    r: GTensor, l: GTensor, Dmax: None | int, 
    eps=EPSS_DEFAULT, absorb_s=True, _self_test=False
):
    """
    Construct projectors from `R`,`L` 
    obtained from QR/LQ decomposition.

    Parameters
    ----
    Dmax: None or int
        maximum bond dimension to be kept by projectors

    Returns
    -------
    - Projectors `Pa` and `Pb`, 
    - singular values `s` from SVD of `R * L`
    - SVD truncation error
    """
    # u * s * vh = R * L
    u, s, vh = gla.svd(gt.tensordot(r, l, [1,0]), 1)
    u, s_cut, vh = gla.svd_cutoff(u, s, vh, Dmax, eps=eps)
    d_cut = (s_cut.DE[0], s_cut.DO[0])
    assert u.dual == s_cut.dual == vh.dual == (0,1)
    
    # create projectors
    uh, v = u.gconj(), vh.gconj()
    s_inv = gla.matrix_inv(s_cut, is_diag=True)
    # when s is not absorbed into Pa, Pb
    
    # Pa = L * v * 1/s 
    Pa = gt.dot_diag(gt.tensordot(l, v, (1,0)), s_inv, (1,0))
    # Pb = 1/s * uh * R
    Pb = gt.dot_diag(gt.tensordot(uh, r, (1,0)), s_inv, (0,1))
    assert Pa.dual == Pb.dual == (0,1)
    
    if absorb_s:
        # split s_cut to square roots
        s1, s2 = gla.matrix_sqrt(s_cut, is_diag=True)
        # absorb sqrt(s) into Pa, Pb
        Pa = gt.dot_diag(Pa, s1, (1,0))
        Pb = gt.dot_diag(Pb, s2, (0,1))
    
    if _self_test:
        # verify Pa * Pb = 1 when there is no svd cutoff
        assert Dmax is None
        iden = (
            gt.tensordot(Pa, Pb, (1,0)) if absorb_s
            else gt.tensordot(Pa.dot_diag(s, (1,0)), Pb, (1,0))
        )
        assert gt.is_identity_matrix(iden)
    # calculate and output (relative) truncation error
    relE = gla.gsvd_error(s, d_cut)
    return Pa, Pb, s_cut, relE


def get_projs(
    Ms: list[GTensor], Dmax: None | int = None, 
    eps=EPSS_DEFAULT, absorb_s=True, normalize=False
) -> tuple[
    list[GTensor], list[GTensor], 
    list[GTensor], list[float]
]:
    """
    Find projectors and singular value spectrum
    on each bond of an OBC-MPS

    Returns
    -------
    Pas, Pbs: tuple[GTensor, ...]
        Projectors A and B on each bond
    s_cuts: tuple[GTensor, ...]
        Truncated singular value spectrum on each bond
    relEs: tuple[float, ...]
        SVD truncation error on each bond
    """
    _check_mps(Ms, bc="o")
    # r and l matrices on each bond
    Rs, Ls = open_qr(Ms, normalize)
    n = len(Ms)
    Pas    = [None] * (n-1)
    Pbs    = [None] * (n-1)
    s_cuts = [None] * (n-1)
    relEs  = [None] * (n-1)
    # get projectors Pa, Pb and SV spectrum on each bond
    for i, (R, L) in enumerate(zip(Rs, Ls)): 
        Pas[i], Pbs[i], s_cuts[i], relEs[i] \
        = proj_from_RL(R, L, Dmax, eps, absorb_s)
    # assert all projectors have even parity
    for t in Pas: assert t.parity == 0
    for t in Pbs: assert t.parity == 0
    return Pas, Pbs, s_cuts, relEs


def canonicalize(
    Ms: list[GTensor], Dmax: None | int = None, 
    eps=EPSS_DEFAULT, absorb_s=True, normalize=False
):
    """
    Canonicalize an OBC-MPS `Ms` by finding projectors
    `As`, `Bs` applying to `Ms`:

        |                       |
        i → A) -- gNew -- (B → i+1
    
    Parameters
    -----------
    Ms: list[GTensor]
        MPS tensors. For each M, first/last axis 
        connects to the last/next tensor
    
    Dmax: int or None
        SVD max bond dimension
        (when None, no truncation will be made)
    """
    Pas, Pbs, s_cuts, relEs = get_projs(
        Ms, Dmax, eps, absorb_s, normalize
    )
    Ms_new = [
        gt.tensordot_keepform(
            M, Pas[i], [-1, 0], anchor='a'
        ) if i == 0
        else gt.tensordot_keepform(
            Pbs[i-1], M, [1, 0], anchor='b'
        ) if i == len(Ms)-1
        else gt.tensordot_keepform(
            gt.tensordot_keepform(
                Pbs[i-1], M, [1, 0], anchor='b'
            ), Pas[i], [-1, 0], anchor='a'
        )
        for i, M in enumerate(Ms)
    ]
    return Ms_new, s_cuts, relEs


def print_svdErrs(svdErrs: list[float], io: IO):
    numStr = ''
    for err in svdErrs:
        numStr += '{:>10.2E}'.format(err)
    io.write('relative cut error: {}\n'.format(numStr))
