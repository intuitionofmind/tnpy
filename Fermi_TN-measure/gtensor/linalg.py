"""
Linear algebra functions for 
fermionic (Grassmann) tensors
"""

import torch
from torch import Tensor
import torch.linalg as tla
import scipy.linalg as spla
import utils
from .tools import matshape_from_block, regularize_axes
from . import core
from .core import GTensor
from .core import _DEBUG, EPSS_DEFAULT, _process_contract_axes
from gtensor.tools import find_remain_axis

def is_unitary(
    u: GTensor, map_axes: list[int], 
    direction = "l"
):
    """
    Check if `u` is left/right unitary

    Parameters
    ----
    map_axes: list[int]
        axes of linear map `u` that corresponds to output (co-domain)
        and can be further acted on by a linear map
    direction: str ("l", "r", "lr")
        check left or right unitarity
    """
    if direction in ("l", "lr"):
        # check left unitarity u+ u = 1
        uu = core.around(
            core.tensordot(u.gT, u.pconj(map_axes), [map_axes] * 2), 
            decimals=8
        )
        assert uu.ndim == 2
        n0, n1 = uu.DE[0], uu.DO[1]
        flagL = core.allclose(uu, core.eye(n0, n1, uu.dual))
    elif direction in ("r", "lr"):
        # check right unitarity u u+ = 1
        map_axes2 = find_remain_axis(map_axes, u.ndim)
        uu = core.around(
            core.tensordot(u, u.gT.pconj(map_axes2), [map_axes2] * 2)
        )
        assert uu.ndim == 2
        n0, n1 = uu.DE[0], uu.DO[1]
        flagR = core.allclose(uu, core.eye(n0, n1, uu.dual))
    else:
        raise ValueError("Unrecognized unitarity direction")
    flag = (
        flagR if direction == "r" else
        flagL if direction == "l" else
        flagL and flagR
    )
    return flag


def expm(a: GTensor):
    """Exponentiate even-parity Grassmann matrix `a`"""
    assert a.ndim == 2 and a.dual == (0, 1) and a.parity == 0
    for par in range(2):
        assert a.shape[par][0] == a.shape[par][1]
    b = core.empty_like(a)
    b.blocks[(0, 0)] = tla.matrix_exp(a.blocks[(0, 0)])
    b.blocks[(1, 1)] = tla.matrix_exp(a.blocks[(1, 1)])
    return b


# ------ Inverse matrix ------


def matrix_inv(t: GTensor, is_diag=False, pseudo=False):
    """
    Find the inverse (or pseudo inverse) of a Grassmann matrix

    Parameters
    ----
    t: GTensor
        The Grassmann square matrix whose inverse is to be calculated
    is_diag: bool
        Specify if `t` is a diagonal matrix
    """
    assert t.ndim == 2, "`t` should be a matrix"
    if t.parity == 0:
        assert all(
            t.shape[p][0] == t.shape[p][1] for p in range(2)
        ), "Even parity `t` should be a *square* matrix"
    else:
        assert t.shape[0][0] == t.shape[1][1] and t.shape[0][1] == t.shape[1][0]
    d1, d2 = t.dual
    dual_inv = (1 - d2, 1 - d1)
    if is_diag:
        assert t.parity == 0, "Diagonal matrix must have even parity"
    blocks = {}
    if t.parity == 0:
        if is_diag:
            blocks[(0, 0)] = torch.diag(1.0 / torch.diagonal(t.blocks[(0, 0)]))
            blocks[(1, 1)] = torch.diag(1.0 / torch.diagonal(t.blocks[(1, 1)]))
        else:
            blocks[(0, 0)] = (
                tla.pinv(t.blocks[(0, 0)]) if pseudo else tla.inv(t.blocks[(0, 0)])
            )
            blocks[(1, 1)] = (
                tla.pinv(t.blocks[(1, 1)]) if pseudo else tla.inv(t.blocks[(1, 1)])
            )
        extra_sign = (-1 if d1 == 1 else 1) * (-1 if d2 == 0 else 1)
        if extra_sign == -1:
            blocks[(1, 1)] *= -1
    else:
        blocks[(0, 1)] = (
            tla.pinv(t.blocks[(1, 0)]) if pseudo else tla.inv(t.blocks[(1, 0)])
        ) * (-1 if d1 == 1 else 1)
        blocks[(1, 0)] = (
            tla.pinv(t.blocks[(0, 1)]) if pseudo else tla.inv(t.blocks[(0, 1)])
        ) * (-1 if d2 == 0 else 1)
    tinv_shape = matshape_from_block(blocks)
    tinv = GTensor(tinv_shape, dual_inv, blocks=blocks)
    return tinv


def matrix_pinv(t: GTensor, is_diag=False):
    """
    Find the pseudo inverse of a Grassmann matrix
    """
    return matrix_inv(t, is_diag, pseudo=True)


# ------ Tensor Decomposition ------


def matrix_sqrt(t: GTensor, is_diag=False):
    """
    Get square root of rank-2 square
    diagonal and non-negative GTensor
    ```
    t = g1 <b| -> |k> g2
    ```
    """
    assert t.ndim == 2
    dual1 = (t.dual[0], 1)
    dual2 = (0, t.dual[1])
    # currently, only diagonal (even parity)
    # and non-negative `t` can be handled
    g1 = core.empty(t.shape, dual1, parity=t.parity)
    g2 = core.empty(t.shape, dual2, parity=t.parity)
    if is_diag:
        assert t.parity == 0
        for key, block in t.blocks.items():
            sqrt_block = torch.diag(torch.sqrt(torch.diagonal(block)))
            g1.blocks[key] = sqrt_block
            g2.blocks[key] = sqrt_block
    else:
        raise NotImplementedError
    if _DEBUG:
        g1.verify()
        g2.verify()
    return g1, g2


def _matrix_polar(a: GTensor, side="right", odd="u"):
    """
    Polar decomposition of Grassmann matrices
    (done separately in the two nonzero blocks)

    Parameters
    ----
    side: str ("right" or "left")
        - "right":  A = U <b| -> |k> P
        - "left":   A = P <b| -> |k> U

    odd: str ("p" or "u")
        specify which one of `p` or `u`
        should be odd when `a` is odd

    Returns
    ----
    g1, g2: GTensors
        - when side == "right": (g1, g2) = (u, p)
        - when side == "left":  (g1, g2) = (p, u)
    """
    assert a.ndim == 2
    assert odd in ("p", "u")
    assert side in ("right", "left")
    if a.parity == 0:
        gIdx1 = {(0, 0): (0, 0), (1, 1): (1, 1)}
        gIdx2 = {(0, 0): (0, 0), (1, 1): (1, 1)}
    else:
        # the 1st tensor has odd parity
        if (odd == "u" and side == "right") or (odd == "p" and side == "left"):
            gIdx1 = {(0, 1): (0, 1), (1, 0): (1, 0)}
            gIdx2 = {(0, 1): (1, 1), (1, 0): (0, 0)}
        # the 2nd tensor has odd parity
        else:
            gIdx1 = {(0, 1): (0, 0), (1, 0): (1, 1)}
            gIdx2 = {(0, 1): (0, 1), (1, 0): (1, 0)}
    # polar decomposition in each nonzero block of `a`
    block1, block2 = {}, {}
    for gIdx, block in a.blocks.items():
        u0, p0 = utils.linalg.polar(block, side=side)
        block1[gIdx1[gIdx]], block2[gIdx2[gIdx]] = (
            (u0, p0) if side == "right" else (p0, u0)
        )
    # create U and P matrix
    # when side == "right": (g1, g2) = (u, p)
    # when side == "left":  (g1, g2) = (p, u)
    dual1 = (a.dual[0], 1)
    shape1 = matshape_from_block(block1)
    g1 = GTensor(shape1, dual1, blocks=block1)
    dual2 = (0, a.dual[1])
    shape2 = matshape_from_block(block2)
    g2 = GTensor(shape2, dual2, blocks=block2)
    if _DEBUG:
        g1.verify()
        g2.verify()
    return g1, g2


def polar(
    a: GTensor, nAxis: int = 1, side="right", odd="u"
) -> tuple[GTensor, GTensor]:
    """
    Polar decomposition of Grassmann tensors
    (done separately in the two nonzero blocks)

    Parameters
    ----
    nAxis: int
        the number of axis to be put in the
        1st tensor produced by the decomposition

    side: str ("right" or "left")
        - "right":  A = U P
        - "left":   A = P U

    typef: int (0 or 1)
        When side == "right"
        - 0: A = U <b| -> |k> P
        - 1: A = P |k> -> <b| U

    odd: str ("p" or "u")
        specify which one of `p` or `u`
        should be odd when `a` is odd

    verify_dual: bool
        Ensure unitarity of `U`
        (default is `True`, contrary to other decompositions)

    Returns
    ----
    g1, g2: GTensors
        - when side == "right": (g1, g2) = (u, p)
        - when side == "left":  (g1, g2) = (p, u)
    """
    if not (nAxis % 1 == 0 and 0 < nAxis < a.ndim):
        raise ValueError("`nAxis` must be greater than 0 and smaller than a.ndim")
    # duals of decomposition result
    dual1 = tuple(a.dual[ax] for ax in range(nAxis))
    dual2 = tuple(a.dual[ax] for ax in range(nAxis, a.ndim))
    # merge axes
    # the dual of merged axes of P is unimportant
    mat = a.merge_axes(
        (nAxis, a.ndim - nAxis),
        dualMerge=((dual1[0], 0) if side == "right" else (0, dual2[0])),
        auto_dual=False,
    )
    g1, g2 = _matrix_polar(mat, side, odd)
    # dimension of the new axis
    dimj = (g2.shape[0][0], g2.shape[1][0])
    # recover tensor shape
    shape1 = tuple(a.shape[par][0:nAxis] + (dimj[par],) for par in range(2))
    g1 = g1.split_axes(
        (nAxis, 1), shape1, dualSplit=dual1 + (g1.dual[1],), auto_dual=False
    )
    shape2 = tuple((dimj[par],) + a.shape[par][nAxis::] for par in range(2))
    g2 = g2.split_axes(
        (1, a.ndim - nAxis), shape2, dualSplit=(g2.dual[0],) + dual2, auto_dual=False
    )
    if _DEBUG:
        g1.verify()
        g2.verify()
    return g1, g2


def _matrix_svd(a: GTensor, odd="vh"):
    """
    Singular value decomposition of Grassmann matrices
    ```
        a[i,j] = u[i,x] s[x,y] vh[y,j]
    ```

    Dual of new axes
    ```
        A = U <b| -> |k> s <b| -> |k> Vh
    ```

    Returns
    ----
    u, s, vh: GTensors
        SVD unitary matrices (basis change) and the singular values
        `s` is always diagonal with even parity

    Parity of U and Vh
    ---
    - P(a) = 0 --> P(u) = 0, P(vh) = 0
    - P(a) = 1 --> P(u) = 0, P(vh) = 1
    """
    assert a.ndim == 2
    assert odd in ("u", "vh")
    # SVD of each nonzero block of `a`
    if a.parity == 0:
        s_key = {(0, 0): 0, (1, 1): 1}
        gIdxU = {(0, 0): (0, 0), (1, 1): (1, 1)}
        gIdxVh = {(0, 0): (0, 0), (1, 1): (1, 1)}
    else:
        if odd == "u":
            s_key = {(0, 1): 1, (1, 0): 0}
            gIdxU = {(0, 1): (0, 1), (1, 0): (1, 0)}
            gIdxVh = {(0, 1): (1, 1), (1, 0): (0, 0)}
        elif odd == "vh":
            s_key = {(0, 1): 0, (1, 0): 1}
            gIdxU = {(0, 1): (0, 0), (1, 0): (1, 1)}
            gIdxVh = {(0, 1): (0, 1), (1, 0): (1, 0)}
    blockU, sdiag, blockVh = {}, {}, {}
    # perfrom svd in the two blocks
    for gIdx, block in a.blocks.items():
        try:
            blockU[gIdxU[gIdx]], sdiag[s_key[gIdx]], blockVh[gIdxVh[gIdx]] \
            = tla.svd(block, full_matrices=False)
        except:
            bu, bs, bvh = spla.svd(block, full_matrices=False, lapack_driver="gesvd")
            blockU[gIdxU[gIdx]], sdiag[s_key[gIdx]], blockVh[gIdxVh[gIdx]] \
            = list(map(torch.from_numpy, (bu, bs, bvh)))
        sdiag[s_key[gIdx]] = sdiag[s_key[gIdx]].to(dtype=block.dtype)
    dualU, dualS, dualVh = (a.dual[0], 1), (0, 1), (0, a.dual[1])
    shapeU = matshape_from_block(blockU)
    u = GTensor(shapeU, dualU, blocks=blockU)
    shapeVh = matshape_from_block(blockVh)
    vh = GTensor(shapeVh, dualVh, blocks=blockVh)
    s = core.diag(sdiag, dualS)
    if _DEBUG:
        u.verify()
        s.verify()
        vh.verify()
    return u, s, vh


def absorb_sv(u: GTensor, s: GTensor, vh: GTensor):
    """
    Absorb singular value matrix `s` into `u` and `vh`
    ```
    s1 = u * sqrt(s), s2 = sqrt(s) * vh
    ```

    It is assumed that `u[-1]` contracts with `s[0]`
    and `s[1]` contracts with `vh[0]`

    Square root of `s`
    ----
    `s = sqrt(s) <b| --> |k> sqrt(s)`
    """
    # square root of weight
    sqrts1, sqrts2 = matrix_sqrt(s, is_diag=True)
    s1 = u.dot_diag(sqrts1, [u.ndim - 1, 0])
    s2 = vh.dot_diag(sqrts2, [0, 1])
    return s1, s2


def _get_sdim(s: dict[int, Tensor]):
    """get the dimension of the singular value spectrum"""
    return (s[0].numel(), s[1].numel())


# @profile
def svd(
    a: GTensor, nAxis: int, cutoff=False, 
    Dmax: None | int = None, De: None | int = None, 
    eps=EPSS_DEFAULT, absorb_s=False, odd="vh"
):
    """
    Singular value decomposition
    ```
    a[0,...,n-1]
    = u[0,...,nAxis-1, x] s[x,y] vh[y, nAxis,...,n-1]
    = s1[0,...,nAxis-1, x] s2[x, nAxis,...,n-1]
    ```

    When `absorb_s` is True
    ```
    A = s1 <b| --> |k> s2
    ```

    Parameters
    ----
    cutoff: bool
        - when True, truncate the singular value spectrum
        - when False, no approximations are made

    Dmax: int or None
        maximum total number of singular values to be kept

    De: int or None
        when a int value is provided, hard truncation will be performed
        (i.e. keep Dmax singular values, with De in even sector)

    eps: float (default 1e-15)
        singular values smaller than `max(s) * eps` will be discarded

    absorb_s: bool
        - when True, return `s1 = U*sqrt(s), s, s2 = sqrt(s)*Vh`
        - when False, return `U, s, Vh`

    odd: str ("vh" or "u")
        when `a.parity == 1`, specify whether `vh` or `u`
        should have odd parity
    """
    if not (nAxis % 1 == 0 and 0 < nAxis < a.ndim):
        raise ValueError("`nAxis` must be greater than 0 and smaller than a.ndim")
    # assert that axes belonging to the same group have the same dual
    mat = a.merge_axes(
        (nAxis, a.ndim - nAxis), dualMerge=(0, 1), auto_dual=False
    )
    # SVD of each nonzero block of `mat`
    u, s, vh = _matrix_svd(mat, odd=odd)
    # shape of new axis
    dimj = (s.DE[0], s.DO[1])
    # recover tensor shape
    shapeU = tuple(
        a.shape[par][0:nAxis] + (dimj[par],) 
        for par in range(2)
    )
    dualU = a.dual[0:nAxis] + (u.dual[1],)
    shapeVh = tuple(
        (dimj[par],) + a.shape[par][nAxis::] 
        for par in range(2)
    )
    dualVh = (vh.dual[0],) + a.dual[nAxis::] 
    u = u.split_axes((nAxis, 1), shapeU, dualU, auto_dual=False)
    vh = vh.split_axes((1, a.ndim - nAxis), shapeVh, dualVh, auto_dual=False)
    # singular value spectrum cutoff
    if cutoff is True:
        u, s, vh = svd_cutoff(u, s, vh, Dmax, De, eps)
    # absorb s into u and vh
    if absorb_s is True:
        u, vh = absorb_sv(u, s, vh)
    if _DEBUG:
        u.verify()
        s.verify()
        vh.verify()
    return u, s, vh


def s_cutoff(
    s: dict[int, Tensor], Dmax: None | int, 
    De: None | int = None, eps=EPSS_DEFAULT
) -> dict[int, Tensor]:
    """
    Dynamical truncation of the singular value spectrum

    s: dict[int, Tensor]
        The singular value spectrum (diagonal elements of s matrix)
    Dmax: None or int
        Maximum total number of kept singular values
    De: None or int
        Maximum number of kept singular values in even sector
    """
    # do nothing on empty input
    if s[0].numel() == 0 and s[1].numel() == 0:
        return s
    smax = (
        max(s[0][0].real, s[1][0].real)
        if s[0].numel() > 0 and s[1].numel() > 0
        else s[0][0].real if s[0].numel() > 0 and s[1].numel() == 0 else s[1][0].real
    )
    # step 1: remove singular values smaller than eps * smax
    scut = dict((p, s[p][s[p].real > eps * smax]) for p in range(2))
    # step 2: keep at most Dmax largest singular values, 
    # with at most De singular values in even sector
    dims = _get_sdim(scut)
    if (Dmax is not None) and (sum(dims) > Dmax):
        ids = torch.argsort(
            torch.cat([scut[0], scut[1]]).real, 
            descending=True
        )[0:Dmax:]
        # elements of s[0] to be kept
        id0s = ids[ids < dims[0]]
        if De is not None: 
            id0s = id0s[0:De]
        scut[0] = scut[0][id0s]
        # elements of s[1] to be kept
        id1s = ids[ids >= dims[0]] - dims[0]
        scut[1] = scut[1][id1s]
    return scut


def svd_cutoff(
    u: GTensor, s: GTensor, vh: GTensor, 
    Dmax: None | int, De: None | int = None, eps=EPSS_DEFAULT
):
    """
    truncate Grassmann SVD singular values
    (even and odd sectors are sorted together)

    Parameters
    ----
    u, s, vh: GTensor
        SVD results. Input can also be `s1, s, s2`
    Dmax: None or int
        maximum total (even + odd) number of kept singular values.
        To avoid truncation in this way, set `Dmax = None`.
    eps: float
        singular values smaller than `max(s) * eps` will be discarded.
        To avoid truncation in this way, set `eps = 0`.
    """
    if De is not None:
        assert Dmax is not None
    scut_diag = s_cutoff(s.diagonal(), Dmax, De, eps)
    d_cut = _get_sdim(scut_diag)
    s_cut = core.diag(scut_diag, s.dual)

    # truncate last axis of U
    ushape_cut = tuple(tuple(
        u.shape[par][i] if i != u.ndim - 1 
        else d_cut[par] for i in range(u.ndim)
    ) for par in range(2))
    u_cut = core.empty(ushape_cut, u.dual, u.parity)
    for gIdx, u_block in u.blocks.items():
        newshape = list(u_block.shape)
        newshape[-1] = d_cut[gIdx[-1]]
        cut = tuple(slice(0, newshape[i]) for i in range(u_cut.ndim))
        u_cut.blocks[gIdx] = u_block[cut]

    # truncate first axis of Vh
    vhshape_cut = tuple(tuple(
        vh.shape[par][i] if i != 0 
        else d_cut[par] for i in range(vh.ndim)
    ) for par in range(2))
    vh_cut = core.empty(vhshape_cut, vh.dual, vh.parity)
    for gIdx, vh_block in vh.blocks.items():
        vh_cut.blocks[gIdx] = vh_block[0 : d_cut[gIdx[0]]]
    return u_cut, s_cut, vh_cut


def gsvd_error(s: GTensor, Dmax: tuple[int, int]) -> float:
    """
    Calculate the truncation error of Grassmann SVD
    using the spectrum of singular values

    d_cut: tuple[int, int]
        kept number of singular values in even and odd sectors
    """
    relE = 0.0
    sdiag = s.diagonal(real=True)
    # sometimes small negative singular values
    # will appear due to numerical errors.
    # here we change them to 0
    for p in range(2):
        sdiag[p] = torch.clamp(sdiag[p], min=0)
    if Dmax[0] == sdiag[0].numel() and Dmax[1] == sdiag[1].numel():
        pass
    elif Dmax[0] == sdiag[0].numel() and Dmax[1] != sdiag[1].numel():
        if Dmax[0] != 0:
            relE = torch.sqrt(
                torch.norm(sdiag[1][Dmax[1] :]) ** 2
                / sum(torch.norm(sdiag[p]) ** 2 for p in range(2))
            )
        else:
            relE = torch.sqrt(
                torch.norm(sdiag[1][Dmax[1] :]) ** 2 / torch.norm(sdiag[1]) ** 2
            )
    elif Dmax[0] != sdiag[0].numel() and Dmax[1] == sdiag[1].numel():
        if Dmax[1] != 0:
            relE = torch.sqrt(
                torch.norm(sdiag[0][Dmax[0] :]) ** 2
                / sum(torch.norm(sdiag[p]) ** 2 for p in range(2))
            )
        else:
            relE = torch.sqrt(
                torch.norm(sdiag[0][Dmax[0] :]) ** 2 / torch.norm(sdiag[0]) ** 2
            )
    else:
        relE = torch.sqrt(
            sum(torch.norm(sdiag[p][Dmax[p] :]) ** 2 for p in range(2))
            / sum(torch.norm(sdiag[p]) ** 2 for p in range(2))
        )
    return relE


def _matrix_qr(
    a: GTensor, return_q=True, odd="q", posR=False
) -> tuple[GTensor, GTensor] | GTensor:
    """
    QR decomposition of Grassmann matrices
    (done separately in the two nonzero blocks)

    Parameters
    ----
    typef: int (0 or 1)
        When side == "right"
        - 0: A = Q <b| -> |k> R
        - 1: A = Q |k> -> <b| R

    odd: str ("q" or "r")
        specify which one of `q` or `r`
        should be odd when `a` is odd
    """
    assert a.ndim == 2
    assert odd in ("q", "r")
    # determine shape and parity of Q and R
    # and gIdx of nonzero blocks of Q, R
    if a.parity == 0:
        gIdxQ = {(0, 0): (0, 0), (1, 1): (1, 1)}
        gIdxR = {(0, 0): (0, 0), (1, 1): (1, 1)}
    else:
        # choose Q with odd parity
        if odd == "q":
            gIdxQ = {(0, 1): (0, 1), (1, 0): (1, 0)}
            gIdxR = {(0, 1): (1, 1), (1, 0): (0, 0)}
        else:
            gIdxQ = {(0, 1): (0, 0), (1, 0): (1, 1)}
            gIdxR = {(0, 1): (0, 1), (1, 0): (1, 0)}
    # QR decomposition of each nonzero block of `a`
    blockQ, blockR = {}, {}
    for gIdx, block in a.blocks.items():
        if return_q:
            blockQ[gIdxQ[gIdx]], blockR[gIdxR[gIdx]] \
                = utils.linalg.qr(block, "reduced", posR)
        else:
            _, blockR[gIdxR[gIdx]] \
                = utils.linalg.qr(block, "r", posR)
    # create Q matrix
    if return_q:
        dual1 = (a.dual[0], 1)
        shapeQ = matshape_from_block(blockQ)
        q = GTensor(shapeQ, dual1, blocks=blockQ)
    # create R matrix
    dual2 = (0, a.dual[1])
    shapeR = matshape_from_block(blockR)
    r = GTensor(shapeR, dual2, blocks=blockR)
    # verification
    if _DEBUG:
        if return_q:
            q.verify()
        r.verify()
    if return_q:
        return q, r
    else:
        return r


def qr(
    a: GTensor, axesR: list[int], 
    return_q=True, odd="q", posR=False
) -> tuple[GTensor, GTensor] | GTensor:
    """
    QR decomposition
    (`axes` will be transferred to the rear of `r`)
    ```
        a[0, ..., n, *axesR] = q[0, ..., n, x] -- r[x, *axesR]
    ```

    Dual of the new axis
    ```
        A = Q <b| --> |k> R
    ```
    """
    # check validity of `axesR`
    assert (
        0 < len(axesR) < a.ndim
    ), "The number of axes moved to R must be in (0, a.ndim)"
    axesR = regularize_axes(axesR, a.ndim)
    # axes kept in Q
    axesQ = tuple(i for i in range(a.ndim) if i not in axesR)
    dualsQ = tuple(a.dual[ax] for ax in axesQ)
    # assert that axes remaining in Q have the same dual
    dualQ = dualsQ[0]
    # dual of old axes in R
    dualsR = tuple(a.dual[ax] for ax in axesR)
    # put *axes to the rightmost
    perm = tuple(i for i in range(a.ndim) if i not in axesR) + tuple(axesR)
    a = a.transpose(*perm)
    # merge axes
    # the dual of merged axes of R is unimportant
    # and can be chosen as 0 or 1 arbitrarily
    mat = a.merge_axes(
        (len(axesQ), len(axesR)), 
        dualMerge=(dualQ, 0), auto_dual=False
    )
    # QR decomposition of nonzero blocks of `mat`
    if return_q is True:
        q, r = _matrix_qr(mat, return_q, odd, posR)
    else:
        r = _matrix_qr(mat, return_q, odd, posR)
    # dimension of the new axis
    dimj = (r.shape[0][0], r.shape[1][0])
    # recover tensor shape
    if return_q:
        shapeQ = tuple(a.shape[par][0 : len(axesQ)] + (dimj[par],) for par in range(2))
        q = q.split_axes(
            (len(axesQ), 1), shapeQ, dualSplit=dualsQ + (q.dual[1],), auto_dual=False
        )
    shapeR = tuple((dimj[par],) + a.shape[par][len(axesQ) : :] for par in range(2))
    r = r.split_axes(
        (1, len(axesR)), shapeR, dualSplit=(r.dual[0],) + dualsR, auto_dual=False
    )
    if _DEBUG:
        if return_q:
            q.verify()
        r.verify()
    if return_q:
        return q, r
    else:
        return r


def lq(
    a: GTensor, axesL: list[int], 
    return_q=True, odd="q", posL=True
) -> tuple[GTensor, GTensor] | GTensor:
    """
    LQ/RQ decomposition
    (`axes` will be transferred to the front of `l`)
    ```
        a[*axes, 0, ..., n] = l[*axes, x] q[x,0, ..., n]
    ```

    Dual of the new axis
    ```
        A = L <b| --> |k> Q
    ```
    """
    assert odd in ("q", "l")
    if odd == "l":
        odd = "r"
    # assert that axes remaining in Q have the same dual
    if return_q is True:
        q, l = qr(a, axesL, return_q, odd, posL)
        q, l = core.flip2_dual(q, l, [-1,0], flip="b")
        l = l.transpose(*(list(range(1, l.ndim)) + [0]))
        q = q.transpose(*([q.ndim - 1] + list(range(q.ndim - 1))))
        return l, q
    elif return_q is False:
        l = qr(a, axesL, return_q, odd, posL)
        l = core.flip_dual(l, 0)
        l = l.transpose(*(list(range(1, l.ndim)) + [0]))
        return l


# ------ Linear Equation Solver ------


def _matrix_solve(a: GTensor, b: GTensor, _lstsq=False):
    """
    Solve the standard equation
    ```
    b = tensordot(a, x, (1,0))
    (any) b (any) = (any) a <b| --> |k> x (any)
    ```
    """
    assert a.dual[1] == 1
    assert a.ndim == 2 and b.ndim == 2
    # create x matrix
    x_shape = tuple((a.shape[p][1], b.shape[p][1]) for p in range(2))
    x_dual = (0, b.dual[1])
    x = core.zeros(x_shape, x_dual, (a.parity + b.parity) % 2)
    # solve from the following blocks of a and b
    if a.parity == 0 and b.parity == 0:
        gIdxA = {(0, 0): (0, 0), (1, 1): (1, 1)}
        gIdxB = {(0, 0): (0, 0), (1, 1): (1, 1)}
    elif a.parity == 1 and b.parity == 1:
        gIdxA = {(0, 0): (1, 0), (1, 1): (0, 1)}
        gIdxB = {(0, 0): (1, 0), (1, 1): (0, 1)}
    elif a.parity == 0 and b.parity == 1:
        gIdxA = {(0, 1): (0, 0), (1, 0): (1, 1)}
        gIdxB = {(0, 1): (0, 1), (1, 0): (1, 0)}
    elif a.parity == 1 and b.parity == 0:
        gIdxA = {(0, 1): (1, 0), (1, 0): (0, 1)}
        gIdxB = {(0, 1): (1, 1), (1, 0): (0, 0)}
    else:
        raise ValueError("Parity of A or B is invalid")
    for gIdx in x.blocks.keys():
        x.blocks[gIdx] = (
            torch.linalg.lstsq(a.blocks[gIdxA[gIdx]], b.blocks[gIdxB[gIdx]])[0]
            if _lstsq
            else torch.linalg.solve(a.blocks[gIdxA[gIdx]], b.blocks[gIdxB[gIdx]])
        )
    return x


def solve(a: GTensor, b: GTensor, axes, _lstsq=False):
    """Find GTensor `x`, such that `b == tensordot(a, x, axes)`"""
    assert a.ndim >= 2 and b.ndim >= 2
    # the number of axes of x
    try:
        ndimX = b.ndim - a.ndim + 2 * len(axes[0])
    except TypeError:
        assert len(axes) == 2
        ndimX = b.ndim - a.ndim + 2
    axisA, axisX, _ = _process_contract_axes(axes, a.ndim, ndimX)
    # free axis of A
    free_axisA = [ax for ax in range(a.ndim) if ax not in axisA]
    # dual and shape of free axes of `a` should match those of `b`
    for ax_b, ax_a in enumerate(free_axisA):
        assert (
            b.dual[ax_b] == a.dual[ax_a]
        ), "Dual of `b` inconsistent with those of `a`"
        assert all(
            b.shape[p][ax_b] == a.shape[p][ax_a] for p in range(2)
        ), "Shape of `b` inconsistent with those of `a`"
    # convert A to matrix
    # put axes to be contracted in A to its rear (normal order)
    permA = [i for i in range(a.ndim) if i not in axisA] + axisA
    a = a.transpose(*permA)
    a_dual_tmp = a.dual
    # flip dual for a, so that axes contracted with x all have dual = 1
    a = core.flip_dual(
        a, axes=[ax for ax in range(len(free_axisA), a.ndim) if a.dual[ax] == 0]
    )
    # reshape a to matrix (order = -1 for the second axis)
    matA = a.merge_axes(
        (a.ndim - len(axisA), len(axisA)), 
        (0, 1), order=(1, -1), auto_dual=False
    )
    # convert b to matrix
    matB = b.merge_axes(
        ((a.ndim - len(axisA), b.ndim - a.ndim + len(axisA))),
        (1, 0), order=(1, 1), auto_dual=False
    )
    # solve for x
    x = _matrix_solve(matA, matB, _lstsq)
    # recover tensor shape for x
    x_shape = tuple(
        a.shape[p][len(free_axisA) : :] + b.shape[p][len(free_axisA) : :]
        for p in range(2)
    )
    x_dual = (
        tuple(1 - d for d in a_dual_tmp[len(free_axisA) : :])
        + b.dual[len(free_axisA) : :]
    )
    x = x.split_axes(
        (len(axisX), b.ndim - len(free_axisA)),
        x_shape, dualSplit=x_dual, auto_dual=False,
    )
    # recover axis order for x
    permX = axisX + [i for i in range(x.ndim) if i not in axisX]
    permXinv = utils.get_invperm(permX)
    x = x.transpose(*permXinv)
    return x


def lstsq(a: GTensor, b: GTensor, axes, return_err=False):
    x = solve(a, b, axes, _lstsq=True)
    if return_err:
        lserr = core.norm(core.tensordot(a, x, axes) - b)
        return x, lserr
    else:
        return x, -1.0


def gate_to_mpo(gate: GTensor, verify=True) -> list[GTensor]:
    """
    Convert multi-site gate to MPO by SVD.
    Singular values too small are discarded.

    Input gate axis order
    ----
    ```
        i  i+1      i  i+1 i+2
        0   1       0   1   2
        ↓---↓       ↓---↓---↓
        ↓---↓       ↓---↓---↓
        2   3       3   4   5
    ```

    Gate MPO axis order
    ----
    the physical indices are always at (0, 1);
    the virtual indices are order from left to right

    Output axis order
    ----
    ```
        0               0               0
        ↓               ↓               ↓
        M[0] → 2 -→ 2 → M[1] → 3 -→ 2 → M[2]
        ↓               ↓               ↓
        1               1               1
    ```
    - Axis 0, 1 are physical axes
    - 0th site:  Axis 2 connects to the next site
    - Last site: Axis 2 connects to the last site
    - Others:    Axis 2/3 connects to the last/next site
    """
    # 2-site gate
    if gate.ndim == 4:
        s1, _, s2 = svd(
            gate.transpose(0,2,1,3), 2, cutoff=True, absorb_s=True
        )
        Ms = [s1, s2.transpose(1,2,0)]
        if verify:
            assert core.allclose(
                gate, core.einsum('abk,cdk->acbd', Ms[0], Ms[1])
            )
    # 3-site gate
    elif gate.ndim == 6:
        s1, _, s23 = svd(
            gate.transpose(0,3,1,4,2,5), 2, 
            cutoff=True, eps=1e-15, absorb_s=True
        )
        s2, _, s3 = svd(s23, 3, cutoff=True, eps=1e-15, absorb_s=True)
        Ms = [s1, s2.transpose(1,2,0,3), s3.transpose(1,2,0)]
        if verify:
            assert core.allclose(gate, core.einsum(
                "abg,cdgh,efh->acebdf", Ms[0], Ms[1], Ms[2]
            ))
    else:
        raise NotImplementedError
    return Ms
