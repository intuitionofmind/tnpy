import numpy as np
import scipy.linalg as linalg
import torch
import torch.linalg as tla
from torch import Tensor
torch.set_default_dtype(torch.float64)
from .utils import get_invperm

def svd_cutoff(
    u: np.ndarray, s: np.ndarray, vh: np.ndarray, 
    Dmax=None, eps=1.e-15
):
    """truncate ordinary matrix SVD result according to the singular values"""
    if Dmax == None:
        Dmax = len(s[s > 0])
    s_cut  = s[s > eps * s[0]][:Dmax]
    d_cut  = len(s_cut)
    u_cut  = u[:, 0:d_cut]
    vh_cut = vh[0:d_cut, :]
    return u_cut, s_cut, vh_cut, d_cut


def svd_error(s: np.ndarray, d_cut: int) -> float:
    """
    Calculate the truncation error of SVD
    using the spectrum of singular values

    Parameters
    ----
    s: ndarray
        singular value spectrum
    d_cut: int
        the number of singular values to be kept
    """
    assert s.ndim == 1
    assert 0 <= d_cut <= s.size
    relE = np.sqrt(
        linalg.norm(s[d_cut:])**2 / linalg.norm(s)**2
    ) if d_cut < s.size else 0.0
    return relE


def polar(a: Tensor, side="right"):
    """
    Polar decomposition for ordinary tensors
    implemented with PyTorch

    Parameters
    ----
    side: str ('left' or 'right')
    - right: A = U P
    - left : A = P U

    Returns
    ----
    u, p: Tensors
        Polar decomposition results
    """
    if side not in ['right', 'left']:
        raise ValueError("`side` must be either 'right' or 'left'")
    if a.ndim != 2:
        raise ValueError("`a` must be a 2-D array.")

    w, s, vh = tla.svd(a, full_matrices=False)
    assert isinstance(w, Tensor) and \
        isinstance(vh, Tensor) and isinstance(s, Tensor)
    u = w @ vh
    if side == 'right':
        # a = up
        p = vh.conj().T @ s.diag().to(dtype=a.dtype) @ vh
    else:
        # a = pu
        p = w @ s.diag().to(dtype=a.dtype) @ w.conj().T
    return u, p


def dot_diag(a: Tensor, diag: Tensor, axis=1):
    """
    Efficiently multiply a tensor with a diagonal matrix

    Parameters
    ----
    diag: np.ndarray (1-dimensional)
        diagonal elements of the matrix
    axis: int
        axis of `a` to be multiplied with `diag`
    """
    assert 0 <= axis <= a.ndim
    assert diag.ndim == 1
    # optimization for matrices
    if a.ndim == 2:
        if axis == 1:
            a = a * diag
        elif axis == 0:
            a = (a.T * diag).T
    # general case
    else:
        # put *axes to the rightmost
        perm = [i for i in range(a.ndim) if i != axis] + [axis,]
        # multiply
        a = a.permute(*perm) * diag
        # inverse transpose
        perm_inv = get_invperm(perm)
        a = a.permute(*perm_inv)
    return a


def qr(a: Tensor, mode="reduced", posR=True) -> tuple[Tensor, Tensor]:
    """
    QR decompoition of tensor `a`. 
    The diagonal elements of `r` are made positive
    """
    assert mode in ("reduced", "r")
    q, r = tla.qr(a, mode=mode)
    # adjust diagonal elements of `r`
    if posR:
        d = torch.exp(1j * torch.angle(torch.diagonal(r)))
        if mode == "reduced":
            q = dot_diag(q, d, 1)
        r = dot_diag(r, d.conj(), 0)
    return q, r
