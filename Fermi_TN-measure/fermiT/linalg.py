from .core import *
from .reshape import merge_axes, split_axes
import scipy.linalg as scila

def norm(a: FermiT):
    return np.linalg.norm(a.val)

def matrix_sqrt(ftau: FermiT, typef=0, is_diag=True):
    assert ftau.ndim == 2
    assert typef in (0, 1)
    if is_diag:
        if typef == 0:
            val = np.diag(np.sqrt(ftau.val.diagonal()))
            f1 = FermiT(ftau.DS, ftau.DE, (ftau.dual[0],1), val)
            f2 = FermiT(ftau.DS, ftau.DE, (0,ftau.dual[1]), val)
        else:
            De = ftau.DE[0]
            val1 = np.sqrt(ftau.val.diagonal())
            val2 = np.sqrt(ftau.val.diagonal())
            val2[De:] = -1 * val2[De:]
            val1 = np.diag(val1)
            val2 = np.diag(val2)
            f1 = FermiT(ftau.DS, ftau.DE, (ftau.dual[0],0), val1)
            f2 = FermiT(ftau.DS, ftau.DE, (1,ftau.dual[1]), val2)
    else:
        raise NotImplementedError
    return f1, f2

def svd(
    a: FermiT, nAxis: int, eps=None, Dcut=None, approx='S', 
):
    """
    SVD of fermion tensors

    Parameters
    ----
    eps: float or None
        the smallest singular value 
        (ratio to the largest) to be kept
    Dcut: int or None
        total number of singular values to be kept
    approx: str ('S' or 'T')
        singular value cutoff scheme
        - "S": keep the same dimension on even and odd sectors
        - "T": according to the magnitude on both sectors
    """
    shapeT = [list(range(nAxis)), list(range(nAxis, a.ndim))]
    rank0 = len(shapeT[0])
    DS0 = a.DS
    DE0 = a.DE
    aGr = merge_axes(a, shapeT)
    DS = aGr.DS
    DE = aGr.DE
    dual = aGr.dual
    Ae = aGr.blocks((0,0))
    Ao = aGr.blocks((1,1))
    Ue, Se, Ve = np.linalg.svd(Ae, full_matrices=False)
    Uo, So, Vo = np.linalg.svd(Ao, full_matrices=False)
    if eps is None:
        Dce = Se.shape[0]
        Dco = So.shape[0]
    else:
        Dce = (Se/max(Se[0],So[0]) > eps).sum()
        Dco = (So/max(Se[0],So[0]) > eps).sum()
        Dce = max(Dce, 2)
        Dco = max(Dco, 2)
        Ue, Se, Ve = Ue[:,:Dce], Se[:Dce], Ve[:Dce,:]
        Uo, So, Vo = Uo[:,:Dco], So[:Dco], Vo[:Dco,:]

    def svd_cutoff(la, lb):
        lab = np.sort(np.append(la[:Dcut], lb[:Dcut]))[::-1]
        val = lab[Dcut]
        De = (la[:Dcut] > val).sum()
        return min(max(De, 1), Dcut-1)
        
    if Dcut < Dce + Dco:
        Ds = Dcut
        if approx == "S":
            De = min(Dce, Ds // 2)
        elif approx == "T":
            De = svd_cutoff(Se, So)
        Do = Ds - De
        Ue = Ue[:,:De]
        Se = Se[:De]
        Ve = Ve[:De,:]
        Uo = Uo[:,:Do]
        So = So[:Do]
        Vo = Vo[:Do,:]
    else:
        Ds = Dce + Dco
        De = Dce
    typef = (0,1)
    dualU = np.array([dual[0], 1-typef[0]])
    dualV = np.array([1-typef[1], dual[1]])
    dualS = np.array(typef, dtype=int)
    valU = np.zeros((DS[0], Ds), dtype=complex)
    valV = np.zeros((Ds, DS[1]), dtype=complex)
    valS = np.zeros((Ds, Ds), dtype=complex)

    valU[:DE[0],:De] = Ue
    valU[DE[0]:,De:] = Uo
    U = FermiT([DS[0],Ds], [DE[0],De], dualU, valU)
    shapeU = [list(np.argsort(shapeT[0])), [rank0]]
    DSU = np.append(DS0[shapeT[0]], Ds)
    DEU = np.append(DE0[shapeT[0]], De)
    U = split_axes(U, shapeU, DSU, DEU)
    U.dual = np.append(a.dual[shapeT[0]], U.dual[-1:])

    valV[:De,:DE[1]] = Ve
    valV[De:,DE[1]:] = Vo
    Vh = FermiT([Ds,DS[1]], [De,DE[1]], dualV, valV)
    shapeV = [[0], list(np.argsort(shapeT[1]) + 1)]
    DSV = np.append(Ds, DS0[shapeT[1]])
    DEV = np.append(De, DE0[shapeT[1]])
    Vh = split_axes(Vh, shapeV, DSV, DEV)
    Vh.dual = np.append(Vh.dual[:1], a.dual[shapeT[1]])

    valS[:De,:De] = np.diag(Se)
    valS[De:,De:] = np.diag(So)
    S = FermiT((Ds,Ds), (De,De), dualS, valS)
    
    return U, S, Vh, Ds

def polar(
    a: FermiT, nAxis: int, side="right", typef=0
):
    """
    Polar decomposition of fermion tensors

    Parameters
    ----
    side: str ("right" or "left")
        - "right", T = U P
        - "left",  T = P U.
        The values "P", "U" are any representation.
    """
    shapeT = [list(range(nAxis)), list(range(nAxis, a.ndim))]
    rank0 = len(shapeT[0])
    DS0 = a.DS
    DE0 = a.DE

    aGr = merge_axes(a, shapeT)
    DS = aGr.DS
    DE = aGr.DE
    dual = aGr.dual
    
    Ae = aGr.blocks((0,0))
    Ao = aGr.blocks((1,1))
    
    Ue, Pe = scila.polar(Ae, side=side)
    if Ao.shape == (0,0):
        Uo, Po = np.empty_like(Ao), np.empty_like(Ao)
    else:
        Uo, Po = scila.polar(Ao, side=side)

    if side == "left":
        Ue, Pe = Pe, Ue
        Uo, Po = Po, Uo

    Ds = Pe.shape[0] + Po.shape[0]
    De = Pe.shape[0]

    if typef == 1:
        if side == "right": Uo = -Uo
        else: Po = -Po
        dualU = np.array([dual[0], 0])
        dualP = np.array([1, dual[1]])
    else:
        dualU = np.array([dual[0], 1])
        dualP = np.array([0, dual[1]])
        
    valU = np.zeros((DS[0],Ds), dtype=complex)
    valP = np.zeros((Ds,DS[1]), dtype=complex)

    valU[:DE[0],:De] = Ue
    valU[DE[0]:,De:] = Uo
    U = FermiT([DS[0],Ds], [DE[0],De], dualU, valU)
    shapeU = [list(np.argsort(shapeT[0])), [rank0]]
    DSU = np.append(DS0[shapeT[0]], Ds)
    DEU = np.append(DE0[shapeT[0]], De)
    U = split_axes(U, shapeU, DSU, DEU)
    U.dual = np.append(a.dual[shapeT[0]], U.dual[-1:])

    valP[:De,:DE[1]] = Pe
    valP[De:,DE[1]:] = Po
    P = FermiT([Ds,DS[1]], [De,DE[1]], dualP, valP)
    shapeP = [[0], list(np.argsort(shapeT[1])+1)]
    DSP = np.append(Ds, DS0[shapeT[1]])
    DEP = np.append(De, DE0[shapeT[1]])
    P = split_axes(P, shapeP, DSP, DEP)
    P.dual = np.append(P.dual[:1], a.dual[shapeT[1]])

    return U, P

def qr(
    a: FermiT, axesR: list[int], mode="reduced"
):
    axesQ = [i for i in range(a.ndim) if i not in axesR]
    shapeT = [axesQ, axesR]
    rank0 = len(shapeT[0])
    DS0 = a.DS
    DE0 = a.DE

    aGr = merge_axes(a, shapeT)
    DS = aGr.DS
    DE = aGr.DE
    dual = aGr.dual
    
    Ae = aGr.blocks((0,0))
    Ao = aGr.blocks((1,1))

    Qe, Re = np.linalg.qr(Ae, mode=mode)
    Qo, Ro = np.linalg.qr(Ao, mode=mode)

    Ds = Re.shape[0] + Ro.shape[0]
    De = Re.shape[0]
    dualQ = np.array([dual[0], 1])
    dualR = np.array([0, dual[1]])
    valQ = np.zeros((DS[0],Ds), dtype=complex)
    valR = np.zeros((Ds,DS[1]), dtype=complex)

    valQ[:DE[0],:De] = Qe
    valQ[DE[0]:,De:] = Qo
    Q = FermiT([DS[0],Ds], [DE[0],De], dualQ, valQ)
    shapeQ = [list(np.argsort(shapeT[0])), [rank0]]
    DSQ = np.append(DS0[shapeT[0]], Ds)
    DEQ = np.append(DE0[shapeT[0]], De)
    Q = split_axes(Q, shapeQ, DSQ, DEQ)
    Q.dual = np.append(a.dual[shapeT[0]], Q.dual[-1:])

    valR[:De,:DE[1]] = Re
    valR[De:,DE[1]:] = Ro
    R = FermiT([Ds,DS[1]], [De,DE[1]], dualR, valR)
    shapeR = [[0], list(np.argsort(shapeT[1]) + 1)]
    DSR = np.append(Ds, DS0[shapeT[1]])
    DER = np.append(De, DE0[shapeT[1]])
    R = split_axes(R, shapeR, DSR, DER)
    R.dual = np.append(R.dual[:1], a.dual[shapeT[1]])

    return Q, R
