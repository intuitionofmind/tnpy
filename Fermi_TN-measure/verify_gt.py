"""
GTensor test script
"""

import torch
torch.set_default_dtype(torch.float64)
import numpy as np
from time import time
from itertools import product
import gtensor as gt
import gtensor.linalg as gla
import gtensor.legacy as gt0
# import atexit
# import line_profiler as lp
# profile = lp.LineProfiler()
# atexit.register(profile.print_stats)

def timing_func(func):
    starttime = time()
    func()
    endtime = time()
    print("{:26s}  {:>7.2f} ms".format(
        func.__name__ + ':', 
        (endtime - starttime) * 1000.0
    ))

# ------ Self-testing ------

def test_flatten():
    shape = ((3,4,5),(4,2,3))
    dual = (1,0,1)
    for par in range(2):
        a0 = gt.rand(shape, dual, par).transpose(2,0,1)
        v = gt.flatten(a0)
        a1 = gt.unflatten(a0.shape, a0.dual, par, v)
        assert gt.array_equal(a0, a1)


def test_merge_split_axes():
    for p1, p2, d, g1, g2, g in product(
        (0,1), (0,1), (0,1), (1,-1), (1,-1), (1,-1)
    ):
        d1, d2 = d, 1-d
        a = gt.rand((2,3,5,4), dual=(d,d1,1,0), parity=p1).transpose(2,3,0,1)
        b = gt.rand((2,3,5,4), dual=(d2,d2,1,0), parity=p2)
        c1 = gt.tensordot(a, b, ((2,3),(0,1)))

        a_ = gt.merge_axes(a, (2,2), (0,d1), order=(g1,g), auto_dual=False)
        b_ = gt.merge_axes(b, (2,2), (d2,0), order=(-g,g2), auto_dual=False)
        c2 = gt.tensordot(a_, b_, (1,0))
        c2 = gt.split_axes(c2, (2,2), c1.shape, c1.dual, order=(g1,g2), auto_dual=False)
        assert gt.allclose(c1, c2)


def test_flip_dual():
    a = gt.rand((3,4,2,5,3), (0,1,0,0,1), 0)
    b = gt.rand((2,3,5,4,2), (0,1,1,0,1), 1)
    axes = ((1,3,2), (3,2,4))
    c = gt.tensordot(a, b, axes)
    for flip in ("a", "b"):
        a2, b2 = gt.flip2_dual(a, b, axes, flip=flip)
        assert gt.allclose(c, gt.tensordot(a2, b2, axes))


def test_tensordot_einsum():
    # ---------- 2-tensor test ----------
    a = gt.rand((4,6,2,3,5,3), (1,0,0,1,1,0), 0)
    b = gt.rand((3,4,2,3,6,5), (0,0,1,0,1,0), 1)
    c1 = gt.tensordot(a, b, ((2,4,1), (2,5,4)))
    assert gt.allclose(c1, gt.tensordot(a, b, ((1,2,4), (4,2,5))))
    # should be the same as gt.einsum result
    c1_ = gt.einsum("abcdef,ghcibe", a, b)
    assert gt.allclose(c1, c1_)
    # gt.tensordot over multiple axes with the same dual
    # can also be performed by first merging these axes 
    # and contract the combined "big index"
    # transpose to bring contracted axes together
    at2 = a.transpose(0,3,5,2,4,1)
    bt2 = b.transpose(2,5,4,0,1,3)
    # flip dual so that axes 
    # to be contracted all have the same dual
    for flip in ("a", "b"):
        at3, bt3 = gt.flip2_dual(at2, bt2, (-2, 1), flip=flip)
        # merge: (3,3,4,[2,5,6])
        at3 = at3.merge_axes((1,1,1,3), order=(1,1,1,-1), auto_dual=True)
        # merge: ([2,5,6],3,4,3)
        bt3 = bt3.merge_axes((3,1,1,1), order=(1,1,1,1), auto_dual=True)
        c2 = gt.tensordot(at3, bt3, [-1,0])
        assert gt.allclose(c1, c2)

    # ---------- 3-tensor test ----------
    r"""
              a
              |
              T1
             /  \
            d    f
           /      \
          T2 --e-- T3
         /          \
        b            c
    """
    t1 = gt.rand(((3,4,5),(2,3,4)), (0,1,0))
    t2 = gt.rand(((3,6,4),(2,5,3)), (1,1,0))
    t3 = gt.rand(((3,5,6),(2,4,5)), (0,1,0))
    res1 = gt.tensordot(
        gt.tensordot(t1, t2, (1,2)), t3, ((1,3),(1,2))
    )
    res2 = gt.einsum("adf, bed, cfe -> abc", t1, t2, t3)
    res3 = gt.fncon([t1, t2, t3], [[-1,1,3], [-2,2,1], [-3,3,2]])
    assert gt.allclose2(res1, res2, res3)


def test_tensordot_keepform():
    a = gt.rand((3,4,1,2,3,5), (1,0,0,1,0,1), 1)
    b = gt.rand((4,2,3,2), (1,0,0,1), 1)
    c = gt.tensordot_keepform(a, b, [0, 2])
    assert c.shape == ((4,2,2,4,1,2,3,5),) * 2 and c.dual == (1,0,1,0,0,1,0,1)
    c_ = gt.einsum("xabcde,fgxh->fghabcde", a, b)
    assert gt.allclose(c, c_)


def test_dotdiag():
    ax1 = 2
    for p, d, ax2 in product((0,1), repeat=3):
        u = gt.rand(((2,5,4,3),(3,4,3,5)), dual=(1,0,d,1), parity=p)
        s = gt.diag(dict(
            (par, torch.rand(u.shape[par][ax1], dtype=torch.cdouble)) 
            for par in range(2)
        ), dual=(
            (1-d,0) if ax2 == 0 else (0,1-d)
        ))
        us1 = gt.dot_diag(u, s, [ax1, ax2])
        # result should be the same as `gt.tensordot_keepform`
        us2 = gt.tensordot_keepform(u, s, [ax1, ax2], anchor="a")
        us3 = gt.tensordot_keepform(s, u, [ax2, ax1], anchor="b")
        assert gt.allclose2(us1, us2, us3)


def test_trace():
    # from cmath import isclose as cisclose
    # shape: (3,2,6,5,5,3)
    a = gt.rand((3,5,2,5,3,6), dual=(0,)*6).transpose(4,2,5,3,1,0)
    a._dual = (1,1,0,0,1,0)
    b = gt.trace(a, axis1=(0,4), axis2=(5,3))
    # gt.trace is the same as contracting with identity
    c = gt.tensordot(gt.eye(3), a, ((0,1), (0,5)))
    c = gt.tensordot(gt.eye(5), c, ((0,1), (3,2)))
    assert gt.allclose(b, c)
    # full gt.trace test
    a = gt.rand(((3,3,4,4), (5,5,2,2)), dual=(0,1,1,0), parity=0)
    tr = gt.trace(a, axis1=(1,3), axis2=(0,2))
    assert tr.shape == ((),())
    value = tr.item()


def test_inv():
    for d1, d2, parity in product(range(2), repeat=3):
        dual = (d1, d2)
        if parity == 0:
            shape = ((3,3), (4,4))
        else:
            shape = ((3,4), (4,3))
        a = gt.rand(shape, dual, parity=parity)
        ainv = gla.matrix_inv(a, is_diag=False)
        apinv = gla.matrix_pinv(a, is_diag=False)
        for side in ("left", "right"):
            iden = (
                gt.tensordot(ainv, a, [1,0]) if side == "left"
                else gt.tensordot(a, ainv, [1,0])
            )
            assert gt.is_identity_matrix(iden)
            piden = (
                gt.tensordot(apinv, a, [1,0]) if side == "left"
                else gt.tensordot(a, apinv, [1,0])
            )
            assert gt.is_identity_matrix(piden)
        # for diagonal matrix
        diag_a = gt.diag({0: torch.rand(4), 1: torch.rand(3)}, dual)
        ainv = gla.matrix_inv(diag_a, is_diag=True)
        assert gt.allclose(ainv, gla.matrix_inv(diag_a, is_diag=False))


def test_matrix_sqrt():
    for dual in product(range(2), repeat=2):
        a_diag = dict(
            (par, torch.rand(dim, dtype=torch.cdouble)) 
            for par, dim in zip((0,1), (3,4))
        )
        a = gt.diag(a_diag, dual)
        g1, g2 = gla.matrix_sqrt(a, is_diag=True)
        assert gt.allclose(a, gt.tensordot(g1, g2, [1,0]))


def test_matrix_polar():
    for odd, parity, d1, d2 in product(
        ("p","u"), (0,1), (0,1), (0,1)
    ):
        # test 1: the number of rows is larger
        # should use UP (right) decomposition
        a = gt.rand(((7,4),(5,3)), (d1,d2), parity=parity)
        u, p = gla.polar(a, 1, "right", odd=odd)
        assert (gt.allclose(a, gt.tensordot(u, p, [-1,0])))
        assert gla.is_unitary(u, [0], "l") 

        # test 2: the number of columns is larger
        # should use PU (left) decomposition
        a = gt.rand(((4,7),(3,5)), (d1,d2), parity=parity)
        p, u = gla.polar(a, 1, "left", odd=odd)
        assert (gt.allclose(a, gt.tensordot(p, u, [-1,0])))
        assert gla.is_unitary(u, [0], "r") 


def test_polar():
    for g1, g2 in product(range(2), range(2)):
        dual = (g1,g1,g2,g2)
        
        # right decomposition
        shape = ((4,5,4,3), (3,4,2,2))
        a = gt.rand(shape, dual, parity=0)
        u, p = gla.polar(a, 2, side="right")
        a1 = gt.tensordot(u, p, [2,0])
        assert gt.allclose(a, a1)
        assert gla.is_unitary(u, [0,1], "l")

        # left decomposition
        shape = ((4,3,4,5), (2,2,3,4))
        a = gt.rand(shape, dual, parity=0)
        p, u = gla.polar(a, 2, side="left")
        a1 = gt.tensordot(p, u, [2,0])
        assert gt.allclose(a, a1)
        assert gla.is_unitary(u, [0], "r")


def test_qr():
    (g1, g2, g3) = (1,0,1)
    for p in range(2):
        a = gt.rand(
            ((3,2,4,6,2), (4,3,5,2,6)), 
            (g1,0,g2,1,g3), parity=p
        )
        for odd in ("q","r"):
            q, r = gla.qr(a, [3,1], odd=odd)
            b = gt.tensordot(q, r, [-1,0]).transpose(0,4,1,3,2)
            assert gt.allclose(a, b)
            assert gla.is_unitary(q, [0,1,2], "l")


def test_lq():
    for g, p, odd in product((0,1), (0,1), ("q","l")):
        a = gt.rand(
            ((3,2,4,6,2), (4,3,5,2,6)), 
            (g,0,g,1,g), parity=p
        )
        # test `lq`
        l, q = gla.lq(a, (3,1), odd=odd)
        b = gt.tensordot(l, q, [-1,0]).transpose(2,1,3,0,4)
        assert gt.allclose(a, b)
        assert gla.is_unitary(q, [0], "r")


def test_svd():
    ndim, nAxis = 5, 2
    shape = (
        tuple(np.random.randint(2,5,ndim)),
        tuple(np.random.randint(2,5,ndim)),
    )
    Dmax = 10
    for p, d1, d2, odd in product(
        (0,1), (0,1), (0,1), ("u", "vh")
    ):
        a = gt.rand(shape, dual=(d1,)*nAxis + (d2,)*(ndim-nAxis), parity=p)
        # verify u -> s -> vh
        u, s, vh = gla.svd(a, nAxis, odd=odd)
        us = gt.dot_diag(u, s, [nAxis, 0])
        a1 = gt.tensordot(us, vh, [nAxis, 0])
        assert gt.allclose(a, a1)
        # verify unirarity of u and vh
        assert gla.is_unitary(u, list(range(nAxis)), "l")
        assert gla.is_unitary(vh, [0], "r")

        # verify SVD cutoff
        u_, s_, vh_ = gla.svd(
            a, nAxis, cutoff=True, Dmax=Dmax, odd=odd
        )
        # SVD cutoff is equivalent to setting small singular values to 0
        sdiag = s.diagonal()
        sdiag_cut = sdiag.copy()
        De, Do = s_.DE[0], s_.DO[0]
        sdiag_cut[0][De::] = 0.0
        sdiag_cut[1][Do::] = 0.0
        s_cut = gt.diag(sdiag_cut, s.dual)
        a_ = gt.tensordot(
            u_.dot_diag(s_, [nAxis, 0]),  vh_, (nAxis, 0)
        )
        a_cut = gt.tensordot(
            u.dot_diag(s_cut, [nAxis, 0]), vh, (nAxis, 0)
        )
        assert gt.allclose(a_, a_cut)


def test_norm():
    # verify equivalence of 2-gt.norm and full contraction
    # of `a.gT` and `a.pconj()`
    from cmath import isclose as cisclose
    from math import isclose
    for p in range(2):
        a = gt.rand(shape=(3,4,4,5), dual=(0,1,1,0), parity=p)
        nrm1 = gt.tensordot(
            a.gT, a.pconj(), [[0,1,2,3]] * 2
        ).item()
        assert cisclose(nrm1, nrm1.real)
        nrm2 = gt.norm(a) 
        assert isclose(nrm1.real, nrm2**2)


def test_matrix_solve():
    for pa, pb in product(range(2), repeat=2):
        a = gt.rand(((4,4),(4,4)), (0,1), parity=pa)
        b = gt.rand(((4,4),(4,4)), (0,1), parity=pb)
        x = gla.solve(a, b, [1,0])
        b_ = gt.tensordot(a, x, [1,0])
        assert gt.allclose(b, b_)


def test_solve():
    from math import isclose
    axes = ((2,0,5),(1,4,2))
    a_ndim, b_ndim = 6, 5
    np.random.seed(100)
    a_dual = np.random.randint(0, 2, a_ndim)
    b_dual = np.random.randint(0, 2, b_ndim)
    free_axisA = [ax for ax in range(a_ndim) if ax not in axes[0]]
    # dual and shape of free axes of `a` should match those of `b`
    for ax_b, ax_a in enumerate(free_axisA):
        b_dual[ax_b] = a_dual[ax_a]
    for apar, bpar in product(range(2), repeat=2):
        a = gt.rand((3,4,5,4,5,3), (0,)*a_ndim, parity=apar).transpose(2,0,5,1,4,3)
        b = gt.rand((4,6,5,2,3), (0,)*b_ndim, parity=bpar).transpose(4,0,2,3,1)
        a._dual, b._dual = tuple(a_dual), tuple(b_dual)
        a0, b0 = a.copy(), b.copy()
        x = gla.solve(a, b, axes)
        x_, lserr1 = gla.lstsq(a, b, axes, True)
        lserr2 = gt.norm(gt.tensordot(a, x_, axes) - b)
        c = gt.tensordot(a, x, axes)
        assert gt.array_equal(a, a0) and gt.array_equal(b, b0)
        assert gt.allclose(b, c) 
        assert isclose(lserr1, lserr2)
        # sverr = gt.norm(gt.tensordot(a, x, axes) - b)
        # print(lserr1, lserr2, sverr)


if __name__ == "__main__":
    print('\n---------- GTensor self-check ----------')
    total_starttime = time()
    timing_func(test_flatten)
    timing_func(test_merge_split_axes)
    timing_func(test_flip_dual)
    timing_func(test_tensordot_einsum)
    timing_func(test_tensordot_keepform)
    timing_func(test_dotdiag)
    timing_func(test_trace)
    timing_func(test_inv)
    timing_func(test_matrix_sqrt)
    timing_func(test_matrix_polar)
    timing_func(test_polar)
    timing_func(test_qr)
    timing_func(test_lq)
    timing_func(test_svd)
    timing_func(test_norm)
    timing_func(test_matrix_solve)
    timing_func(test_solve)
    total_endtime = time()
    print('{:26s}  {:>7.2f} ms'.format(
        "Total self-check time:", 
        (total_endtime - total_starttime) * 1000.0
    ))
    print('--------- Self-check complete ---------\n')
