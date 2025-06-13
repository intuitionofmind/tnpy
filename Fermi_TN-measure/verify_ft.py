import numpy as np
import gtensor as gt
import fermiT as ft
from fermiT.linalg import *
from fermiT.conversion import ft2gt, gt2ft


def test_flip_dual():
    a = ft.rand((6,8,4,10,6), (3,4,2,5,3), (0,1,0,0,1), 0)
    b = ft.rand((4,6,10,8,4), (2,3,5,4,2), (0,1,1,0,1), 1)
    ag, bg = ft2gt(a), ft2gt(b)
    axes = ((1,3,2), (3,2,4))
    c = ft.tensordot(a, b, axes)
    assert gt.allclose(ft2gt(c), gt.tensordot(ag, bg, axes))
    for flip in ("a", "b"):
        a2, b2 = ft.flip2_dual(a, b, axes, flip=flip)
        ag2, bg2 = gt.flip2_dual(ag, bg, axes, flip=flip)
        compare = list(map(
            gt.allclose, (ag2, bg2), map(ft2gt, (a2, b2))
        ))
        assert all(compare)
        c2 = ft.tensordot(a2, b2, axes)
        assert ft.allclose(c, c2)


def test_conversion(seed=None):
    """Test GTensor - FermiT conversion"""
    from random import shuffle
    if seed is not None:
        np.random.rand(seed)
    # ---------- 2-way conversion ----------
    ndim = 5
    DS = np.random.randint(3,9,ndim)
    DE = np.array([np.random.randint(1, d) for d in DS])
    Dual = np.random.randint(0,2,ndim)
    fts = rand(DS, DE, Dual)
    gts = ft2gt(fts)
    fts2 = gt2ft(gts)
    gts2 = ft2gt(fts2)
    assert array_equal(fts, fts2)
    assert gt.array_equal(gts, gts2)
    # ---------- transpose ----------
    ndim = 5
    DS = np.random.randint(3,9,ndim)
    DE = np.array([np.random.randint(1, d) for d in DS])
    fts = rand(
        DS, DE, np.random.randint(0,2,ndim)
    )
    nonzero_idxs = np.array(np.nonzero(fts.val)).T
    order = list(range(ndim))
    shuffle(order)
    assert gt.allclose(
        ft2gt(fts.transpose(*order)),
        ft2gt(fts).transpose(*order)
    )
    # ---------- tensordot ----------
    a_ndim = 6
    a_DS = np.random.randint(3,8,a_ndim)
    a_DE = np.array([np.random.randint(1, d) for d in a_DS])
    a_Dual = np.random.randint(0,2,a_ndim)
    b_ndim = 5
    b_DS = np.random.randint(3,8,b_ndim)
    b_DE = np.array([np.random.randint(1, d) for d in b_DS])
    b_Dual = np.random.randint(0,2,b_ndim)
    axes = [(4,2,5), (1,3,2)]
    gMetrics = (-1,1,-1)
    naxes = len(axes[0])
    dims = np.random.randint(3,8,naxes)
    dimsE = np.array([np.random.randint(1, d) for d in dims])
    # # set dim for axes to be contracted
    for a_ax, b_ax, gMetric, dim, dimE in zip(axes[0], axes[1], gMetrics, dims, dimsE):
        a_DS[a_ax] = b_DS[b_ax] = dim
        a_DE[a_ax] = b_DE[b_ax] = dimE
        if gMetric == 1:
            # a -> b: <a| and |b>
            a_Dual[a_ax] = 1; b_Dual[b_ax] = 0
        else:
            # a <- b: |a> and <b|
            a_Dual[a_ax] = 0; b_Dual[b_ax] = 1
    # create tensors
    aft = rand(a_DS, a_DE, a_Dual)
    bft = rand(b_DS, b_DE, b_Dual)
    # tensordot in two ways
    assert gt.allclose(
        ft2gt(tensordot(aft, bft, axes)), 
        gt.tensordot(ft2gt(aft), ft2gt(bft), axes)
    )


if __name__ == "__main__":
    test_flip_dual()
    for _ in range(3):
        seed = None
        test_conversion(seed)
