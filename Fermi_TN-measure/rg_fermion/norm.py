"""
Calculate norms of a local part 
of the tensor network
"""

import gtensor as gt
# import gtensor.legacy as gt
from gtensor import GTensor
import numpy as np

def norm4(
    a: GTensor, b: GTensor, 
    c: None | GTensor=None, d: None | GTensor=None
):
    """
    Calculate the norm of the following network
    (using loop convention)
    ```
            1         1
            ↑         ↓
        2 →B/D← 0 2 ←A/C→ 0
            ↓         ↑
            3         3
            1         1
            ↓         ↑
        2 ← A → 0 2 → B ← 0
            ↑         ↓
            3         3
    ```
    By default, C = A, D = B

    The contraction order is bottom -> top, left -> right
    (A -> B -> D -> C) or (A -> D -> B -> C)
    """
    if (c is None) and (d is None):
        c, d = a, b
    else:
        assert isinstance(c, GTensor)
        assert isinstance(d, GTensor)
    assert all(
        a.shape[par][0] == b.shape[par][2] and
        d.shape[par][0] == c.shape[par][2] and
        a.shape[par][1] == d.shape[par][3] and
        b.shape[par][1] == c.shape[par][3] for par in range(2)
    ), 'a, b, c, d have incompatible shapes'
    # cost estimation
    ## row cost
    cost1 = np.prod(a.shape[0]) * b.shape[0][1] * b.shape[0][3] + \
            np.prod(b.shape[0]) * a.shape[0][1] * a.shape[0][3] + \
            a.shape[0][1] * a.shape[0][3] * b.shape[0][1] * b.shape[0][3]
    ## column cost
    cost2 = np.prod(a.shape[0]) * b.shape[0][0] * b.shape[0][2] + \
            np.prod(b.shape[0]) * a.shape[0][0] * a.shape[0][2] + \
            a.shape[0][0] * a.shape[0][2] * b.shape[0][0] * b.shape[0][2]
    # based on cost estimation, choose a scheme to compute norm
    if cost1 <= cost2:
        # row calculation
        ab = gt.tensordot(a, b, [(0,2), (2,0)])
        dc = gt.tensordot(d, c, [(0,2), (2,0)])
        return gt.tensordot(ab, dc, [(0,2,1,3), (1,3,0,2)]).item()
    else:
        # column calculation
        ad = gt.tensordot(a, d, [(3,1), (1,3)])
        bc = gt.tensordot(b, c, [(3,1), (1,3)])
        return gt.tensordot(ad, bc, [(1,3,0,2), (0,2,1,3)]).item()

def norm2(a: GTensor, b: GTensor):
    """
    Calculate the norm of the following network
    (using loop convention)
    
            1         1
            ↓         ↑
        2 ← A → 0 2 → B ← 0
            ↑         ↓
            3         3
        a[r,u,l,d] * b[l,d,r,u]
    """
    return gt.tensordot(a, b, [(0,1,2,3),(2,3,0,1)]).item()
