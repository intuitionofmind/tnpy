"""
Merge two layers of TPS (tensor product state)
"""

import gtensor as gt
from gtensor import GTensor

def double_ts1(
    T: GTensor, op: None | GTensor=None, 
    sublat='A'
):
    """
    Given TPS tensors Ta, Tb on square lattice 
    and the 1-site operators to be measured, 
    multiply two layers to form double tensor

    TPS tensor axis convention
    - Physical index : 0 (dual = 0, |ket>)
    - Virtual index: (top view; loop convention)

    ```
            2           2
            ↓           ↑
        3 ← A → 1   3 → B ← 1
            ↑           ↓
            4           4
    ```

    Operator axis convention (side view)
    ```
        0
        ↓
        op
        ↓
        1
    ```
    """
    assert sublat in ('A', 'B')
    tmp = (T if op is None else gt.tensordot(op, T, (1,0)))
    if sublat == 'A':
        # the minus signs are absorbed to sub-lattice A
        doubleT = gt.tensordot(
            T.gT.flip_dual((1,2,3,4), minus=True), tmp, (0,0)
        ).transpose(0,4,1,5,2,6,3,7).merge_axes((2,2,2,2), order=(1,1,-1,-1))
    else:
        doubleT = gt.tensordot(
            T.gT.flip_dual((1,2,3,4), minus=False), tmp, (0,0)
        ).transpose(0,4,1,5,2,6,3,7).merge_axes((2,2,2,2), order=(1,1,-1,-1))
    return doubleT

def build_dbts(
    ops: list[None | GTensor], 
    Ta: GTensor, Tb: GTensor, 
    Tc: None | GTensor=None, Td: None | GTensor=None
):
    """
    Create double tensors 
    (using two 1-site operators)

    Impurity/Uniform tensor positions
    ```
            ↑   ↓           ↑   ↓
        ..→ W ← Z →..   ..→B/D←A/C→..
            ↓   ↑           ↓   ↑
        ..← X → Y ←..   ..← A → B ←..
            ↑   ↓           ↑   ↓
    ```
    
    Parameters
    ----
    ops: list[GTensor | None]
        operators acting on sites X, Y, Z, W of the plaquette
    Ta, Tb, Tc, Td: GTensor
        TPS tensors (loop convention)

    Return
    ----
    Double tensors X, Y, Z, W, A, B, (C|None), (D|None)
    """
    sites = ("x", "y", "z", "w")
    sublats = ("A", "B", "A", "B")
    if Tc is None:
        assert Td is None
        Tc, Td = Ta, Tb
        t4 = False
    else:
        assert isinstance(Tc, GTensor) and isinstance(Td, GTensor)
        t4 = True
    tensors = [Ta, Tb, Tc, Td]
    # uniform double tensors A,B,C,D
    dtensors = dict(
        (f"T{site}", double_ts1(t, None, sublat))
        for site, t, sublat in zip(sites, tensors, sublats)
    )
    dTa, dTb, dTc, dTd = tuple(
        dtensors[f"T{site}"] for site in sites
    )
    # double tensors with operators X,Y,Z,W
    op_dtensors = dict(
        (f"T{site}", double_ts1(t, op, sublat))
        for site, t, op, sublat in zip(sites, tensors, ops, sublats)
    )
    dTx, dTy, dTz, dTw = tuple(
        op_dtensors[f"T{site}"] for site in sites
    )
    return dTx, dTy, dTz, dTw, dTa, dTb, \
        (dTc if t4 else None), (dTd if t4 else None)
        
# ---- measuring with nearest neighbor 2-site gate ----

def build_dbts2(
    gate: GTensor, bond: str, Ta: GTensor, Tb: GTensor, 
    Tc: None | GTensor=None, Td: None | GTensor=None,
):
    """
    Create double tensors 
    (using 2-site gates on nearest neighbors)

    Impurity/Uniform tensor positions
    ```
            ↑   ↓           ↑   ↓
        ..→ W ← Z →..   ..→B/D←A/C→..
            ↓   ↑           ↓   ↑
        ..← X → Y ←..   ..← A → B ←..
            ↑   ↓           ↑   ↓
    ```
    
    Parameters
    ----
    gate: GTensor
        gate acting on two nearest neighbor sites
    bond: str 
        length-2 string specifying the sites to be acted on by the gate
        (left right) or (bottom top)

    Return
    ----
    Double tensors X, Y, Z, W, A, B, (C|None), (D|None)
    """
    sites = ("x", "y", "z", "w")
    sublats = ("A", "B", "A", "B")
    if Tc is None:
        assert Td is None
        Tc, Td = Ta, Tb
        t4 = False
    else:
        assert isinstance(Tc, GTensor) and isinstance(Td, GTensor)
        t4 = True
    tensors = [Ta, Tb, Tc, Td]
    # uniform double tensors A,B,C,D
    dtensors = dict(
        (f"T{site}", double_ts1(t, None, sublat))
        for site, t, sublat in zip(sites, tensors, sublats)
    )
    dTa, dTb, dTc, dTd = tuple(
        dtensors[f"T{site}"] for site in sites
    )
    # split the gate to mpo
    op1, op2 = gt.linalg.gate_to_mpo(gate)
    # horizontal bond
    if bond in ("xy", "wz"):
        tmp1, tmp2 = (
            (Ta, Tb) if bond == "xy"
            else (Td, Tc)
        )
        if bond == "wz":
            op1, op2 = gt.flip2_dual(op1, op2, [2,2])
        # minus signs when changing dual are absorbed to sub-lattice A
        dT1 = gt.tensordot(
            tmp1.gT.flip_dual((1,2,3,4), minus=(bond[0] == "x")), 
            gt.einsum("acb,cdefg->abdefg", op1, tmp1).merge_axes(
                (1,2,1,1,1), order=(1,)*5
            ), (0,0)
        ).transpose(0,4,1,5,2,6,3,7).merge_axes((2,2,2,2), order=(1,1,-1,-1))
        dT2 = gt.tensordot(
            tmp2.gT.flip_dual((1,2,3,4), minus=(bond[1] == "z")), 
            gt.einsum("acb,cdefg->adebfg", op2, tmp2).merge_axes(
                (1,1,1,2,1), order=(1,1,1,-1,1)
            ), (0,0)
        ).transpose(0,4,1,5,2,6,3,7).merge_axes((2,2,2,2), order=(1,1,-1,-1))
        dTx, dTy, dTz, dTw = (
            (dT1, dT2, dTc.copy(), dTd.copy()) if bond == "xy"
            else (dTa.copy(), dTb.copy(), dT2, dT1)
        )
    # vertical bond
    elif bond in ("xw", "yz"):
        tmp1, tmp2 = (
            (Ta, Td) if bond == "xw"
            else (Tb, Tc)
        )
        if bond == "xw":
            op1, op2 = gt.flip2_dual(op1, op2, [2,2])
        dT1 = gt.tensordot(
            tmp1.gT.flip_dual((1,2,3,4), minus=(bond[0] == "x")), 
            gt.einsum("acb,cdefg->adbefg", op1, tmp1).merge_axes(
                (1,1,2,1,1), order=(1,)*5
            ), (0,0)
        ).transpose(0,4,1,5,2,6,3,7).merge_axes((2,2,2,2), order=(1,1,-1,-1))
        dT2 = gt.tensordot(
            tmp2.gT.flip_dual((1,2,3,4), minus=(bond[1] == "z")), 
            gt.einsum("acb,cdefg->adefbg", op2, tmp2).merge_axes(
                (1,1,1,1,2), order=(1,1,1,1,-1)
            ), (0,0)
        ).transpose(0,4,1,5,2,6,3,7).merge_axes((2,2,2,2), order=(1,1,-1,-1))
        dTx, dTy, dTz, dTw = (
            (dT1, dTb.copy(), dTc.copy(), dT2) if bond == "xw"
            else (dTa.copy(), dT1, dT2, dTd.copy())
        )
    else:
        raise ValueError("Unrecognized nearest neighbor bond")
    return dTx, dTy, dTz, dTw, dTa, dTb, \
        (dTc if t4 else None), (dTd if t4 else None)
