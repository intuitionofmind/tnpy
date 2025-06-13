"""
Coarse graining for square lattice
"""

import gtensor as gt
from gtensor import GTensor

def coarsegrain(S: list[GTensor]):
    """
    Form new tensor from the octagon-square network
    ```
                (A-new)
             ↑    ↓
        (B)  S2 ← S1  (A-old)
            ↙       ↖
        → S3          S0 →
          ↓           ↑   (B-new)
        ← S4          S7 ←
            ↘       ↗ 
        (A)  S5 → S6  (B-old)
             ↑    ↓
    ```
    """
    a = combineA(S[6], S[5], S[2], S[1])
    b = combineB(S[3], S[0], S[7], S[4])
    return a, b

def coarsegrain_impure(S: list[GTensor], Su: list[GTensor]):
    """
        Form new tensor from the octagon-square network
        (Su means the S tensors obtained from the uniform part of the network)
    """
    x = combineA(S[6], S[5], Su[2], Su[1])
    y = combineB(Su[3], S[0], S[7], Su[4])
    z = combineA(Su[6], Su[5], S[2], S[1])
    w = combineB(S[3], Su[0], Su[7], S[4])
    return x, y, z, w

def combineA(s0: GTensor, s1: GTensor, s2: GTensor, s3: GTensor):
    """
    Form new tensor A
    ```
        0                   2
          ↘               ↗
            s1--2 → 0--s0
            ↑           ↓
            1           1
            1           1
            ↑           ↓
            s2--0 ← 2--s3
          ↙               ↖
        2                   0
    ```

    Result (counter-clockwise rotation by 45 deg)
    ```
        1       0
          ↘   ↗
            A
          ↙   ↖
        2       3
    ```
    """
    a = gt.tensordot(s0, s1, [0,2])
    b = gt.tensordot(s2, s3, [0,2])
    return gt.tensordot(a, b, ((0,3),(3,0)))

def combineB(s0: GTensor, s1: GTensor, s2: GTensor, s3: GTensor):
    """
    Form new tensor B

        2                   0
          ↖               ↙
            s1 → 1 1 → s0
            ↑           ↓
            0           2
            2           0
            ↑           ↓
            s2--1 ← 1--s3
          ↗               ↘
        0                   2

    Result (counter-clockwise rotation by 45 deg)

        1       0
          ↖   ↙
            B
          ↗   ↘
        2       3
    """
    a = gt.tensordot(s0, s1, (1,1))
    b = gt.tensordot(s2, s3, (1,1))
    return gt.tensordot(a, b, ((1,2),(2,1)))
