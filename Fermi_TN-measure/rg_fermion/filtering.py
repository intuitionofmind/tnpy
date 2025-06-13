"""
filtering.py
============
Entanglement filtering on the tensor network   
Canonicalization on the octagon loop during coarse graining
"""

import logs
import mps.pbc as mps
import gtensor as gt
from gtensor import GTensor
import gtensor.linalg as gla
from .norm import norm4

def square(
    Ta: GTensor, Tb: GTensor, output=True, norm_check=True
):
    """
    Filter the square network
    ```
        (1)     1         1     (0)
                ↑         ↓
            2 → B ← 0 2 ← A → 0
                ↓         ↑
                3         3
                1         1
                ↓         ↑
            2 ← A → 0 2 → B ← 0
                ↑         ↓
        (2)     3         3     (3)

                ↓  to MPO form

            0   1       1   2       2   3       3   0
             ↖ ↙         ↖ ↙         ↖ ↙         ↖ ↙
        → 3 → A → 2 → 0 → B → 3 → 1 → A → 0 → 2 → B → 1 →
    ```
    """
    if norm_check is True:
        logs.error.write("Before filtering: |ABAB| = {:.6e}\n"\
            .format(norm4(Ta, Tb))
        )
    # Use QR to get projectors
    Pas, Pbs, _, svdErrs, qrErr = mps.get_projs([
        Ta.transpose(3,0,1,2), Tb, 
        Ta.transpose(1,2,3,0), Tb.transpose(2,3,0,1)
    ])
    if output:
        mps.print_qrErr(qrErr, logs.error)
        mps.print_svdErrs(svdErrs, logs.error)
    """
    (1) B--B1 ← A0--A (0)
        |           |
        A1          B0
        ↓           ↑
        B2          A3
        |           |
    (2) A--A2 → B3--B (3)
    """
    Ta = gt.tensordot_keepform(Ta, Pas[2], (0,0))
    Ta = gt.tensordot_keepform(Ta, Pbs[2], (1,1))
    Ta = gt.tensordot_keepform(Ta, Pas[0], (2,0))
    Ta = gt.tensordot_keepform(Ta, Pbs[0], (3,1))
    
    Tb = gt.tensordot_keepform(Tb, Pbs[1], (0,1))
    Tb = gt.tensordot_keepform(Tb, Pas[3], (1,0))
    Tb = gt.tensordot_keepform(Tb, Pbs[3], (2,1))
    Tb = gt.tensordot_keepform(Tb, Pas[1], (3,0))
    
    if norm_check is True:
        logs.error.write("After filtering: |ABAB| = {:.6e}\n"\
            .format(norm4(Ta, Tb))
        )
    return Ta, Tb, Pas, Pbs

def square_impure(
    Tx: GTensor, Ty: GTensor, Tz: GTensor, Tw: GTensor, 
    Pa2s: list[GTensor], Pb2s: list[GTensor],
    output=True, norm_check=True
):
    """
    Filter the square network with two adjacent impurity tensors

    Parameters
    ----
    Pa2s, Pb2s: list[GTensor]
        Projectors on the uniform (AB) part of the network
    """
    if norm_check is True:
        logs.error.write("Before filtering: |XYZW| = {:.6e}\n"\
            .format(norm4(Tx, Ty, Tz, Tw))
        )
    # QR parameters
    """
    Impurity square loop
        
        (1)     1         1     (0)
                ↑         ↓
            2 → W ← 0 2 ← Z → 0
                ↓         ↑
                3         3
                1         1
                ↓         ↑
            2 ← X → 0 2 → Y ← 0
                ↑         ↓
        (2)     3         3     (3)
    """
    Pa1s, Pb1s, _, svdErr1s, qrErr1 = mps.get_projs([
        Tz.transpose(3,0,1,2), Tw,
        Tx.transpose(1,2,3,0), Ty.transpose(2,3,0,1)
    ])
    if output:
        mps.print_qrErr(qrErr1, logs.error)
        mps.print_svdErrs(svdErr1s, logs.error)
    
    # apply the projectors
    Tz = gt.tensordot_keepform(Tz, Pa2s[2], (0,0))
    Tz = gt.tensordot_keepform(Tz, Pb2s[2], (1,1))
    Tz = gt.tensordot_keepform(Tz, Pa1s[0], (2,0))
    Tz = gt.tensordot_keepform(Tz, Pb1s[0], (3,1))
    
    Tw = gt.tensordot_keepform(Tw, Pb1s[1], (0,1))
    Tw = gt.tensordot_keepform(Tw, Pa2s[3], (1,0))
    Tw = gt.tensordot_keepform(Tw, Pb2s[3], (2,1))
    Tw = gt.tensordot_keepform(Tw, Pa1s[1], (3,0))
    
    Tx = gt.tensordot_keepform(Tx, Pa1s[2], (0,0))
    Tx = gt.tensordot_keepform(Tx, Pb1s[2], (1,1))
    Tx = gt.tensordot_keepform(Tx, Pa2s[0], (2,0))
    Tx = gt.tensordot_keepform(Tx, Pb2s[0], (3,1))
    
    Ty = gt.tensordot_keepform(Ty, Pb2s[1], (0,1))
    Ty = gt.tensordot_keepform(Ty, Pa1s[3], (1,0))
    Ty = gt.tensordot_keepform(Ty, Pb1s[3], (2,1))
    Ty = gt.tensordot_keepform(Ty, Pa2s[1], (3,0))
    
    if norm_check is True:
        # should be the same as before filtering
        logs.error.write("After filtering: |XYZW| = {:.6e}\n"\
            .format(norm4(Tx, Ty, Tz, Tw))
        )
    return Tx, Ty, Tz, Tw

def octagon(
    d_cut: int, Ta: GTensor, Tb: GTensor, 
    Tc: None | GTensor=None, Td: None | GTensor=None, canon=True
) -> list[GTensor]:
    """
    Filter the octagon network
    ```
             ↑    ↓
        (B/D)S2 ← S1(A/C)
            ↙       ↖
        → S3         S0 →
          ↓          ↑
        ← S4         S7 ←
            ↘       ↗ 
        (A)  S5 → S6  (B)
             ↑    ↓
    ```

    Parematers
    ----
    d_cut: int
        Total (even and odd) bond dimension cutoff
    canon: bool
        When True, perform canonicalization on the octagon
    """
    if Tc is None:
        assert Td is None
        Tc, Td = Ta, Tb
    else:
        assert isinstance(Tc, GTensor) and isinstance(Td, GTensor)
    # hard svd cutoff
    if canon is False:
        Sapp = [None] * 8
        Sapp[0], _, Sapp[1] = gla.svd(
            Tc.transpose(0,3,1,2), 2, absorb_s=True, 
            cutoff=True, Dmax=d_cut
        )
        Sapp[2], _, Sapp[3] = gla.svd(
            Td.transpose(1,0,2,3), 2, absorb_s=True, 
            cutoff=True, Dmax=d_cut
        )
        Sapp[4], _, Sapp[5] = gla.svd(
            Ta.transpose(2,1,3,0), 2, absorb_s=True, 
            cutoff=True, Dmax=d_cut
        )
        Sapp[6], _, Sapp[7] = gla.svd(
            Tb.transpose(3,2,0,1), 2, absorb_s=True, 
            cutoff=True, Dmax=d_cut
        )
        for i in range(0,8,2):
            Sapp[i] = Sapp[i].transpose(1,0,2)
    # use projector to perform dimension cutoff
    else:
        Sapp = [None] * 8
        Sapp[0], _, Sapp[1] = gla.svd(
            Tc.transpose(0,3,1,2), 2, cutoff=True, absorb_s=True
        )
        Sapp[2], _, Sapp[3] = gla.svd(
            Td.transpose(1,0,2,3), 2, absorb_s=True)
        Sapp[4], _, Sapp[5] = gla.svd(
            Ta.transpose(2,1,3,0), 2, absorb_s=True)
        Sapp[6], _, Sapp[7] = gla.svd(
            Tb.transpose(3,2,0,1), 2, cutoff=True, absorb_s=True
        )
        for i in range(0,8,2):
            Sapp[i] = Sapp[i].transpose(1,0,2)
        Sapp, _, svdErrs, qrErr = mps.canonicalize(Sapp, d_cut, eps=1.e-6)
        mps.print_qrErr(qrErr, logs.error)
        mps.print_svdErrs(svdErrs, logs.error)
    return Sapp
