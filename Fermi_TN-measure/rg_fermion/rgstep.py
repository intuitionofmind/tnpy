"""
rgstep.py
============
Perform tensor network renormalization 
on 2D square lattice tensor network
"""

from gtensor import GTensor
from .approxloop import approxloop
from . import coarsegrain as cg
from . import norm
import logs
from . import filtering as flt
import numpy as np


def rgstep(
    paramRG: dict[str], Ta: GTensor, Tb: GTensor, 
    Tc: None | GTensor=None, Td: None | GTensor=None, 
    clockwise=False
):
    """
    One step of RG of uniform part of 
    2D bi-partite square network

    Input tensor convention
    ```
        A, C        B, D
            1           1    
            ↓           ↑
        2 ← A → 0   2 → B ← 0
            ↑           ↓
            3           3
    ```

    Entanglement filtering is performed only when 
    `Tc, Td = None, None`
    """
    fltOn = paramRG["flt"]
    maxloop = paramRG["maxloop"]
    d_cut = paramRG["d_cutRG"]
    if Tc is None:
        assert Td is None
        for name, t in zip("AB", [Ta, Tb]):
            logs.error.write(f"shape of {name} = {t.shape}, parity = {t.parity}\n")
    else:
        assert isinstance(Tc, GTensor) and isinstance(Td, GTensor)
        for name, t in zip("ABCD", [Ta, Tb, Tc, Td]):
            logs.error.write(f"shape of {name} = {t.shape}, parity = {t.parity}\n")
    
    if fltOn is True:
        if Tc is None:
            assert Td is None
            Ta, Tb, Pas, Pbs = flt.square(Ta, Tb, norm_check=True)
            print("ABAB: filtering done")
        else:
            Pas, Pbs = None, None
            print("ABCD: skip filtering on 1st RG step")
    else: 
        Pas, Pbs = None, None
        
    # canonicalization on the octagon
    Sapp_u = flt.octagon(d_cut, Ta, Tb, Tc, Td, canon=True)
    print("ABAB: square to octagon done")
    if maxloop > 0: logs.error.write("ABAB: optimizing\n")
    Sapp_u = approxloop(Sapp_u, Ta, Tb, Ta, Tb, maxloop)
    if maxloop > 0: print("ABAB: loop optimization done")
    
    Ta, Tb = cg.coarsegrain(Sapp_u)
    # rotate the network back to the original orientation
    if clockwise is True:
        print("ABAB: restore network orientation")
        Ta, Tb = Tb.transpose(3,0,1,2), Ta.transpose(3,0,1,2)
    print("ABAB: coarsegrain done")
    
    return Ta, Tb, Pas, Pbs, Sapp_u


def rgstep_impure(
    paramRG: dict[str], 
    Tx: GTensor, Ty: GTensor, Tz: GTensor, Tw: GTensor, 
    Pas: list[GTensor] | None, Pbs: list[GTensor] | None, 
    Sapp_u: list[GTensor], clockwise=False
):
    """
    One step of RG of 2D square network with impurity center

    Input tensors adopt the loop convention of gMetrics
    ```
        X, Z        Y, W
            1           1    
            ↓           ↑
        2 ← A → 0   2 → B ← 0
            ↑           ↓
            3           3
    ```

    Entanglement filtering only applies to `Tc, Td = None` case

    Parameters
    ----
    Pas, Pbs: list[GTensor]
        Projectors from filtering uniform (ABAB) part of the network in the same RG step
    Sapp_u: list[GTensor]
        The 8 octagon tensors from coarsegraining uniform (ABAB) part of the network in the same RG step
    """
    fltOn = paramRG["flt"]
    maxloop = paramRG["maxloop"]
    d_cut = paramRG["d_cutRG"]
    for name, t in zip("XYZW", [Tx, Ty, Tz, Tw]):
        logs.error.write(
            f"shape of {name} = {t.shape}, parity = {t.parity}\n"
        )
    
    if fltOn is True and Pas is not None: 
        assert Pbs is not None
        Tx, Ty, Tz, Tw = flt.square_impure(Tx, Ty, Tz, Tw, Pas, Pbs)
        print("XYZW: filtering done")
        
    Sapp = flt.octagon(d_cut, Tx, Ty, Tz, Tw, canon=True)
    print("XYZW: square to octagon done")
    
    # if maxloop = 0, the truncation error is still calculated.
    if maxloop > 0: logs.error.write("XYZW: optimizing\n")
    Sapp = approxloop(Sapp, Tx, Ty, Tz, Tw, maxloop)
    if maxloop > 0: print("XYZW: loop optimization done")
    
    Tx, Ty, Tz, Tw = cg.coarsegrain_impure(Sapp, Sapp_u)
    # rotate the network back to the original orientation
    if clockwise is True:
        print("XYZW: restore network orientation")
        Ty, Tz, Tw, Tx = \
            Tx.transpose(3,0,1,2), Ty.transpose(3,0,1,2), \
            Tz.transpose(3,0,1,2), Tw.transpose(3,0,1,2)
    print("XYZW: coarsegrain done")
    return Tx, Ty, Tz, Tw, Sapp


def normalize(
    Ta: GTensor, Tb: GTensor, 
    Tc: None | GTensor=None, Td: None | GTensor=None
):
    """
    Use norm4(Ta, Tb, Tc, Td)**(1/4) to normalize network tensors
    """
    fac = norm.norm4(Ta, Tb, Tc, Td)
    # always use a number with positive real part to normalize
    sgn = 1
    if fac.real < 0: 
        fac *= -1; sgn = -1
    fac = np.around(fac**0.25, decimals=14)
    if Tc is None:
        return fac, sgn, Ta/fac, Tb/fac
    else:
        return fac, sgn, Ta/fac, Tb/fac, Tc/fac, Td/fac
