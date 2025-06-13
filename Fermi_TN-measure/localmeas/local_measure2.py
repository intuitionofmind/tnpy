"""
The wrapped local measure functions can:
- construct the operators given their names;
- use various methods to measure the same quantity. 
"""

from gtensor import GTensor
from itertools import product
from update_ftps.sutools import get_tpstJconv
from . import local_measure as lm
from phys_models.onesiteop import makeops
from utils import split_measkey

# each nearest neighbor bond and the weight on it
wtkey_bonds = {
    "x1": "xy1", "x2": "wz1", "y1": "yz1", "y2": "xw1",
    "x1_": "xy2", "x2_": "wz2", "y1_": "yz2", "y2_": "xw2",
}
bond_wtkeys = {
    "xy1": "x1", "yz1": "y1", "wz1": "x2", "xw1": "y2",
    "xy2": "x1_", "yz2": "y1_", "wz2": "x2_", "xw2": "y2_",
}

def get_bonds1(t4: bool):
    """
    Get the list of all 1st neighbor bond names
    """
    if t4:
        return ["xy1", "yz1", "wz1", "xw1", "xy2", "yz2", "wz2", "xw2"]
    else:
        return ["xy1", "yz1", "wz1", "xw1"]


def get_bonds2(t4: bool):
    """
    Get the list of all 2nd neighbor bond names
    """
    return list(
        bond + str(plq) for bond, plq in
        product(["xz", "wy"], [1,2,3,4] if t4 else [1,3])
    )


def check_t4(measures: dict[str]):
    """
    Determine if current measurement is performed 
    on a TPS of type AB or generic-2x2
    """
    t4 = False
    for key in measures.keys():
        if "-" in key: continue
        if key[-1] == "2": t4 = True; break
    return t4


def meas_1site(
    opname: str, tname: str, 
    tensors: dict[str, GTensor], weights: dict[str, GTensor],
    mode="site", model='tJ'
) -> dict[str, float|complex]:
    """Measure one-site quantities"""
    assert mode in ("site", "bond", "loop")
    def makeops2(name: str):
        if model == 'tJ':
            tJ_conv = get_tpstJconv(tensors)
        else:
            tJ_conv = None
        return makeops(name, model, tJ_conv=tJ_conv)

    def _get_measkey(tname: str, opname: str):
        """
        Get measurement key:
        bond + (operator on the left/bottom) + (operator on the right/top)
        """
        bond = ("xy" if tname in ("Ta", "Tb") else "wz")
        term = (
            opname + "Id" if tname in ("Ta", "Td")
            else "Id" + opname
        )
        return bond, term

    def _get_bondops(tname: str, opname: str) -> list[None | GTensor]:
        """Bond operators for measuring one-site quantities"""
        bondops = (
            [makeops2(opname), None]
            if tname in ("Ta", "Td") else
            [None, makeops(opname)]
        )
        return bondops

    def _get_loopops(tname: str, opname: str) -> list[None | GTensor]:
        """Loop operators (on red loop) 
        for measuring one-site quantities"""
        loopops = [None] * 4
        op_id = (
            0 if tname == "Ta" else 1 if tname == "Tb"
            else 2 if tname == "Tc" else 3
        )
        loopops[op_id] = makeops2(opname)
        return loopops

    (bond, term), plq = _get_measkey(tname, opname), "1"
    meas = {
        bond + term + plq: lm.meas_site(
            tname, makeops2(opname), tensors, weights
        ) if mode == "site" else lm.meas_bond(
            bond_wtkeys[bond+plq],
            _get_bondops(tname, opname), tensors, weights
        ) if mode == "bond" else lm.meas_loop(
            1, _get_loopops(tname, opname), tensors, weights
        )
    }
    return meas


def meas_2site(
    term: str, bond_plq: str, 
    tensors: dict[str, GTensor], weights: dict[str, GTensor],
    mode="bond", model='tJ'
) -> dict[str, float|complex]:
    """
    Measure two-site quantities

    Site order for `term`
    ----
    [left, right] or [bottom, top]

    Parameters
    ----
    term: str
        name of the two operators on the two sites
        (e.g. SpSm, CpuCmu, NhId)
    bond_plq: str
        bond name and the plaquette it is on
        (e.g. xy1, wz2)
    mode: str ("bond", "loop")
        "bond" can measure 1st neighbor terms only;
        "loop" can measure both 1st and 2nd neighbor terms
    """
    assert mode in ("bond", "loop")
    def makeops2(name: str):
        if model == 'tJ':
            tJ_conv = get_tpstJconv(tensors)
        else:
            tJ_conv = None
        return makeops(name, model, tJ_conv=tJ_conv)

    def _get_bondops(term: str) -> list[GTensor]:
        """Bond operators for measuring one-site quantities"""
        bondops = [
            makeops2(opname)
            for opname in split_measkey(term)
        ]
        return bondops

    def _get_loopops(bond: str, term: str) -> list[None | GTensor]:
        """
        operators acting on the loop
        
        Site order
        ----
        ```
            1(W) --- 3(Z)
            |        |
            0(X) --- 2(Y)
        ```
        """
        def _xyzw_to_id(site: str):
            if site == "x":     return 0
            elif site == "y":   return 2
            elif site == "z":   return 3
            elif site == "w":   return 1
            else: raise ValueError("Unrecognized site name")

        loopops = [None] * 4
        op_ids = [_xyzw_to_id(s) for s in bond]
        opnames = split_measkey(term)
        assert len(opnames) == 2
        for op_id, opname in zip(op_ids, opnames):
            loopops[op_id] = makeops2(opname)
        return loopops

    bond, plq = split_measkey(bond_plq)
    meas = {
        bond + term + plq: lm.meas_bond(
            bond_wtkeys[bond_plq],
            _get_bondops(term), tensors, weights
        ) if mode == "bond" else lm.meas_loop(
            int(plq), _get_loopops(bond, term), tensors, weights
        )
    }
    return meas


def meas_loop(
    opnames: list[str], loopname: int,
    tensors: dict[str, GTensor], weights: dict[str, GTensor],
    model='tJ'
):
    """
    Measure 4-site quantities on 2x2 plaquette

    Site order for `opnames`
    ----
    ```
        1 --- 3
        |     |
        0 --- 2
    ```

    loop names
    ----
    ```
        A ---x1---- B ---x2_--- A
        |           |           |
        y1_   4     y2_   2     y1_
        |           |           |
        D ---x2---- C ---x1_--- D
        |           |           |
        y2    1     y1    3     y2
        |           |           |
        A ---x1---- B ---x2_--- A
    ```
    """
    def makeops2(name: str):
        if model == 'tJ':
            tJ_conv = get_tpstJconv(tensors)
        else:
            tJ_conv = None
        return makeops(name, model, tJ_conv=tJ_conv)
    return lm.meas_loop(
        loopname, 
        [makeops2(opname) for opname in opnames], 
        tensors, weights
    )

def meas_4site(
    opnames: list[str], direction: str,
    tensors: dict[str, GTensor], weights: dict[str, GTensor],
    model='tJ'
):
    """
    Measure 4-site quantities on the same row/column
    """
    assert direction in ('h', 'v')
    def makeops2(name: str):
        if model == 'tJ':
            tJ_conv = get_tpstJconv(tensors)
        else:
            tJ_conv = None
        return makeops(name, model, tJ_conv=tJ_conv)
    return lm.meas_site4(
        [makeops2(opname) for opname in opnames], 
        direction, tensors, weights
    )

