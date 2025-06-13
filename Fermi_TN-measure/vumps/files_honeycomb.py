"""
Load and process PEPS on honeycomb lattice
(Reference: PhysRevB.108.035144)
"""

from .files_par import load_tensors
import fermiT as ft
# from fermiT import FermiT

def load_peps_hc(folder: str):
    r"""
    Load PEPS with weights on honeycomb lattice
    and convert to square lattice PEPS

    Honeycomb tensor axis order
    (matches the bond weight names)
    ```
        2           1                   1
         \         /                    |
          A - 3 - B     =>  2 - A - 3 - B - 2
         /         \            |
        1           2           1 
    ```

    Honeycomb bond weight axis order
    (from A to B)
    ```
        A → 0 → w ← 1 ← B
    ```

    Square tensor axis order
    (merged physical axis on sub-lattice A/B is at 0)
    ```
            3
            ↑ (B)
        2 ← T → 4
        (A) ↓
            1
    ```
    - The square lattice PEPS has 1-site unit cell
    - Two physical axes (A,B) on each site are merged
    - The x/y-weights are w2, w1

    Returns
    ----
    [[[T]], [[y]], [[x]]] = [site tensor, y-weight, x-weight]
    """
    fTs = load_tensors(5, folder)
    tA, tB, w1, w2, w3 = fTs
    t = ft.tensordot(tA, w3, [[3], [0]])
    t = ft.tensordot(t, tB, [[3], [3]]).transpose(0,3,1,2,4,5)
    t = ft.merge_axes(t, [[0,1],[2],[3],[4],[5]])
    """
    Now, the square lattice PEPS is
    (x/y weights are shown with axis order)

            y           y
            ↑           ↑
        x ← T → 1-x-0 ← T → x
            ↓           ↓
            0           0
            y           y
            1           1
            ↑           ↑
        x ← T → 1-x-0 ← T → x
            ↓           ↓
            y           y

    We still need to transpose the weights

    - x-weight
        T → 1-x-0 ← T ==> T → 0-x-1 ← T

    - y-weight
        T       T
        ↓       ↓
        0       1
        y  ==>  y
        1       0
        ↑       ↑
        T       T
    
    The extra minus sign in the odd sector 
    of the weight should be absorbed to T
    """
    t = t.flip_dual([3,4], change_dual=False)
    # return as square lattice PEPS with 1-site unit cell
    # [fGss, ftyss, ftxss] = [site tensor, y-weight, x-weight]
    return [[[t]], [[w1]], [[w2]]]
