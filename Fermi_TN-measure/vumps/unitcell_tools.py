from fermiT import FermiT

def relabel_unitcell(
    fAss: list[list[FermiT]],
    direction: str, reverse=False
) -> list[list[FermiT]]:
    """
    Relabel tensors in a unit cell

    Parameters
    ----
    fAss: list[list[FermiT]]
        tensors in a unit cell
    direction: str ("up", "down", "left", "right")
        the base direction
    reverse: bool
        if True, perform reverse relabelling

    Details
    ----
    When `reverse is False`, 
    the input label order is for direction "down"
    (e.g. 3 x 4 unit cell)
    ```
        (2,0) (2,1) (2,2) (2,3)
        (1,0) (1,1) (1,2) (1,3)
        (0,0) (0,1) (0,2) (0,3)
        |-----|-----|-----|
    ```

    - For up MPS, we will relabel fAss as

        ```
        |-----|-----|-----|
        (0,0) (0,1) (0,2) (0,3)
        (1,0) (1,1) (1,2) (1,3)
        (2,0) (2,1) (2,2) (2,3)
        ```
    
    - For left MPS, we will relabel fAss as

        ```
        |-- (0,2) (1,2) (2,2) (3,2)
        |-- (0,1) (1,1) (2,1) (3,1)
        |-- (0,0) (1,0) (2,0) (3,0)
        ```

    - For right MPS, we will relabel fAss as

        ```
        (3,2) (2,2) (1,2) (0,2) --|
        (3,1) (2,1) (1,1) (0,1) --|
        (3,0) (2,0) (1,0) (0,0) --|
        ```
    """
    N1, N2 = len(fAss), len(fAss[0])
    if (reverse) and (direction in ("left", "right")): 
        N1, N2 = N2, N1
    iter_down = [(i,j) for i in range(N1) for j in range(N2)]
    if direction == "up":
        fApss = [[None]*N2 for _ in range(N1)]
        for i, j in iter_down:
            fApss[N1-i-1][j] = fAss[i][j]
        fAss = fApss
    elif direction == "down":
        pass
    elif direction == "left":
        if not reverse:
            fApss = [[None]*N1 for _ in range(N2)]
            for i, j in iter_down:
                fApss[j][i] = fAss[i][j]
        else:
            fApss = [[None]*N2 for _ in range(N1)]
            for i, j in iter_down:
                fApss[i][j] = fAss[j][i]
        fAss = fApss
    elif direction == "right":
        if not reverse:
            fApss = [[None]*N1 for _ in range(N2)]
            for i, j in iter_down:
                fApss[N2-j-1][i] = fAss[i][j]
        else:
            fApss = [[None]*N2 for _ in range(N1)]
            for i, j in iter_down:
                fApss[i][j] = fAss[N2-j-1][i]
        fAss = fApss
    else:
        raise ValueError("Unrecognized direction")
    return fAss

def rotate_unitcell(
    fAss: list[list[FermiT]],
    direction: str, reverse=False
):
    """
    Rotate (transpose) tensors in the unit cell

    Parameters
    ----
    fAss: list[list[FermiT]]
        tensors in a unit cell
    direction: str ("up", "down", "left", "right")
        the base direction
    reverse: bool
        if True, perform reverse rotation

    Details
    ----
    Input PEPS tensor dual and axis order
    (for up MPS)
    ```
                U
                3  0
                ↑ /
        L   2 → A → 4   R
                ↑
                1
                D
    ```

    - For down MPS, transpose to

        ```
            1  0
            ↑ /
        2 → A → 4
            ↑
            3
        ```

    - For left MPS, transpose to

        ```
            4  0
            ↑ /
        3 → A → 1
            ↑
            2
        ```

    - For right MPS, transpose to (mirror by y = x)

        ```
            4  0
            ↑ /
        1 → A → 3
            ↑
            2
        ```
    """
    if reverse:
        raise NotImplementedError
    assert direction in (
        "up", "down", "left", "right"
    ), "Unrecognized boundary MPS position"
    perm = (
        [0,1,2,3,4] if direction == "up"
        else [0,3,2,1,4] if direction == "down"
        else [0,4,1,2,3] if direction == "left"
        else [0,2,1,4,3] 
    )
    fApss = [[
        t.copy() if direction == "up" 
        else t.transpose(*perm) for t in fAs
    ] for fAs in fAss]
    return fApss
