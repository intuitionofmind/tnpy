"""
approxloop.py
=============
Optimize the octagon loop during coarse graining
(arXiv 1512.04938, Supplementary B)
"""

import logs
import mps.pbc as mps
from math import inf, sqrt
from typing import Sequence
import gtensor.linalg as gla
from gtensor import GTensor, tensordot

# import atexit
# import line_profiler as lp
# profile = lp.LineProfiler()
# atexit.register(profile.print_stats)

LEN_LOOP = 8
MAX_ITER = 150
INT_CHECK, INT_LOG = 10, 10
ITER_MED = 100
ERR_CUT0, ERR_CUT1, ERR_CUT2 = 0.99999, 0.999, 0.99
TOL = 2e-8
# np.seterr(all='raise')

def _range_mod(start: int, end: int, step=1, mod=8):
    """
    Custom-implemented cyclic range
    """
    start %= mod
    end %= mod
    for i in range(start, end+mod, step):
        iMod = i % mod
        if iMod == end:
            break
        yield iMod

def _process_idx(idx):
    if isinstance(idx, int):
        return idx % LEN_LOOP
    elif isinstance(idx, Sequence):
        if not len(idx) == 2:
            raise ValueError('`idx` must be of length 2 if it is a sequence')
        idxFull = tuple(_range_mod(idx[0], idx[1]))
        if not 2 <= len(idxFull) < LEN_LOOP:
            raise ValueError(f'`idx` (if sequence) must specify a segment of length [2, {LEN_LOOP-1}].')
        return idxFull
    raise TypeError('`idx` must be int or sequence of int')

class RepoError(Exception):
    pass

class TensorRepo:
    """
    Stores the 4 exact (E) tensors on the square network
    and the 8 approximate (A) tensors on the octagon network obtained from SVD

    Conventions
    ----
    - "ket"s are always positioned in the lower row
    
    - "bra"s are always positioned in the upper row, and need t.gconj()
    
    - rank-4 AA/AE tensor indices are ordered as:
        ```
        ← 1---a†--b†--3 ←
              ↓   ↓
        → 0---a---b---2 →
        ```
    
    - Only kets and brakets are stored (excluding bras).
    """
    _repo: dict[str, dict[tuple[int, ...], GTensor]]

    def __init__(self):
        self._repo = {
            'A': dict(),    # 1/2-site approximate kets
            'E': dict(),    # exact kets
            'AA': dict(),   # approx†-approx tensors
            'AE': dict()    # approx†-exact tensors
        }
        self._braketEE = 0.0
        self._braketEE_calculated = False

    @property
    def braketEE(self):
        """Squared norm of the exact MPS"""
        if not self._braketEE_calculated:
            assert len(self._repo['E']) > 0
            self._braketEE = self.braket('E', 'E')
            self._braketEE_calculated = True
        return self._braketEE
    
    def update_all_approx(self, sApp: list[GTensor]):
        """Update all approx tensors"""
        if not isinstance(sApp, Sequence):
            raise TypeError
        if not len(sApp) == LEN_LOOP:
            raise ValueError
        self._repo['A'] = dict()
        self._repo['AA'] = dict()
        self._repo['AE'] = dict()
        for i in range(LEN_LOOP):
            self._repo['A'][i] = sApp[i]
    
    def update_approx(self, i: int, x: GTensor):
        """Update the i-th approx tensor"""
        for label in self._repo:
            if 'A' not in label:
                continue
            for idx in tuple(self._repo[label].keys()):
                try:
                    if i in idx:
                        del self._repo[label][idx]
                except TypeError:
                    pass
        assert 'A' in self._repo
        self._repo['A'][i] = x
    
    def get_approx(self) -> list[GTensor]:
        """return a list of approximated tensors `Sapp`"""
        s = []
        for i in range(LEN_LOOP):
            s.append(self._repo['A'][i])
        return s
    
    def get(self, label: str, idx: tuple[int, int]):
        """
        Get tensor from repo. 
        Return `None` if the tensor has not been calculated yet.
        """
        if label == 'EE':
            return None
        if label not in self._repo:
            raise ValueError('wrong value for `label`')
        idx = _process_idx(idx)
        return self._repo[label].get(idx, None)
    
    def add(self, label: str, idx: tuple[int, int], tensor: GTensor):
        """
        Newly added tensor should not already be in repository;
        update is only allowed through `update_approx` and `update_all_approx`
        
        Parameters
        -------------
        label:  str ('A', 'E', 'AA', 'AE', or 'EE')

        idx:    an int or a sequence. 
            If a sequence, must be of the form 
            `(left_end, right_end) mod LEN_LOOP `
            (follow Python `range` convention)
        """
        if label == 'EE':   # 'EE' brakets are one-off's
            return
        if label not in self._repo:
            raise ValueError('wrong value for `label`')
        if not isinstance(tensor, GTensor):
            raise TypeError('`tensor` must be a Grassmann tensor')
        idx = _process_idx(idx)
        if idx in self._repo[label]:
            import sys
            print(label, idx, file=sys.stderr)
            print(self._repo, file=sys.stderr)
            raise RepoError
        self._repo[label][idx] = tensor

    def cost_func(self):
        """
        Calculate cost function
        """
        braketAA = self.braket('A', 'A')
        braketAE = self.braket('A', 'E')
        braketEE = self.braketEE
        absE2 = braketEE + braketAA - 2 * braketAE.real
        relE = sqrt(abs(absE2.real/braketEE.real))
        return relE.real, absE2.real

    def braket(self, braLabel: str, ketLabel: str):
        """full inner product, resulting in a scalar"""
        if not (braLabel in ('A', 'E') and ketLabel in ('A', 'E')):
            raise ValueError('wrong labels')
        label = braLabel+ketLabel
        for i in range(0,8,2):
            for j in (4,6):
                a = self.get(label, (i, i+j))
                b = self.get(label, (i+j, i+8))
                if a and b:
                    return tensordot(a, b, [(2,3,0,1), (0,1,2,3)]).item()
        for i in range(0,8,2):
            a = self.get(label, (i, i+4))
            b = self.get(label, (i+4, i+6))
            c = self.get(label, (i+6, i+8))
            if a and b and c:
                ab = tensordot(a, b, [(2,3),(0,1)])
                return tensordot(ab, c, [(2,3,0,1), (0,1,2,3)]).item()
        for i in range(0,8,2):
            if self.get(label, (i, i+6)):
                a = self.braket_6x2(braLabel, ketLabel, i)
                b = self.braket_2x2(braLabel, ketLabel, i+6)
                return tensordot(a, b, [(2,3,0,1), (0,1,2,3)]).item()
        a = self.braket_6x2(braLabel, ketLabel, 0)
        b = self.braket_2x2(braLabel, ketLabel, 6)
        return tensordot(a, b, [(2,3,0,1), (0,1,2,3)]).item()

    def braket_7x2(self, braLabel: str, ketLabel: str, i: int):
        """
        Calculate 7-site tensor
        """
        if i % 2 == 0:
            a = self.braket_6x2(braLabel, ketLabel, i)
            b = self.braket_1x2(braLabel, ketLabel, i+6)
        else:
            a = self.braket_1x2(braLabel, ketLabel, i)
            b = self.braket_6x2(braLabel, ketLabel, i+1)
        return tensordot(a, b, [(2,3),(0,1)])

    def bra7_ket8(self, braLabel: str, ketLabel: str, i: int):
        r"""
        Calculate the W_i tensors
        ```
        Example: i = 5

        |-------------------------------------------------|
        |--A0†---A1†---A2†---A3†---A4†- 1   0 -A6†---A7†--| 
            \    /      \    /      \    2      \    /
             \  /        \  /        \  /        \  /
        |---- E0 -------- E1 -------- E2 -------- E3 -----|
        |_________________________________________________|
        ```
        """
        if i % 2 == 0:
            r"""
                3   1         1    0           1     0
               /     \       /      \         /       \
            ..A       .. =>    2  3    =>  ...    2    ...
               \     /       \_|__|_/         \___|___/
                2   0
            """
            a = self.braket_6x2(braLabel, ketLabel, i+2)
            b = self.ket_2(ketLabel, i)
            ab = tensordot(a, b, [(0,2), (3,0)])
            s = self.get(braLabel, i+1)
            return tensordot(s.gconj(), ab.pconj([3]), [(0,1), (0,3)])
        else:
            a = self.braket_6x2(braLabel, ketLabel, i+1)
            b = self.ket_2(ketLabel, i-1)
            ab = tensordot(a, b, [(0,2), (3,0)])
            s = self.get(braLabel, i-1)
            return tensordot(
                s.gconj(), ab.pconj([2]), [(2,1), (1,2)]
            ).transpose(1,0,2)

    def braket_6x2(self, braLabel: str, ketLabel: str, i: int):
        """
        Calculate 6-site tensor
        """
        def _compute(braLabel: str, ketLabel: str, i: int):
            """
            Calculate 6-site tensor using 
            previously calculated 2/4-site tensors
            """
            label = braLabel + ketLabel
            for j in (2,4):
                a = self.get(label, (i,i+j))
                b = self.get(label, (i+j,i+6))
                if a and b:
                    return tensordot(a, b, [(2,3),(0,1)])
            a = self.get(label, (i,i+4))
            if a:
                b = self.braket_2x2(braLabel, ketLabel, i+4)
                return tensordot(a, b, [(2,3),(0,1)])
            b = self.get(label, (i+2,i+6))
            if b:
                a = self.braket_2x2(braLabel, ketLabel, i)
                return tensordot(a, b, [(2,3),(0,1)])
            a = self.braket_4x2(braLabel, ketLabel, i)
            b = self.braket_2x2(braLabel, ketLabel, i+4)
            return tensordot(a, b, [(2,3),(0,1)])
        
        if not i % 2 == 0:
            raise ValueError
        label = braLabel + ketLabel
        t = self.get(label, (i, i+6))
        if t:
            return t
        t = _compute(braLabel, ketLabel, i)
        self.add(label, (i,i+6), t)
        return t

    def braket_4x2(self, braLabel: str, ketLabel: str, i: int):
        """
        Calculate 4-site tensor
        """
        if not i % 2 == 0:
            raise ValueError
        label = braLabel + ketLabel
        t = self.get(label, (i, i+4))
        if t:
            return t
        a = self.braket_2x2(braLabel, ketLabel, i)
        b = self.braket_2x2(braLabel, ketLabel, i+2)
        t = tensordot(a, b, [(2,3),(0,1)])
        self.add(label, (i,i+4), t)
        return t

    def braket_2x2(self, braLabel: str, ketLabel: str, i: int):
        r"""
        Calculate 2-site tensor
        ```
            1 -←-u1†-←-u2†-←- 3
                |     |
            0 -→-d1--→-d2--→- 2
        ```
        """
        if not i % 2 == 0:
            raise ValueError
        label = braLabel + ketLabel
        t = self.get(label, (i, i+2))
        if t:
            return t
        bra = self.ket_2(braLabel, i)
        ket = self.ket_2(ketLabel, i)
        t = tensordot(
            bra.gconj(), ket.pconj([1,2]), [(1,2),(2,1)]
        ).transpose(2,1,3,0)
        self.add(label, (i,i+2), t)
        return t

    def braket_1x2(self, braLabel: str, ketLabel: str, i: int):
        """
        Calculate 1-site tensor
        ```
            1 --- A†--- 3
                  |
                  |
            0 --- A --- 2
        ```
        """
        bra = self.get(braLabel, i)
        ket = self.get(ketLabel, i)
        return tensordot(
            bra.gconj(), ket.pconj([1]), [[1], [1]]
        ).transpose(2,1,3,0)

    def ket_2(self, label: str, i: int):
        r"""
        Combine two S tensors into one T
        ```
                                      1   2
                |      |        =      \ /
            -→-S[i]-→-S[i+1]-→-     0 → T → 3
        ```
        """
        if not i % 2 == 0:
            raise ValueError
        t = self.get(label, (i,i+2))
        if t:
            return t
        a = self.get(label, i)
        b = self.get(label, i+1)
        t = tensordot(a, b, [2,0])
        self.add(label, (i,i+2), t)
        return t


# @profile
def approxloop(
    Sapp: list[GTensor], Ta: GTensor, Tb: GTensor, 
    Tc=None, Td=None, maxloop=MAX_ITER
):
    r"""
    Loop optimization of the SVD results on the octagon

        (1)     1         1     (0)
                ↑         ↓
            2 →B/D← 0 2 ←A/C→ 0
                ↓         ↑
                3         3
                1         1
                ↓         ↑
            2 ← A → 0 2 → B ← 0
                ↑         ↓
        (2)     3         3     (3)

            ↓  to MPS form (tensors are transposed)

            1   2       1   2       1   2       1   2
             ↖ ↙         ↖ ↙         ↖ ↙         ↖ ↙
        → 0 →A/C→ 3 → 0 →B/D→ 3 → 0 → A → 3 → 0 → B → 3 →
             (0)         (1)         (2)         (3)
    """
    assert maxloop >= 0
    assert len(Sapp) == LEN_LOOP
    solve_axes = [[0,2],[2,0]]
    repo = TensorRepo()
    if (Tc is None) and (Td is None):
        repo.add('E', (0,2), Ta.transpose(3,0,1,2))
        repo.add('E', (2,4), Tb)
    else:
        assert isinstance(Tc, GTensor)
        assert isinstance(Td, GTensor)
        repo.add('E', (0,2), Tc.transpose(3,0,1,2))
        repo.add('E', (2,4), Td)
    repo.add('E', (4,6), Ta.transpose(1,2,3,0))
    repo.add('E', (6,8), Tb.transpose(2,3,0,1))     
    repo.update_all_approx(Sapp)
    
    # main iteration loop
    relE_old = inf
    if maxloop == 0:
        # calculate truncation error 
        # when loop optimization is not enabled
        relE, absE2 = repo.cost_func()
        writeErrorToLog(0, relE, absE2, repo.braketEE)
    for step in range(maxloop):
        # check error and convergence every INT_CHECK steps
        if step % INT_CHECK == 0:
            # Canonicalize approx tensors
            if step > 0:
                sAppTmp = repo.get_approx()
                sAppTmp, _, svdErrs, qrErr = mps.canonicalize(sAppTmp)
                repo.update_all_approx(sAppTmp)
                del sAppTmp
            relE, absE2 = repo.cost_func()
            # check convergence
            if relE >= 1e-2:
                converge = (relE / relE_old > ERR_CUT0)
            elif step <= ITER_MED:
                converge = (relE < TOL) or (relE / relE_old > ERR_CUT1)
            else:
                converge = (relE < TOL) or (relE / relE_old > ERR_CUT2)
            if converge or (step % INT_LOG==0):
                writeErrorToLog(step, relE, absE2, repo.braketEE)
            if converge:
                info = 'Converged at it = {}. relE / relE_old = {:.5f}\n'.format(step, relE/relE_old)
                try:
                    logs.error.write(info)
                except AttributeError:
                    print(info, end="")
                break
            relE_old = relE
        
        # loop through 8 S tensors
        for i in range(LEN_LOOP):
            n = repo.braket_7x2('A', 'A', i+1)
            w = repo.bra7_ket8('A', 'E', i)
            r"""
            |-----------------|     |-------------|
            |--- 3       1 ---|     |--- 1   0 ---|

                     1           =         2       
                     |                     |       
            |-- 2 0-----2 0 --|     |-------------|
            |_________________|     |_____________|
            """
            try:
                x = gla.solve(n, w, solve_axes)
            except RuntimeError:
                logs.error.write('Warning: a is singular. optim_it, i = {:4d}, {:2d}\n'.format(step, i) )
                x, lserr = gla.lstsq(n, w, solve_axes, return_err=True)
                logs.error.write('least square error = {:.3E}\n'.format(lserr))
            repo.update_approx(i, x)
    return repo.get_approx()


def writeErrorToLog(
    step: int, relE: float, absE2: float, ee: float
):
    info = f'iter = {step:<4d}, '
    info += '(relErr, absErr, norm, absErr^2) '
    info += '{:11.4E}{:9.1E}{:9.1E}{:9.1E}\n'.format(relE, sqrt(abs(absE2)), sqrt(ee.real), absE2)
    try:
        logs.error.write(info)
    except AttributeError:
        print(info, end="")
