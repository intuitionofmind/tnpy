import opt_einsum as oe
from copy import deepcopy
import itertools
import math

import torch
torch.set_default_dtype(torch.float64)

import tnpy as tp
from tnpy import GTensor, Z2gTensor

class SquareTJ(object):
    r'''
    t-J model on a square lattice
    '''

    def __init__(self, t: float, J: float, mu: float, schwinger_boson=False):
        r'''
        schwinger_boson: bool, if use the Schwinger boson representation
        '''

        self._dim_phys = 2
        self._t, self._J, self._mu = t, J, mu
        self._schwinger_boson = schwinger_boson

    def onsite_n(self) -> GTensor:
        r'''
        onsite particle-number operator
        '''

        dual = (0, 1)
        shape = tuple([(self._dim_phys, self._dim_phys)]*2)
        blocks = {}
        block_shape = tuple([self._dim_phys]*2)
        temp = torch.zeros(block_shape)
        temp[0, 0] = 1.0
        temp[1, 1] = 1.0

        qns = [1, 1]
        if self._schwinger_boson:
            qns = [q ^ 1 for q in qns]
        blocks[tuple(qns)] = temp

        return GTensor(dual, shape, blocks)

    def onsite_sz(self) -> GTensor:
        r'''
        onsite Sz operator
        '''

        dual = (0, 1)
        shape = tuple([(self._dim_phys, self._dim_phys)]*2)
        blocks = {}
        block_shape = tuple([self._dim_phys]*2)
        temp = torch.zeros(block_shape)
        temp[0, 0] = 0.5
        temp[1, 1] = -0.5
        
        qns = [1, 1]
        if self._schwinger_boson:
            qns = [q ^ 1 for q in qns]
        blocks[tuple(qns)] = temp

        return GTensor(dual, shape, blocks)

    def onsite_sx(self) -> GTensor:
        r'''
        onsite Sx operator
        '''

        dual = (0, 1)
        shape = tuple([(self._dim_phys, self._dim_phys)]*2)
        blocks = {}
        block_shape = tuple([self._dim_phys]*2)
        temp = torch.zeros(block_shape)
        temp[1, 0] = 0.5
        temp[0, 1] = 0.5

        qns = [1, 1]
        if self._schwinger_boson:
            qns = [q ^ 1 for q in qns]
        blocks[tuple(qns)] = temp

        return GTensor(dual, shape, blocks)

    def onsite_sy(self) -> GTensor:
        r'''
        onsite Sx operator
        '''

        dual = (0, 1)
        shape = tuple([(self._dim_phys, self._dim_phys)]*2)
        blocks = {}
        block_shape = tuple([self._dim_phys]*2)
        temp = torch.zeros(block_shape)
        temp[1, 0] = 0.5
        temp[0, 1] = -0.5

        qns = [1, 1]
        if self._schwinger_boson:
            qns = [q ^ 1 for q in qns]
        blocks[tuple(qns)] = temp

        return GTensor(dual, shape, blocks)

    def c_up(self) -> Z2gTensor:

        dual = (0, 1)
        shape = tuple([(self._dim_phys, self._dim_phys)]*2)
        blocks = {}
        temp = torch.zeros(tuple([self._dim_phys]*2))
        temp[0, 0] = 1.0

        qns = [0, 1]
        if self._schwinger_boson:
            qns = [q ^ 1 for q in qns]
        blocks[tuple(qns)] = temp

        return Z2gTensor(dual, shape, blocks)

    def c_up_dagger(self) -> Z2gTensor:

        dual = (0, 1)
        shape = tuple([(self._dim_phys, self._dim_phys)]*2)
        blocks = {}
        temp = torch.zeros(tuple([self._dim_phys]*2))
        temp[0, 0] = 1.0

        qns = [1, 0]
        if self._schwinger_boson:
            qns = [q ^ 1 for q in qns]
        blocks[tuple(qns)] = temp

        return Z2gTensor(dual, shape, blocks)

    def c_down(self) -> Z2gTensor:

        dual = (0, 1)
        shape = tuple([(self._dim_phys, self._dim_phys)]*2)
        blocks = {}
        temp = torch.zeros(tuple([self._dim_phys]*2))
        temp[0, 1] = 1.0

        qns = [0, 1]
        if self._schwinger_boson:
            qns = [q ^ 1 for q in qns]
        blocks[tuple(qns)] = temp

        return Z2gTensor(dual, shape, blocks)

    def c_down_dagger(self) -> Z2gTensor:

        dual = (0, 1)
        shape = tuple([(self._dim_phys, self._dim_phys)]*2)
        blocks = {}
        temp = torch.zeros(tuple([self._dim_phys]*2))
        temp[1, 0] = 1.0

        qns = [1, 0]
        if self._schwinger_boson:
            qns = [q ^ 1 for q in qns]
        blocks[tuple(qns)] = temp

        return Z2gTensor(dual, shape, blocks)

    def c_down_dagger_c_up(self) -> Z2gTensor:

        dual = (0, 1)
        shape = tuple([(self._dim_phys, self._dim_phys)]*2)
        blocks = {}
        temp = torch.zeros(tuple([self._dim_phys]*2))
        temp[1, 0] = 1.0

        qns = [1, 1]
        if self._schwinger_boson:
            qns = [q ^ 1 for q in qns]
        blocks[tuple(qns)] = temp

        return Z2gTensor(dual, shape, blocks)

    def c_up_dagger_c_down(self) -> Z2gTensor:

        dual = (0, 1)
        shape = tuple([(self._dim_phys, self._dim_phys)]*2)
        blocks = {}
        temp = torch.zeros(tuple([self._dim_phys]*2))
        temp[0, 1] = 1.0

        qns = [1, 1]
        if self._schwinger_boson:
            qns = [q ^ 1 for q in qns]
        blocks[tuple(qns)] = temp

        return Z2gTensor(dual, shape, blocks)

    def n_up(self) -> Z2gTensor:

        dual = (0, 1)
        shape = tuple([(self._dim_phys, self._dim_phys)]*2)
        blocks = {}
        temp = torch.zeros(tuple([self._dim_phys]*2))
        temp[0, 0] = 1.0

        qns = [1, 1]
        if self._schwinger_boson:
            qns = [q ^ 1 for q in qns]
        blocks[tuple(qns)] = temp

        return Z2gTensor(dual, shape, blocks)

    def n_down(self) -> Z2gTensor:

        dual = (0, 1)
        shape = tuple([(self._dim_phys, self._dim_phys)]*2)
        blocks = {}
        temp = torch.zeros(tuple([self._dim_phys]*2))
        temp[1, 1] = 1.0

        qns = [1, 1]
        if self._schwinger_boson:
            qns = [q ^ 1 for q in qns]
        blocks[tuple(qns)] = temp

        return Z2gTensor(dual, shape, blocks)

    def onebody_id(self) -> Z2gTensor:
        r'''
        identity closure condition
        '''

        # build the identity operator
        dual = (0, 1)
        shape = tuple([(self._dim_phys, self._dim_phys)]*2)
        block_shape = tuple([self._dim_phys]*2)

        temp = torch.zeros(block_shape)
        bare = torch.zeros(block_shape)
        blocks = {}

        if self._schwinger_boson:
            id_states = (0, 1), (0, 0), (1, 0)
        else:
            id_states = (0, 0), (0, 1), (1, 1)

        for l, n in id_states:
            temp.zero_()
            temp[l, l] = 1.0
            blocks[(n, n)] = blocks.get((n, n), bare)+temp

        return Z2gTensor(dual, shape, blocks)

    def twobody_hopping_mu(self) -> GTensor:
        r'''
        # H_{i'j'ij}^{n_{i'}n_{j'}n_{i}n_{j}}
        #   i'   j'
        #   |0   |0
        #   ^    ^
        #   |    |
        #   *----*
        #   |    |
        #   ^    ^
        #   |1   |1
        #   i    j
        '''

        # in the first place
        # build H_{i'ij'j}^{n_{i'}n_{i}n_{j'}n_{j}}
        # pay attention to the natural tensor product order
        dual = (0, 1, 0, 1)
        shape = tuple([(self._dim_phys, self._dim_phys)]*4)
        blocks = {}
        block_shape = tuple([self._dim_phys]*4)

        bare = torch.zeros(block_shape)
        temp = torch.zeros(block_shape)

        # hopping t-terms
        temp[0, 0, 0, 0] = 1.0 
        temp[1, 0, 0, 1] = 1.0
        qns = [1, 0, 0, 1]
        if self._schwinger_boson:
            qns = [q ^ 1 for q in qns]
        blocks[tuple(qns)] = blocks.get(tuple(qns), bare)+(-1.0*self._t*temp)
        # complex conjugate
        temp.zero_()
        temp[0, 0, 0, 0] = -1.0
        temp[0, 1, 1, 0] = -1.0
        qns = [0, 1, 1, 0]
        if self._schwinger_boson:
            qns = [q ^ 1 for q in qns]
        blocks[tuple(qns)] = blocks.get(tuple(qns), bare)+(-1.0*self._t*temp)

        # chemical potential
        # there is a minus sign by default
        temp.zero_()
        temp[0, 0, 0, 0] = 0.25
        temp[1, 1, 0, 0] = 0.25
        blocks[(1, 1, 0, 0)] = blocks.get((1, 1, 0, 0), bare)+(-1.0*self._mu*temp)
        temp.zero_()
        temp[0, 0, 0, 0] = 0.25
        temp[1, 1, 0, 0] = 0.25
        temp[0, 0, 1, 1] = 0.25
        temp[1, 1, 1, 1] = 0.25
        blocks[(1, 1, 1, 1)] = blocks.get((1, 1, 1, 1), bare)+(-1.0*self._mu*temp)

        temp.zero_()
        temp[0, 0, 0, 0] = 0.25
        temp[0, 0, 1, 1] = 0.25
        blocks[(0, 0, 1, 1)] = blocks.get((0, 0, 1, 1), bare)+(-1.0*self._mu*temp)
        temp.zero_()
        temp[0, 0, 0, 0] = 0.25
        temp[1, 1, 0, 0] = 0.25
        temp[0, 0, 1, 1] = 0.25
        temp[1, 1, 1, 1] = 0.25
        blocks[(1, 1, 1, 1)] = blocks.get((1, 1, 1, 1), bare)+(-1.0*self._mu*temp)

        # then permute to H_{i'j'ij}^{n_{i'}n_{j'}n_{i}n_{j}}
        return GTensor(dual, shape, blocks).permute((0, 2, 1, 3))

    def twobody_cc(self) -> GTensor:

        # in the first place
        # build H_{i'ij'j}^{n_{i'}n_{i}n_{j'}n_{j}}
        # pay attention to the natural tensor product order
        dual = (0, 1, 0, 1)
        shape = tuple([(self._dim_phys, self._dim_phys)]*4)
        blocks = {}
        block_shape = tuple([self._dim_phys]*4)

        bare = torch.zeros(block_shape)
        temp = torch.zeros(block_shape)

        # hopping t-terms
        temp.zero_()
        temp[0, 0, 0, 0] = 1.0
        # temp[1, 0, 0, 1] = 1.0
        blocks[(1, 0, 0, 1)] = blocks.get((1, 0, 0, 1), bare)+temp
        # complex conjugate
        temp.zero_()
        # temp[0, 0, 0, 0] = -1.0
        # temp[0, 1, 1, 0] = -1.0
        blocks[(0, 1, 1, 0)] = blocks.get((0, 1, 1, 0), bare)+temp

        # then permute to H_{i'j'ij}^{n_{i'}n_{j'}n_{i}n_{j}}
        return GTensor(dual, shape, blocks).permute((0, 2, 1, 3))

    def twobody_hopping(self) -> GTensor:

        dual = (0, 1, 0, 1)
        shape = tuple([(self._dim_phys, self._dim_phys)]*4)
        blocks = {}
        block_shape = tuple([self._dim_phys]*4)

        bare = torch.zeros(block_shape)
        temp = torch.zeros(block_shape)

        # hopping t-terms
        temp[0, 0, 0, 0] = 1.0 
        temp[1, 0, 0, 1] = 1.0
        qns = [1, 0, 0, 1]
        if self._schwinger_boson:
            qns = [q ^ 1 for q in qns]
        blocks[tuple(qns)] = blocks.get(tuple(qns), bare)+(-1.0*self._t*temp)
        # complex conjugate
        temp.zero_()
        temp[0, 0, 0, 0] = -1.0
        temp[0, 1, 1, 0] = -1.0
        qns = [0, 1, 1, 0]
        if self._schwinger_boson:
            qns = [q ^ 1 for q in qns]
        blocks[tuple(qns)] = blocks.get(tuple(qns), bare)+(-1.0*self._t*temp)

        return GTensor(dual, shape, blocks).permute((0, 2, 1, 3))

    '''
    def twobody_heisenberg(self) -> GTensor:
        # in the first place
        # build H_{i'ij'j}^{n_{i'}n_{i}n_{j'}n_{j}}
        # pay attention to the natural tensor product order
        dual = (0, 1, 0, 1)
        shape = tuple([(self._dim_phys, self._dim_phys)]*4)
        blocks = {}
        block_shape = tuple([self._dim_phys]*4)

        bare = torch.zeros(block_shape)
        temp = torch.zeros(block_shape)

        # Heisenberg J-terms
        temp.zero_()
        # off-diagonal
        temp[1, 0, 0, 1] = 0.5
        temp[0, 1, 1, 0] = 0.5
        # diagonal
        temp[1, 1, 0, 0] = -0.5
        temp[0, 0, 1, 1] = -0.5
        blocks[(1, 1, 1, 1)] = blocks.get((1, 1, 1, 1), bare)+(self._J*temp)
        # then permute to H_{i'j'ij}^{n_{i'}n_{j'}n_{i}n_{j}}
        # return tp.gpermute(GTensor(dual, blocks), (0, 2, 1, 3))

        return GTensor(dual, shape, blocks).permute((0, 2, 1, 3))
    '''

    def twobody_ham(self, cflag=False) -> GTensor:
        r'''
        build the Hamiltonian as a GTensor by listing all possible quantum channels

        # H_{i'j'ij}^{n_{i'}n_{j'}n_{i}n_{j}}
        #   i'   j'
        #   |0   |0
        #   ^    ^
        #   |    |
        #   *----*
        #   |    |
        #   ^    ^
        #   |1   |1
        #   i    j
        '''

        # in the first place
        # build H_{i'ij'j}^{n_{i'}n_{i}n_{j'}n_{j}}
        # pay attention to the natural tensor product order
        dual = (0, 1, 0, 1)
        shape = tuple([(self._dim_phys, self._dim_phys)]*4)

        blocks = {}
        block_shape = tuple([self._dim_phys]*4)
        bare = torch.zeros(block_shape)
        temp = torch.zeros(block_shape)
        # hopping t-terms
        temp[0, 0, 0, 0] = 1.0 
        temp[1, 0, 0, 1] = 1.0
        qns = [1, 0, 0, 1]
        if self._schwinger_boson:
            qns = [q ^ 1 for q in qns]
        blocks[tuple(qns)] = blocks.get(tuple(qns), bare)+(-1.0*self._t*temp)
        # complex conjugate
        temp.zero_()
        temp[0, 0, 0, 0] = -1.0
        temp[0, 1, 1, 0] = -1.0
        qns = [0, 1, 1, 0]
        if self._schwinger_boson:
            qns = [q ^ 1 for q in qns]
        blocks[tuple(qns)] = blocks.get(tuple(qns), bare)+(-1.0*self._t*temp)

        # Heisenberg J-terms
        temp.zero_()
        # off-diagonal
        temp[1, 0, 0, 1] = 0.5
        temp[0, 1, 1, 0] = 0.5
        # diagonal
        temp[1, 1, 0, 0] = -0.5
        temp[0, 0, 1, 1] = -0.5
        qns = [1, 1, 1, 1]
        if self._schwinger_boson:
            qns = [q ^ 1 for q in qns]
        blocks[tuple(qns)] = blocks.get(tuple(qns), bare)+(self._J*temp)

        # chemical potential
        # there is a minus sign by default
        # double occupied state should be projected out: | e_{k}^{n} >, k=1, n=0
        id_states = (0, 0), (0, 1), (1, 1)
        # n_{i} \otimes 1
        for k, n in id_states:
            temp.zero_()
            temp[0, 0, k, k] = 0.25
            temp[1, 1, k, k] = 0.25
            qns = [1, 1, n, n]
            if self._schwinger_boson:
                qns = [q ^ 1 for q in qns]
            blocks[tuple(qns)] = blocks.get(tuple(qns), bare)+(-1.0*self._mu*temp)
        # 1 \otimes n_{j}
        for k, n in id_states:
            temp.zero_()
            temp[k, k, 0, 0] = 0.25
            temp[k, k, 1, 1] = 0.25
            qns = [n, n, 1, 1]
            if self._schwinger_boson:
                qns = [q ^ 1 for q in qns]
            blocks[tuple(qns)] = blocks.get(tuple(qns), bare)+(-1.0*self._mu*temp)

        return GTensor(dual, shape, blocks, cflag).permute((0, 2, 1, 3))

    def twobody_time_evo(self, op: GTensor, delta: float, order: int) -> GTensor:
        r'''
        time evolution by Talyor expansion

        Parameters
        ----------
        delta: float, time evolution step size
        order: int, how many Taylor orders you want
        '''

        # build the identity operator
        dual = (0, 1, 0, 1)
        shape = tuple([(self._dim_phys, self._dim_phys)]*4)

        block_shape = tuple([self._dim_phys]*4)
        temp = torch.zeros(block_shape)
        bare = torch.zeros(block_shape)
        blocks = {}
        id_states = (0, 0), (0, 1), (1, 1)
        # k, m: the first
        # l, n: the second
        for k, m in id_states:
            for l, n in id_states:
                temp.zero_()
                temp[k, k, l, l] = 1.0
                qns = [m, m, n, n]
                if self._schwinger_boson:
                    qns = [q ^ 1 for q in qns]
                blocks[tuple(qns)] = blocks.get(tuple(qns), bare)+temp

        id_gt = GTensor(dual, shape, blocks).permute((0, 2, 1, 3))
        # powers of operators
        op_powers = [id_gt, op]
        for i in range(2, order):
            op_powers.append(tp.gcontract('abcd,cdef->abef', op, op_powers[i-1]))
        # taylor expansion
        time_evo = op_powers[0]
        for i in range(1, order):
            time_evo = time_evo+(1.0/math.factorial(i))*((-delta)**i)*op_powers[i]

        return time_evo

class SquareTJ1J2(object):
    r'''
    t-J model on a square lattice
    '''

    def __init__(self, t: float, J1: float, J2: float, mu: float):

        self._dim_phys = 2
        self._t, self._J1, self._J2, self._mu = t, J1, J2, mu

    def onsite_n(self) -> GTensor:
        r'''
        onsite particle-number operator
        '''

        dual = (0, 1)
        shape = tuple([(self._dim_phys, self._dim_phys)]*2)
        blocks = {}
        block_shape = tuple([self._dim_phys]*2)
        temp = torch.zeros(block_shape)
        temp[0, 0] = 1.0
        temp[1, 1] = 1.0
        blocks[(1, 1)] = temp

        return GTensor(dual, shape, blocks)

    def onsite_sz(self) -> GTensor:
        r'''
        onsite Sz operator
        '''

        dual = (0, 1)
        shape = tuple([(self._dim_phys, self._dim_phys)]*2)
        blocks = {}
        block_shape = tuple([self._dim_phys]*2)
        temp = torch.zeros(block_shape)
        temp[0, 0] = 0.5
        temp[1, 1] = -0.5
        blocks[(1, 1)] = temp

        return GTensor(dual, shape, blocks)

    def onsite_sy(self) -> GTensor:
        r'''
        onsite Sx operator
        '''

        dual = (0, 1)
        shape = tuple([(self._dim_phys, self._dim_phys)]*2)
        blocks = {}
        block_shape = tuple([self._dim_phys]*2)
        temp = torch.zeros(block_shape)
        temp[1, 0] = 0.5
        temp[0, 1] = -0.5

        qns = [1, 1]
        blocks[tuple(qns)] = temp

        return GTensor(dual, shape, blocks)

    def onsite_sx(self) -> GTensor:
        r'''
        onsite Sx operator
        '''

        dual = (0, 1)
        shape = tuple([(self._dim_phys, self._dim_phys)]*2)
        blocks = {}
        block_shape = tuple([self._dim_phys]*2)
        temp = torch.zeros(block_shape)
        temp[1, 0] = 0.5
        temp[0, 1] = 0.5
        blocks[(1, 1)] = temp

        return GTensor(dual, shape, blocks)

    def c_up(self) -> Z2gTensor:

        dual = (0, 1)
        shape = tuple([(self._dim_phys, self._dim_phys)]*2)
        blocks = {}
        temp = torch.zeros(tuple([self._dim_phys]*2))
        temp[0, 0] = 1.0
        blocks[(0, 1)] = temp

        return Z2gTensor(dual, shape, blocks)

    def c_up_dagger(self) -> Z2gTensor:

        dual = (0, 1)
        shape = tuple([(self._dim_phys, self._dim_phys)]*2)
        blocks = {}
        temp = torch.zeros(tuple([self._dim_phys]*2))
        temp[0, 0] = 1.0
        blocks[(1, 0)] = temp

        return Z2gTensor(dual, shape, blocks)

    def c_down(self) -> Z2gTensor:

        dual = (0, 1)
        shape = tuple([(self._dim_phys, self._dim_phys)]*2)
        blocks = {}
        temp = torch.zeros(tuple([self._dim_phys]*2))
        temp[0, 1] = 1.0
        blocks[(0, 1)] = temp

        return Z2gTensor(dual, shape, blocks)

    def c_down_dagger(self) -> Z2gTensor:

        dual = (0, 1)
        shape = tuple([(self._dim_phys, self._dim_phys)]*2)
        blocks = {}
        temp = torch.zeros(tuple([self._dim_phys]*2))
        temp[1, 0] = 1.0
        blocks[(1, 0)] = temp

        return Z2gTensor(dual, shape, blocks)

    def c_down_dagger_c_up(self) -> Z2gTensor:

        dual = (0, 1)
        shape = tuple([(self._dim_phys, self._dim_phys)]*2)
        blocks = {}
        temp = torch.zeros(tuple([self._dim_phys]*2))
        temp[1, 0] = 1.0
        blocks[(1, 1)] = temp

        return Z2gTensor(dual, shape, blocks)

    def c_up_dagger_c_down(self) -> Z2gTensor:

        dual = (0, 1)
        shape = tuple([(self._dim_phys, self._dim_phys)]*2)
        blocks = {}
        temp = torch.zeros(tuple([self._dim_phys]*2))
        temp[0, 1] = 1.0
        blocks[(1, 1)] = temp

        return Z2gTensor(dual, shape, blocks)

    def n_up(self) -> Z2gTensor:

        dual = (0, 1)
        shape = tuple([(self._dim_phys, self._dim_phys)]*2)
        blocks = {}
        temp = torch.zeros(tuple([self._dim_phys]*2))
        temp[0, 0] = 1.0
        blocks[(1, 1)] = temp

        return Z2gTensor(dual, shape, blocks)

    def n_down(self) -> Z2gTensor:

        dual = (0, 1)
        shape = tuple([(self._dim_phys, self._dim_phys)]*2)
        blocks = {}
        temp = torch.zeros(tuple([self._dim_phys]*2))
        temp[1, 1] = 1.0
        blocks[(1, 1)] = temp

        return Z2gTensor(dual, shape, blocks)

    def onebody_id(self) -> Z2gTensor:
        r'''
        time evolution by Talyor expansion

        Parameters
        ----------
        delta: float, time evolution step size
        order: int, how many Taylor orders you want
        '''

        # build the identity operator
        dual = (0, 1)
        shape = tuple([(self._dim_phys, self._dim_phys)]*2)
        block_shape = tuple([self._dim_phys]*2)

        temp = torch.zeros(block_shape)
        bare = torch.zeros(block_shape)
        blocks = {}
        id_states = (0, 0), (0, 1), (1, 1)
        for l, n in id_states:
            temp.zero_()
            temp[l, l] = 1.0
            blocks[(n, n)] = blocks.get((n, n), bare)+temp

        return Z2gTensor(dual, shape, blocks)

    def threebody_ham(self) -> GTensor:
        r'''
        build the Hamiltonian as a GTensor by listing all possible quantum channels

        # H_{i'j'k'ijk}^{n_{i'}n_{j'}n_{k'}n_{i}n_{j}n_{k}}
        #   i'   j'   k'
        #   |0   |0   |0
        #   ^    ^    ^
        #   |    |    |
        #   *----*----*
        #   |    |    |
        #   ^    ^    ^
        #   |1   |1   |1
        #   i    j    k
        '''

        # in the first place
        # build H_{i'ij'jk'k}^{n_{i'}n_{i}n_{j'}n_{j}n_{k'}n_{k}}
        # pay attention to the natural tensor product order
        dual = (0, 1, 0, 1, 0, 1)
        shape = tuple([(self._dim_phys, self._dim_phys)]*6)
        blocks = {}
        block_shape = tuple([self._dim_phys]*6)

        bare = torch.zeros(block_shape)
        temp = torch.zeros(block_shape)

        # (phy index, fermion index)
        id_states = (0, 0), (0, 1), (1, 1)

        # hopping t-terms
        over_counting = 0.125
        # graded tensor product idenity of k
        # c_{i}^{dagger} c_{j} 1_{k}
        for l, n in id_states:
            temp.zero_()
            # uparrow
            temp[0, 0, 0, 0, l, l] = 1.0 
            # downarrow
            temp[1, 0, 0, 1, l, l] = 1.0 
            blocks[(1, 0, 0, 1, n, n)] = blocks.get((1, 0, 0, 1, n, n), bare)+(-1.0*over_counting*self._t*temp)
        # c_{i} c_{j}^{dagger} 1_{k}
        for l, n in id_states:
            temp.zero_()
            temp[0, 0, 0, 0, l, l] = -1.0 
            temp[0, 1, 1, 0, l, l] = -1.0
            blocks[(0, 1, 1, 0, n, n)] = blocks.get((0, 1, 1, 0, n, n), bare)+(-1.0*over_counting*self._t*temp)

        # 1_{i} c_{j}^{dagger}c_{k}
        for l, n in id_states:
            temp.zero_()
            temp[l, l, 0, 0, 0, 0] = 1.0 
            temp[l, l, 1, 0, 0, 1] = 1.0 
            blocks[(n, n, 1, 0, 0, 1)] = blocks.get((n, n, 1, 0, 0, 1), bare)+(-1.0*over_counting*self._t*temp)
        # 1_{i} c_{j}^{dagger}c_{k}
        for l, n in id_states:
            temp.zero_()
            temp[l, l, 0, 0, 0, 0] = -1.0 
            temp[l, l, 0, 1, 1, 0] = -1.0
            blocks[(n, n, 0, 1, 1, 0)] = blocks.get((n, n, 0, 1, 1, 0), bare)+(-1.0*over_counting*self._t*temp)

        # Heisenberg J1-terms
        # i, j
        for l, n in id_states:
            temp.zero_()
            temp[1, 0, 0, 1, l, l] = 0.5 
            temp[0, 1, 1, 0, l, l] = 0.5 
            temp[1, 1, 0, 0, l, l] = -0.5
            temp[0, 0, 1, 1, l, l] = -0.5
            blocks[(1, 1, 1, 1, n, n)] = blocks.get((1, 1, 1, 1, n, n), bare)+(over_counting*self._J1*temp)
        # j, k
        for l, n in id_states:
            temp.zero_()
            temp[l, l, 1, 0, 0, 1] = 0.5 
            temp[l, l, 0, 1, 1, 0] = 0.5 
            temp[l, l, 1, 1, 0, 0] = -0.5
            temp[l, l, 0, 0, 1, 1] = -0.5
            blocks[(n, n, 1, 1, 1, 1)] = blocks.get((n, n, 1, 1, 1, 1), bare)+(over_counting*self._J1*temp)

        # Heisenberg J2-terms
        # i, k
        over_counting = 0.25
        for l, n in id_states:
            temp.zero_()
            temp[1, 0, l, l, 0, 1] = 0.5 
            temp[0, 1, l, l, 1, 0] = 0.5 
            temp[1, 1, l, l, 0, 0] = -0.5
            temp[0, 0, l, l, 1, 1] = -0.5
            blocks[(1, 1, n, n, 1, 1)] = blocks.get((1, 1, n, n, 1, 1), bare)+(over_counting*self._J2*temp)

        # chemical potential
        # there is a minus sign by default
        over_counting = 1.0/24.0
        # n_{i}
        for r, m in id_states:
            for s, n in id_states:
                temp.zero_()
                temp[0, 0, r, r, s, s] = 1.0
                temp[1, 1, r, r, s, s] = 1.0
                blocks[(1, 1, m, m, n, n)] = blocks.get((1, 1, m, m, n, n), bare)+(-1.0*over_counting*self._mu*temp)
        # n_{j}
        for r, m in id_states:
            for s, n in id_states:
                temp.zero_()
                temp[r, r, 0, 0, s, s] = 1.0
                temp[r, r, 1, 1, s, s] = 1.0
                blocks[(m, m, 1, 1, n, n)] = blocks.get((m, m, 1, 1, n, n), bare)+(-1.0*over_counting*self._mu*temp)
        # n_{k}
        for r, m in id_states:
            for s, n in id_states:
                temp.zero_()
                temp[r, r, s, s, 0, 0] = 1.0
                temp[r, r, s, s, 1, 1] = 1.0
                blocks[(m, m, n, n, 1, 1)] = blocks.get((m, m, n, n, 1, 1), bare)+(-1.0*over_counting*self._mu*temp)

        return GTensor(dual, shape, blocks).permute((0, 2, 4, 1, 3, 5))

    def threebody_J2ham(self) -> GTensor:
        r'''
        build the Hamiltonian as a GTensor by listing all possible quantum channels

        # H_{i'j'k'ijk}^{n_{i'}n_{j'}n_{k'}n_{i}n_{j}n_{k}}
        #   i'   j'   k'
        #   |0   |0   |0
        #   ^    ^    ^
        #   |    |    |
        #   *----*----*
        #   |    |    |
        #   ^    ^    ^
        #   |1   |1   |1
        #   i    j    k
        '''

        # in the first place
        # build H_{i'ij'jk'k}^{n_{i'}n_{i}n_{j'}n_{j}n_{k'}n_{k}}
        # pay attention to the natural tensor product order
        dual = (0, 1, 0, 1, 0, 1)
        shape = tuple([(self._dim_phys, self._dim_phys)]*6)
        blocks = {}
        block_shape = tuple([self._dim_phys]*6)

        bare = torch.zeros(block_shape)
        temp = torch.zeros(block_shape)

        # (phy index, fermion index)
        id_states = (0, 0), (0, 1), (1, 1)

        # Heisenberg J2-terms
        for l, n in id_states:
            temp.zero_()
            temp[1, 0, l, l, 0, 1] = 0.5 
            temp[0, 1, l, l, 1, 0] = 0.5 
            temp[1, 1, l, l, 0, 0] = -0.5
            temp[0, 0, l, l, 1, 1] = -0.5
            blocks[(1, 1, n, n, 1, 1)] = blocks.get((1, 1, n, n, 1, 1), bare)+(self._J2*temp)

        return GTensor(dual, shape, blocks).permute((0, 2, 4, 1, 3, 5))

    def threebody_time_evo(self, op: GTensor, delta: float, order: int) -> GTensor:
        r'''
        time evolution by Talyor expansion

        Parameters
        ----------
        op: GTensor, the operator
        delta: float, time evolution step size
        order: int, how many Taylor orders you want
        '''

        id_states = (0, 0), (0, 1), (1, 1)

        # build the identity operator
        dual = (0, 1, 0, 1, 0, 1)
        shape = tuple([(self._dim_phys, self._dim_phys)]*6)
        block_shape = tuple([self._dim_phys]*6)

        temp = torch.zeros(block_shape)
        bare = torch.zeros(block_shape)
        blocks = {}
        for i, m in id_states:
            for j, n in id_states:
                for k, o in id_states:
                    temp.zero_()
                    temp[i, i, j, j, k, k] = 1.0
                    blocks[(m, m, n, n, o, o)] = blocks.get((m, m, n, n, o, o), bare)+temp

        id_gt = GTensor(dual, shape, blocks).permute((0, 2, 4, 1, 3, 5))

        # powers of the operator
        powers = [id_gt, op]
        for i in range(2, order):
            powers.append(tp.gcontract('abcdef,defghi->abcghi', op, powers[i-1]))
        # taylor expansion
        time_evo = powers[0]
        for i in range(1, order):
            time_evo = time_evo+(1.0/math.factorial(i))*((-delta)**i)*powers[i]

        return time_evo

    def twobody_tJham(self, cflag=False) -> GTensor:
        r'''
        build the Hamiltonian as a GTensor by listing all possible quantum channels
        # H_{i'j'ij}^{n_{i'}n_{j'}n_{i}n_{j}}
        #   i'   j'
        #   |0   |0
        #   ^    ^
        #   |    |
        #   *----*
        #   |    |
        #   ^    ^
        #   |1   |1
        #   i    j
        '''

        # in the first place
        # build H_{i'ij'j}^{n_{i'}n_{i}n_{j'}n_{j}}
        # pay attention to the natural tensor product order
        dual = (0, 1, 0, 1)
        shape = tuple([(self._dim_phys, self._dim_phys)]*4)

        blocks = {}
        block_shape = tuple([self._dim_phys]*4)
        bare = torch.zeros(block_shape)
        temp = torch.zeros(block_shape)
        # hopping t-terms
        temp[0, 0, 0, 0] = 1.0 
        temp[1, 0, 0, 1] = 1.0
        qns = [1, 0, 0, 1]
        blocks[tuple(qns)] = blocks.get(tuple(qns), bare)+(-1.0*self._t*temp)
        # complex conjugate
        temp.zero_()
        temp[0, 0, 0, 0] = -1.0
        temp[0, 1, 1, 0] = -1.0
        qns = [0, 1, 1, 0]
        blocks[tuple(qns)] = blocks.get(tuple(qns), bare)+(-1.0*self._t*temp)

        # Heisenberg J-terms
        temp.zero_()
        # off-diagonal
        temp[1, 0, 0, 1] = 0.5
        temp[0, 1, 1, 0] = 0.5
        # diagonal
        temp[1, 1, 0, 0] = -0.5
        temp[0, 0, 1, 1] = -0.5
        qns = [1, 1, 1, 1]
        blocks[tuple(qns)] = blocks.get(tuple(qns), bare)+(self._J1*temp)

        # chemical potential
        # there is a minus sign by default
        # double occupied state should be projected out: | e_{k}^{n} >, k=1, n=0
        id_states = (0, 0), (0, 1), (1, 1)
        # n_{i} \otimes 1
        for k, n in id_states:
            temp.zero_()
            temp[0, 0, k, k] = 0.25
            temp[1, 1, k, k] = 0.25
            qns = [1, 1, n, n]
            blocks[tuple(qns)] = blocks.get(tuple(qns), bare)+(-1.0*self._mu*temp)
        # 1 \otimes n_{j}
        for k, n in id_states:
            temp.zero_()
            temp[k, k, 0, 0] = 0.25
            temp[k, k, 1, 1] = 0.25
            qns = [n, n, 1, 1]
            blocks[tuple(qns)] = blocks.get(tuple(qns), bare)+(-1.0*self._mu*temp)

        return GTensor(dual, shape, blocks, cflag).permute((0, 2, 1, 3))

    def twobody_time_evo(self, op: GTensor, delta: float, order: int) -> GTensor:
        r'''
        time evolution by Talyor expansion

        Parameters
        ----------
        delta: float, time evolution step size
        order: int, how many Taylor orders you want
        '''

        # build the identity operator
        dual = (0, 1, 0, 1)
        shape = tuple([(self._dim_phys, self._dim_phys)]*4)

        block_shape = tuple([self._dim_phys]*4)
        temp = torch.zeros(block_shape)
        bare = torch.zeros(block_shape)
        blocks = {}
        id_states = (0, 0), (0, 1), (1, 1)
        # k, m: the first
        # l, n: the second
        for k, m in id_states:
            for l, n in id_states:
                temp.zero_()
                temp[k, k, l, l] = 1.0
                qns = [m, m, n, n]
                blocks[tuple(qns)] = blocks.get(tuple(qns), bare)+temp

        id_gt = GTensor(dual, shape, blocks).permute((0, 2, 1, 3))
        # powers of operators
        op_powers = [id_gt, op]
        for i in range(2, order):
            op_powers.append(tp.gcontract('abcd,cdef->abef', op, op_powers[i-1]))
        # taylor expansion
        time_evo = op_powers[0]
        for i in range(1, order):
            time_evo = time_evo+(1.0/math.factorial(i))*((-delta)**i)*op_powers[i]

        return time_evo


class SquareSigmaTJ(object):
    r'''
    sigma-t-J model on a square lattice
    '''

    def __init__(self, t: float, J: float, mu: float):
        r'''
        '''

        self._dim_phys = 2
        self._t, self._J, self._mu = t, J, mu

    def onsite_n(self) -> GTensor:
        r'''
        onsite particle-number operator
        '''

        dual = (0, 1)
        shape = tuple([(self._dim_phys, self._dim_phys)]*2)
        blocks = {}
        block_shape = tuple([self._dim_phys]*2)
        temp = torch.zeros(block_shape)
        temp[0, 0] = 1.0
        temp[1, 1] = 1.0

        qns = [1, 1]
        blocks[tuple(qns)] = temp

        return GTensor(dual, shape, blocks)

    def onsite_sz(self) -> GTensor:
        r'''
        onsite Sz operator
        '''

        dual = (0, 1)
        shape = tuple([(self._dim_phys, self._dim_phys)]*2)
        blocks = {}
        block_shape = tuple([self._dim_phys]*2)
        temp = torch.zeros(block_shape)
        temp[0, 0] = 0.5
        temp[1, 1] = -0.5
        
        qns = [1, 1]
        blocks[tuple(qns)] = temp

        return GTensor(dual, shape, blocks)

    def onsite_sx(self) -> GTensor:
        r'''
        onsite Sx operator
        '''

        dual = (0, 1)
        shape = tuple([(self._dim_phys, self._dim_phys)]*2)
        blocks = {}
        block_shape = tuple([self._dim_phys]*2)
        temp = torch.zeros(block_shape)
        temp[1, 0] = 0.5
        temp[0, 1] = 0.5

        qns = [1, 1]
        blocks[tuple(qns)] = temp

        return GTensor(dual, shape, blocks)

    def onsite_sy(self) -> GTensor:
        r'''
        onsite Sx operator
        '''

        dual = (0, 1)
        shape = tuple([(self._dim_phys, self._dim_phys)]*2)
        blocks = {}
        block_shape = tuple([self._dim_phys]*2)
        temp = torch.zeros(block_shape)
        temp[1, 0] = 0.5
        temp[0, 1] = -0.5

        qns = [1, 1]
        blocks[tuple(qns)] = temp

        return GTensor(dual, shape, blocks)

    def c_up(self) -> Z2gTensor:

        dual = (0, 1)
        shape = tuple([(self._dim_phys, self._dim_phys)]*2)
        blocks = {}
        temp = torch.zeros(tuple([self._dim_phys]*2))
        temp[0, 0] = 1.0

        qns = [0, 1]
        blocks[tuple(qns)] = temp

        return Z2gTensor(dual, shape, blocks)

    def c_up_dagger(self) -> Z2gTensor:

        dual = (0, 1)
        shape = tuple([(self._dim_phys, self._dim_phys)]*2)
        blocks = {}
        temp = torch.zeros(tuple([self._dim_phys]*2))
        temp[0, 0] = 1.0

        qns = [1, 0]
        blocks[tuple(qns)] = temp

        return Z2gTensor(dual, shape, blocks)

    def c_down(self) -> Z2gTensor:

        dual = (0, 1)
        shape = tuple([(self._dim_phys, self._dim_phys)]*2)
        blocks = {}
        temp = torch.zeros(tuple([self._dim_phys]*2))
        temp[0, 1] = 1.0

        qns = [0, 1]
        blocks[tuple(qns)] = temp

        return Z2gTensor(dual, shape, blocks)

    def c_down_dagger(self) -> Z2gTensor:

        dual = (0, 1)
        shape = tuple([(self._dim_phys, self._dim_phys)]*2)
        blocks = {}
        temp = torch.zeros(tuple([self._dim_phys]*2))
        temp[1, 0] = 1.0

        qns = [1, 0]
        blocks[tuple(qns)] = temp

        return Z2gTensor(dual, shape, blocks)

    def c_down_dagger_c_up(self) -> Z2gTensor:

        dual = (0, 1)
        shape = tuple([(self._dim_phys, self._dim_phys)]*2)
        blocks = {}
        temp = torch.zeros(tuple([self._dim_phys]*2))
        temp[1, 0] = 1.0

        qns = [1, 1]
        blocks[tuple(qns)] = temp

        return Z2gTensor(dual, shape, blocks)

    def c_up_dagger_c_down(self) -> Z2gTensor:

        dual = (0, 1)
        shape = tuple([(self._dim_phys, self._dim_phys)]*2)
        blocks = {}
        temp = torch.zeros(tuple([self._dim_phys]*2))
        temp[0, 1] = 1.0

        qns = [1, 1]
        blocks[tuple(qns)] = temp

        return Z2gTensor(dual, shape, blocks)

    def n_up(self) -> Z2gTensor:

        dual = (0, 1)
        shape = tuple([(self._dim_phys, self._dim_phys)]*2)
        blocks = {}
        temp = torch.zeros(tuple([self._dim_phys]*2))
        temp[0, 0] = 1.0

        qns = [1, 1]
        blocks[tuple(qns)] = temp

        return Z2gTensor(dual, shape, blocks)

    def n_down(self) -> Z2gTensor:

        dual = (0, 1)
        shape = tuple([(self._dim_phys, self._dim_phys)]*2)
        blocks = {}
        temp = torch.zeros(tuple([self._dim_phys]*2))
        temp[1, 1] = 1.0

        qns = [1, 1]
        blocks[tuple(qns)] = temp

        return Z2gTensor(dual, shape, blocks)

    def onebody_id(self) -> Z2gTensor:
        r'''
        identity closure condition
        '''

        # build the identity operator
        dual = (0, 1)
        shape = tuple([(self._dim_phys, self._dim_phys)]*2)
        block_shape = tuple([self._dim_phys]*2)

        temp = torch.zeros(block_shape)
        bare = torch.zeros(block_shape)
        blocks = {}

        id_states = (0, 0), (0, 1), (1, 1)

        for l, n in id_states:
            temp.zero_()
            temp[l, l] = 1.0
            blocks[(n, n)] = blocks.get((n, n), bare)+temp

        return Z2gTensor(dual, shape, blocks)

    def twobody_hopping_mu(self) -> GTensor:
        r'''
        # H_{i'j'ij}^{n_{i'}n_{j'}n_{i}n_{j}}
        #   i'   j'
        #   |0   |0
        #   ^    ^
        #   |    |
        #   *----*
        #   |    |
        #   ^    ^
        #   |1   |1
        #   i    j
        '''

        # in the first place
        # build H_{i'ij'j}^{n_{i'}n_{i}n_{j'}n_{j}}
        # pay attention to the natural tensor product order
        dual = (0, 1, 0, 1)
        shape = tuple([(self._dim_phys, self._dim_phys)]*4)
        blocks = {}
        block_shape = tuple([self._dim_phys]*4)

        bare = torch.zeros(block_shape)
        temp = torch.zeros(block_shape)

        # hopping t-terms
        temp[0, 0, 0, 0] = 1.0 
        temp[1, 0, 0, 1] = 1.0
        qns = [1, 0, 0, 1]
        blocks[tuple(qns)] = blocks.get(tuple(qns), bare)+(-1.0*self._t*temp)
        # complex conjugate
        temp.zero_()
        temp[0, 0, 0, 0] = -1.0
        temp[0, 1, 1, 0] = -1.0
        qns = [0, 1, 1, 0]
        if self._schwinger_boson:
            qns = [q ^ 1 for q in qns]
        blocks[tuple(qns)] = blocks.get(tuple(qns), bare)+(-1.0*self._t*temp)

        # chemical potential
        # there is a minus sign by default
        temp.zero_()
        temp[0, 0, 0, 0] = 0.25
        temp[1, 1, 0, 0] = 0.25
        blocks[(1, 1, 0, 0)] = blocks.get((1, 1, 0, 0), bare)+(-1.0*self._mu*temp)
        temp.zero_()
        temp[0, 0, 0, 0] = 0.25
        temp[1, 1, 0, 0] = 0.25
        temp[0, 0, 1, 1] = 0.25
        temp[1, 1, 1, 1] = 0.25
        blocks[(1, 1, 1, 1)] = blocks.get((1, 1, 1, 1), bare)+(-1.0*self._mu*temp)

        temp.zero_()
        temp[0, 0, 0, 0] = 0.25
        temp[0, 0, 1, 1] = 0.25
        blocks[(0, 0, 1, 1)] = blocks.get((0, 0, 1, 1), bare)+(-1.0*self._mu*temp)
        temp.zero_()
        temp[0, 0, 0, 0] = 0.25
        temp[1, 1, 0, 0] = 0.25
        temp[0, 0, 1, 1] = 0.25
        temp[1, 1, 1, 1] = 0.25
        blocks[(1, 1, 1, 1)] = blocks.get((1, 1, 1, 1), bare)+(-1.0*self._mu*temp)

        # then permute to H_{i'j'ij}^{n_{i'}n_{j'}n_{i}n_{j}}
        return GTensor(dual, shape, blocks).permute((0, 2, 1, 3))

    def twobody_cc(self) -> GTensor:

        # in the first place
        # build H_{i'ij'j}^{n_{i'}n_{i}n_{j'}n_{j}}
        # pay attention to the natural tensor product order
        dual = (0, 1, 0, 1)
        shape = tuple([(self._dim_phys, self._dim_phys)]*4)
        blocks = {}
        block_shape = tuple([self._dim_phys]*4)

        bare = torch.zeros(block_shape)
        temp = torch.zeros(block_shape)

        # hopping t-terms
        temp.zero_()
        temp[0, 0, 0, 0] = 1.0
        # temp[1, 0, 0, 1] = 1.0
        blocks[(1, 0, 0, 1)] = blocks.get((1, 0, 0, 1), bare)+temp
        # complex conjugate
        temp.zero_()
        # temp[0, 0, 0, 0] = -1.0
        # temp[0, 1, 1, 0] = -1.0
        blocks[(0, 1, 1, 0)] = blocks.get((0, 1, 1, 0), bare)+temp

        # then permute to H_{i'j'ij}^{n_{i'}n_{j'}n_{i}n_{j}}
        return GTensor(dual, shape, blocks).permute((0, 2, 1, 3))

    def twobody_hopping(self) -> GTensor:

        dual = (0, 1, 0, 1)
        shape = tuple([(self._dim_phys, self._dim_phys)]*4)
        blocks = {}
        block_shape = tuple([self._dim_phys]*4)

        bare = torch.zeros(block_shape)
        temp = torch.zeros(block_shape)

        # hopping t-terms
        temp[0, 0, 0, 0] = 1.0 
        temp[1, 0, 0, 1] = 1.0
        qns = [1, 0, 0, 1]
        blocks[tuple(qns)] = blocks.get(tuple(qns), bare)+(-1.0*self._t*temp)
        # complex conjugate
        temp.zero_()
        temp[0, 0, 0, 0] = -1.0
        temp[0, 1, 1, 0] = -1.0
        qns = [0, 1, 1, 0]
        if self._schwinger_boson:
            qns = [q ^ 1 for q in qns]
        blocks[tuple(qns)] = blocks.get(tuple(qns), bare)+(-1.0*self._t*temp)

        return GTensor(dual, shape, blocks).permute((0, 2, 1, 3))


    def twobody_ham(self, cflag=False) -> GTensor:
        r'''
        build the Hamiltonian as a GTensor by listing all possible quantum channels

        # H_{i'j'ij}^{n_{i'}n_{j'}n_{i}n_{j}}
        #   i'   j'
        #   |0   |0
        #   ^    ^
        #   |    |
        #   *----*
        #   |    |
        #   ^    ^
        #   |1   |1
        #   i    j
        '''

        # in the first place
        # build H_{i'ij'j}^{n_{i'}n_{i}n_{j'}n_{j}}
        # pay attention to the natural tensor product order
        dual = (0, 1, 0, 1)
        shape = tuple([(self._dim_phys, self._dim_phys)]*4)

        blocks = {}
        block_shape = tuple([self._dim_phys]*4)
        bare = torch.zeros(block_shape)
        temp = torch.zeros(block_shape)
        # hopping t-terms
        # sigma tJ
        temp[0, 0, 0, 0] = 1.0 
        temp[1, 0, 0, 1] = -1.0
        qns = [1, 0, 0, 1]
        blocks[tuple(qns)] = blocks.get(tuple(qns), bare)+(-1.0*self._t*temp)
        # complex conjugate
        temp.zero_()
        temp[0, 0, 0, 0] = -1.0
        temp[0, 1, 1, 0] = 1.0
        qns = [0, 1, 1, 0]
        blocks[tuple(qns)] = blocks.get(tuple(qns), bare)+(-1.0*self._t*temp)

        # Heisenberg J-terms
        temp.zero_()
        # off-diagonal
        temp[1, 0, 0, 1] = 0.5
        temp[0, 1, 1, 0] = 0.5
        # diagonal
        temp[1, 1, 0, 0] = -0.5
        temp[0, 0, 1, 1] = -0.5
        qns = [1, 1, 1, 1]
        blocks[tuple(qns)] = blocks.get(tuple(qns), bare)+(self._J*temp)

        # chemical potential
        # there is a minus sign by default
        # double occupied state should be projected out: | e_{k}^{n} >, k=1, n=0
        id_states = (0, 0), (0, 1), (1, 1)
        # n_{i} \otimes 1
        for k, n in id_states:
            temp.zero_()
            temp[0, 0, k, k] = 0.25
            temp[1, 1, k, k] = 0.25
            qns = [1, 1, n, n]
            blocks[tuple(qns)] = blocks.get(tuple(qns), bare)+(-1.0*self._mu*temp)
        # 1 \otimes n_{j}
        for k, n in id_states:
            temp.zero_()
            temp[k, k, 0, 0] = 0.25
            temp[k, k, 1, 1] = 0.25
            qns = [n, n, 1, 1]
            blocks[tuple(qns)] = blocks.get(tuple(qns), bare)+(-1.0*self._mu*temp)

        return GTensor(dual, shape, blocks, cflag).permute((0, 2, 1, 3))

    def twobody_time_evo(self, op: GTensor, delta: float, order: int) -> GTensor:
        r'''
        time evolution by Talyor expansion

        Parameters
        ----------
        delta: float, time evolution step size
        order: int, how many Taylor orders you want
        '''

        # build the identity operator
        dual = (0, 1, 0, 1)
        shape = tuple([(self._dim_phys, self._dim_phys)]*4)

        block_shape = tuple([self._dim_phys]*4)
        temp = torch.zeros(block_shape)
        bare = torch.zeros(block_shape)
        blocks = {}
        id_states = (0, 0), (0, 1), (1, 1)
        # k, m: the first
        # l, n: the second
        for k, m in id_states:
            for l, n in id_states:
                temp.zero_()
                temp[k, k, l, l] = 1.0
                qns = [m, m, n, n]
                blocks[tuple(qns)] = blocks.get(tuple(qns), bare)+temp

        id_gt = GTensor(dual, shape, blocks).permute((0, 2, 1, 3))
        # powers of operators
        op_powers = [id_gt, op]
        for i in range(2, order):
            op_powers.append(tp.gcontract('abcd,cdef->abef', op, op_powers[i-1]))
        # taylor expansion
        time_evo = op_powers[0]
        for i in range(1, order):
            time_evo = time_evo+(1.0/math.factorial(i))*((-delta)**i)*op_powers[i]

        return time_evo
