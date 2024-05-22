import itertools
import math
import opt_einsum as oe

import torch
torch.set_default_dtype(torch.float64)

class Z2gTensor(object):
    r'''
    Z2-graded tensor
    bond permutations should obey the Z2-graded tensor product structure
    '''

    @staticmethod
    def flatten_tuples(ts: tuple) -> tuple:
        r'''
        flatten tuples to one tuple

        Parameters
        ----------
        ts: tuple[tuple] or list[tuple]
        '''

        ft = ()
        for e in ts:
            ft += e

        return ft

    def __init__(self, dual: tuple, shape: tuple, blocks: dict, cflag=False, info=None) -> None:
        r'''
        Parameters
        ----------
        dual: tuple[int], like (0, 0, 1, 1), denote the type of vector spaces associated with each bond
            CONVENTION: 0: super vector space (outgoing arrow); 1: dual super vector space (incoming arrow)
        shape: tuple[tuple], denote the shape of even and odd sectors like ((1, 2), (3, 4), ...)
        blocks: dict, key: (parity quantum numbers, value: degeneracy tensor)
        parity: int, 0 or 1
        cflag: bool, set the tensor with complex number entries or not
            DEFAULT: False
        '''

        self._dual = dual
        # rank (r, s), r: outgoing #, s: incoming #
        self._rank = (self._dual.count(0), self._dual.count(1))
        self._ndim = len(self._dual)

        self._shape = shape

        self._blocks = blocks
        # sort the blocks by viewing 'key' as a binary number
        self._blocks = dict(sorted(self._blocks.items(), key=lambda item: int(''.join(str(e) for e in item[0]), 2)))
        # check the parity and block shapes
        parity = sum(next(iter(self._blocks.keys()))) & 1
        for k, v in self._blocks.items():
            assert parity == sum(k) & 1, 'quantum number is not consistent with the parity'
            block_shape = [self._shape[i][q] for i, q in enumerate(k)]
            assert v.shape == tuple(block_shape), ('block shape %s is not consistent with the whole shape %s' % (v.shape, shape))
            if cflag:
                self._blocks[k] = v.cdouble()

        self._parity = parity
        self._dtype = tuple(self._blocks.values())[0].dtype
        self._cflag = cflag
        self._info = info

    @property
    def dual(self) -> tuple:
        return self._dual

    @property
    def shape(self) -> tuple:
        return self._shape

    @property
    def even_shape(self) -> tuple:
        return tuple([d[0] for d in self._shape])

    @property
    def odd_shape(self) -> tuple:
        return tuple([d[1] for d in self._shape])

    @property
    def whole_shape(self) -> tuple:
        return tuple([sum(d) for d in self._shape])

    @property
    def parity(self) -> int:
        return self._parity

    @property
    def dtype(self):
        return self._dtype

    @property
    def cflag(self):
        return self._cflag

    @property
    def info(self):
        return self._info

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def ndim(self) -> int:
        return self._ndim

    def blocks(self) -> dict:
        return self._blocks

    def norm(self) -> float:
        nors = [t.norm().item() for t in self._blocks.values()]
        return sum(nors)

    def max(self):
        r'''
        return the max value in all blocks
        '''

        ms = []
        for t in self._blocks.values():
            if 0 != t.numel():
                ms.append(t.abs().max().item())

        return max(ms)

    @classmethod
    def rand(cls, dual: tuple, shape: tuple, parity: int, cflag=False, info=None):
        r'''
        generate a random Z2gTensor

        Parameters
        ----------
        dual: refer to '__init__()'
        shape: tuple[tuple], refer to '__init__()'
        parity: int, 0 or 1, even or odd
        '''

        assert len(shape) == len(dual), 'tensor shape not match'
        qns_list = [(0, 1) for i in range(len(shape))]
        qns = tuple(itertools.product(*qns_list))
        blocks = {}
        for q in qns:
            # the structure tensor condition
            if parity == sum(q) & 1:
                block_shape = [shape[i][p] for i, p in enumerate(q)]
                if cflag:
                    blocks[q] = torch.rand(block_shape)+1.j*torch.rand(block_shape)
                else:
                    blocks[q] = torch.rand(block_shape)

        return cls(dual, shape, blocks, cflag, info)

    @staticmethod
    def permute_qnums(qs: tuple, dims: tuple):
        r'''
        permute a tuple of quantum numbers 'qs' and
        return the puermuted quantum numbers as well as the sign

        example
        -------
        dims: (0, 1, 2, 3, 4) -> (4, 1, 2, 0, 3)
        qs:   (0, 0, 1, 1, 1) -> (1, 0, 1, 0, 1)

        Parameters
        ----------
        qs: tuple or list, Z2 parity quantum numbers for a block
        dims: tuple or list, the desired ordering of dimensions

        Returns
        -------
        temp_qs: tuple[int], permuted 'qs'
        sign: int, the fermionic sign arised from permutation
        '''

        assert len(qs) == len(dims), 'permutation dims not long as quantum numbers'
        num = len(qs)

        '''
        sgn = 1
        temp, new_qs = list(range(num)), list(qs)
        # permute 'temp' to 'dims' step by step
        for j in range(num):
            if temp[j] != dims[j]:
                k = temp.index(dims[j])
                # fermions crossed
                sgn *= (-1)**(new_qs[k]*sum(new_qs[j:k]))
                temp.insert(j, temp.pop(k))
                new_qs.insert(j, new_qs.pop(k))
        '''

        sgn = 1
        temp, new_qs = list(range(num)), list(qs)

        for j in range(num):
            if dims[j] != temp[j]:
                # indices for two elements
                k, l = temp.index(temp[j]), temp.index(dims[j])
                # swap two fermions:
                # each one crosses fermions between + permute themselves
                sgn *= (-1)**((new_qs[k]+new_qs[l])*sum(new_qs[k+1:l])+new_qs[k]*new_qs[l])
                # swap elements k and l
                temp[k], temp[l] = temp[l], temp[k]
                new_qs[k], new_qs[l] = new_qs[l], new_qs[k]

        return tuple(new_qs), sgn

    def push_blocks(self):
        r'''
        push blocks of Z2gTensor into a dense tensor, filled with zeros
        '''
       
        frame = torch.zeros(self.whole_shape, dtype=self._dtype)
        for q, t in self._blocks.items():
            # bulid slices for the block
            ss = []
            for i, p in enumerate(q):
                if 0 == p:
                    ss.append(slice(0, self._shape[i][0]))
                elif 1 == p:
                    ss.append(slice(self._shape[i][0], sum(self._shape[i])))
            frame[ss] = t

        return frame

    def permute(self, dims: tuple):
        r'''
        permute Z2gTensor

        Parameters
        ----------
        dims: tuple or list[int], the desired ordering of dimensions
            the j-th bond of the returned tensor corresponds to the dims[j] of the input
        '''
        
        # permute dual
        new_dual = tuple([self._dual[i] for i in dims])
        # permute shape
        new_shape = tuple([self._shape[i] for i in dims])
        # permute each blocks
        new_blocks = {}
        for q, t in self._blocks.items():
            new_q, sign = self.permute_qnums(q, dims)
            new_blocks[new_q] = sign*t.permute(dims)

        return Z2gTensor(new_dual, new_shape, new_blocks, self._parity, self._info)

    @classmethod
    def contract(cls, *args: any, bosonic_dims=(), info=None):
        r'''
        contract
        powered by 'opt_einsum'
        https://optimized-einsum.readthedocs.io/en/stable/

        Parameters
        ----------
        args[0]: contraction string, MUST specify the output string with '->'
        args[1], args[2], ...: Z2gTensor, tensors to be contracted
        bosonic_dims: list[str] or tuple[str], dims to be contracted without the super-sign
        info:

        Returns
        -------
        if all bonds are contracted, return a torch.tensor();
        else, return a Z2gTensor
        '''

        def _check_qnums(qs: tuple, pairs: tuple):
            r'''
            ancillary for 'contract()'
            check the fused quantum numbers if valid given 'pairs' to be contracted
            ONLY IF the pair quantum numbers are the same

            Parameters
            ----------
            qs: tuple[int], a tuple of Z2 quantum numbers
            pairs: list[tuple] or tuple[tuple], a list (or tuple) of pairs of quantum numbers' positions
            '''

            for i, j in pairs:
                if qs[i] ^ qs[j]:
                    return False

            return True

        def _adjacent_pairs_sign(dual: tuple, qs: tuple, pairs: list) -> int:
            r'''
            ancillary for 'contract()'
            find the Z2-fermionic sign if we move pairs of quantum numbers to be adjacent according to the 'dual'
            ALWAYS permute the outgoing quantum number (dual 0) behind the incoming one (bond-type 1) as: 1 <-- 0

            Parameters
            ----------
            dual: tuple[int], a tuple denoting vector spaces types
            qs: tuple[int], a tuple of Z2 quantum numbers
            pairs: list[tuple] or tuple[tuple], a list (or tuple) of pairs of quantum numbers' positions

            Returns
            -------
            sign: int, fermionic sign arised from this permutation
            '''

            # used to track the sign
            temp_qs = list(qs)
            sign = 1
            # i < j
            for i, j in pairs:
                # permute 'qs[j]' behind 'qs[i]'
                # if dual[i] ^ dual[j]:
                    # sign *= (-1)**(temp_qs[j]*sum(temp_qs[(i+1):j]))

                # case: i -<- j
                # permute 'qs[j]' behind 'qs[i]'
                if 1 == dual[i] and 0 == dual[j]:
                    sign *= (-1)**(temp_qs[j]*sum(temp_qs[(i+1):j]))
                    # set to zero as this very pair has been already contracted
                    # !do not pop out since we need to keep other pairs' position unchanged
                    temp_qs[i], temp_qs[j] = 0, 0
                # case: i ->- j
                # permute 'qs[i]' behind 'qs[j]'
                elif 0 == dual[i] and 1 == dual[j]:
                    sign *= (-1)**(temp_qs[i]*sum(temp_qs[(i+1):(j+1)]))
                    temp_qs[i], temp_qs[j] = 0, 0
                else:
                    raise TypeError('bonds be contracted are NOT matched (one and a dual one, or vice versa)')

            return sign

        oe_str = args[0]
        num_gt = oe_str.count(',')+1
        assert num_gt == len(args)-1, 'number of Z2gTensors and the string epxression do NOT match'
        # split
        oe_str_split = oe_str.split('->')
        assert len(oe_str_split) > 1, 'you need to specify the output string after ->'

        # contraction
        fused_str = oe_str_split[0].replace(',', '')
        fused_length = len(fused_str)
        assert fused_length == sum([args[i+1].ndim for i in range(num_gt)]), 'string length and total dimensions do NOT match'

        # find all the pairs to be contracted
        # each pair is represented by positions in 'fused_str'
        contracted_pairs = []
        uncontracted_str = ''
        for c in fused_str:
            # duplicate characters
            if 2 == fused_str.count(c):
                first = fused_str.find(c)
                second = fused_str.find(c, first+1)
                # if the second is found
                if second > 0 and (first, second) not in contracted_pairs:
                    contracted_pairs.append((first, second))
            # uncontracted characters
            elif 1 == fused_str.count(c):
                uncontracted_str += c

        per_dims = [uncontracted_str.index(v) for i, v in enumerate(oe_str_split[1])]
        assert len(per_dims) == len(uncontracted_str), 'permutation string after -> does NOT match the uncontracted ones'

        # flatten pairs
        contracted_bonds = cls.flatten_tuples(contracted_pairs)
        # fuse to a combined tuple of dual and shape
        fused_dual = cls.flatten_tuples([args[i+1].dual for i in range(num_gt)])
        fused_shape = cls.flatten_tuples([args[i+1].shape for i in range(num_gt)])

        # build new dual and shape by removing the contracted pairs
        new_dual = [fused_dual[i] for i in range(fused_length) if i not in contracted_bonds]
        new_shape = [fused_shape[i] for i in range(fused_length) if i not in contracted_bonds]

        # permute dual and block shape
        new_dual = [new_dual[i] for i in per_dims]
        new_shape = [new_shape[i] for i in per_dims]

        # list all combinations of quantum numbers
        qns_list = [tuple(args[i+1].blocks().keys()) for i in range(num_gt)]
        qns_comb = tuple(itertools.product(*qns_list))
        new_blocks = {}
        for qs in qns_comb:
            fused_qs = cls.flatten_tuples(qs)
            # only identical quantum numbers can be contracted
            if _check_qnums(fused_qs, contracted_pairs):
                # contraction sign
                c_sign = _adjacent_pairs_sign(fused_dual, fused_qs, contracted_pairs)
                block_tensors = [args[i+1].blocks()[q] for i, q in enumerate(qs)]
                # new quantum numbers by removing contracted ones
                new_qs = tuple([fused_qs[i] for i in range(fused_length) if i not in contracted_bonds])
                # permutation to the desired order after the permutation
                # print(new_qs, per_dims)
                new_qs, p_sign = cls.permute_qnums(new_qs, per_dims)
                # possible summations of the contracted quantum numbers
                block_shape = [new_shape[i][q] for i, q in enumerate(new_qs)]
                bare = torch.zeros(block_shape, dtype=args[1].dtype)
                block = oe.contract(oe_str, *block_tensors, backend='torch')
                assert bare.shape == block.shape, 'shape of the contracted block is NOT correct'
                new_blocks[new_qs] = new_blocks.get(new_qs, bare)+c_sign*p_sign*block

        if 0 == len(new_dual):
            return new_blocks[()]
        else:
            return cls(dual=tuple(new_dual), shape=tuple(new_shape), blocks=new_blocks, cflag=args[1].cflag, info=info)

class GTensor(Z2gTensor):
    r'''
    class of Z2-symmetric Z2-graded tensor/Grassmann tensor
    to be used in fermionic systems

    two essential properties of GTensor:
    1. must fullfil the Abelian Z2 symmetric fusion rule: outgoing charges = incoming charges
        -> only EVEN parity is valid
    2. obey the Z2-graded braiding statistics, namely super tensor product structure, to capture fermionic anticommutation physics
    '''

    def __init__(self, dual: tuple, shape: tuple, blocks: dict, cflag=False, info=None) -> None:
        r'''
        Parameters
        ----------
        dual: tuple[int], (0, 0, 1, 1), denoting the type of vector spaces associated with each bond
            0: super vector space (outgoing); 1: dual super vector space (incoming)
        shape: tuple[tuple]
        blocks: dict,
            key: parity quantum numbers, value: bosonic degeneracy tensor
        '''

        def _check_Z2symmetry(dual, q):

            qo, qi = 0, 0
            for b, k in zip(dual, q):
                if b:
                    qi += k
                else:
                    qo += k
            # Z2 parity fusion condition: in == out
            if (qo % 2) == (qi % 2):
                return True
            else:
                return False

        # check the Z2 symmetry
        for k in blocks:
            assert _check_Z2symmetry(dual, k), 'quantum number not fulfill the Z2 parity symmetry'

        # list ALL possible quantum numbers fulfilling the Z2-symmetry
        qns_list = [(0, 1)]*len(dual)
        qns = list(itertools.product(*qns_list))
        sym_qns = [q for q in qns if _check_Z2symmetry(dual, q)]
        # fill empty blocks
        # useful for the addition or subtraction of GTensors
        for qs in sym_qns:
            if qs not in blocks:
                block_shape = [shape[i][v] for i, v in enumerate(qs)]
                blocks[qs] = torch.zeros(block_shape)

        super(GTensor, self).__init__(dual, shape, blocks, cflag, info)
        # check the dual
        assert self._rank[0] > 0 and self._rank[1] > 0, 'GTensor must have both outgoing and incoming bonds'

    def __mul__(self, other):
        r'''
        left multiplication overloading
        GTensor * float or complex
        '''

        new_blocks = {}
        self._blocks.update((x, other*y) for x, y in self._blocks.items())

        return GTensor(self._dual, self._shape, new_blocks, cflag=False, info=self._info)

    def __rmul__(self, other):
        r'''
        right multiplication overloading
        float or complex * GTensor
        '''

        new_blocks = {}
        new_blocks.update((x, other*y) for x, y in self._blocks.items())

        return GTensor(self._dual, self._shape, new_blocks, cflag=False, info=self._info)

    def __add__(self, other):
        r'''
        addition overloading
        '''

        assert self._dual == other._dual, 'GTensors with different duals CANNOT be added'
        assert self._shape == other._shape, 'GTensors with different shapes CANNOT be added'

        new_blocks = {}
        new_blocks.update((q, t+other.blocks()[q]) for q, t in self._blocks.items())

        return GTensor(self._dual, self._shape, new_blocks, cflag=False, info=self._info)

    @classmethod
    def zeros(cls, dual: tuple, shape: tuple, cflag=False, info=None):
        r'''
        generate a zero GTensor

        Parameters
        ----------
        dual: refer to '__init__()'
        shape: tuple or list, shape of degenerated bosonic tensors
        '''

        assert len(shape) == len(dual), 'tensor shape not match'
        qns_list = [(0, 1)]*len(dual)
        qns = tuple(itertools.product(*qns_list))
        blocks = {}
        for q in qns:
            # the structure tensor condition
            if 0 == sum(q) & 1:
                block_shape = [shape[i][v] for i, v in enumerate(q)]
                blocks[q] = torch.zeros(block_shape)

        return cls(dual, shape, blocks, cflag, info)

    @classmethod
    def eye(cls, dual: tuple, shape: tuple, cflag=False):
        r'''
        generate an identity GTensor matrix

        Parameters
        ----------
        dual: refer to '__init__()'
        shape: tuple[tuple], shapes of all bonds
        '''

        assert shape[0] == shape[1], 'GTensor identity must have identical dimensions'

        blocks = {}
        blocks[(0, 0)] = torch.eye(shape[0][0])
        blocks[(1, 1)] = torch.eye(shape[0][1])

        if (0, 1) == dual:
            return cls((0, 1), shape, blocks, cflag)
        elif (1, 0) == dual:
            return cls((0, 1), shape, blocks, cflag).permute((1, 0))
        else:
            raise ValueError('input dual is not valid')

    @classmethod
    def fermion_parity_operator(cls, dual: tuple, shape: tuple, cflag=False):
        r'''
        generate a fermion parity operator

        References
        ----------
        PHYSICAL REVIEW B 95, 075108 (2017)

        Parameters
        ----------
        shape: tuple[tuple], shapes of all bonds
        '''

        assert shape[0] == shape[1], 'GTensor parity operator must have identical dimensions'

        blocks = {}
        blocks[(0, 0)] = torch.eye(shape[0][0])
        blocks[(1, 1)] = -1.0*torch.eye(shape[0][1])

        if (0, 1) == dual:
            return cls((0, 1), shape, blocks, cflag)
        elif (1, 0) == dual:
            return cls((0, 1), shape, blocks, cflag).permute((1, 0))
        else:
            raise ValueError('input dual is not valid')

    @classmethod
    def rand(cls, dual: tuple, shape: tuple, cflag=False, info=None):
        r'''
        generate a random GTensor

        Parameters
        ----------
        dual: refer to '__init__()'
        shape: tuple[tuple], refer to '__init__()'
        parity: int, 0 or 1, even or odd
        '''

        assert len(shape) == len(dual), 'tensor shape not match'
        qns_list = [(0, 1)]*len(dual)
        qns = tuple(itertools.product(*qns_list))
        blocks = {}
        for q in qns:
            # the structure tensor condition
            if 0 == sum(q) & 1:
                block_shape = [shape[i][v] for i, v in enumerate(q)]
                if cflag:
                    blocks[q] = torch.rand(block_shape)+1.j*torch.rand(block_shape)
                else:
                    blocks[q] = torch.rand(block_shape)

        return cls(dual, shape, blocks, cflag, info)

    @classmethod
    def rand_diag(cls, dual: tuple, shape: tuple, cflag=False, info=None):
        r'''
        generate an identity GTensor

        Parameters
        ----------
        dual: refer to '__init__()'
        dims: tuple[int], denote the dims of even and odd sector
        '''

        assert shape[0] == shape[1], 'diagonal GTensor must have identical dimensions'

        if (0, 1) == dual:
            sgn = 1.0
        elif (1, 0) == dual:
            sgn = -1.0
        else:
            raise ValueError('input dual is not valid')

        blocks = {}
        blocks[(0, 0)] = torch.sort(torch.rand(shape[0][0]), descending=True).values.diag()
        blocks[(1, 1)] = sgn*torch.sort(torch.rand(shape[0][1]), descending=True).values.diag()

        return cls(dual, shape, blocks, cflag, info)

    def cdouble(self):
        r'''
        convert blocks of this GTensor to complex128
        '''

        for k, v in self._blocks.items():
            self._blocks[k] = v.cdouble()

        self._dtype = tuple(self._blocks.values())[0].dtype

    def conj(self, reverse=False):
        r'''
        normal conjugation of a GTensor

        Parameters
        ----------
        reverse: bool, if all the bonds are reversed or not
        '''

        # reverse
        dims = [i for i in range(self._ndim)]
        dims.reverse()
        new_dual = [d ^ 1 for d in self._dual]
        new_dual.reverse()
        new_shape = list(self._shape)
        new_shape.reverse()
        # build new blocks
        new_blocks = {}
        for q, t in self._blocks.items():
            new_q = list(q)
            new_q.reverse()
            new_blocks[tuple(new_q)] = t.conj().permute(dims)

        # permute back to the original order if needed
        if reverse:
            return GTensor(dual=tuple(new_dual), shape=tuple(new_shape), blocks=new_blocks, cflag=self._cflag)
        else:
            return GTensor(dual=tuple(new_dual), shape=tuple(new_shape), blocks=new_blocks, cflag=self._cflag).permute(dims)

    def graded_conj(self, free_dims=(), side=0, reverse=False):
        r'''
        graded conjugation of a GTensor

        Parameters
        ----------
        free_dims: tuple[int], free bonds not to be conjugated
        side: int, 0 or 1, namely the dagger on the left or right
            crrespond to the unitary condition from QR or super-QR
        reverse: bool, if all the bonds are reversed or not
        '''

        # reverse
        dims = [i for i in range(self._ndim)]
        dims.reverse()
        new_dual = [d ^ 1 for d in self._dual]
        new_dual.reverse()
        new_shape = list(self._shape)
        new_shape.reverse()
        # build new blocks
        new_blocks = {}
        for q, t in self._blocks.items():
            # possible super trace sign should be considered
            if 1 == side:
                sgns = [q[i]*(self._dual[i] ^ 1) for i in range(self._ndim) if i not in free_dims]
            elif 0 == side:
                sgns = [q[i]*self._dual[i] for i in range(self._ndim) if i not in free_dims]
            sign = (-1)**sum(sgns)
            new_q = list(q)
            new_q.reverse()
            new_blocks[tuple(new_q)] = sign*t.conj().permute(dims)

        # permute back to the original order if needed
        if reverse:
            return GTensor(dual=tuple(new_dual), shape=tuple(new_shape), blocks=new_blocks)
        else:
            return GTensor(dual=tuple(new_dual), shape=tuple(new_shape), blocks=new_blocks).permute(dims)

    # def graded_conj(self, iso_dims: tuple, side: int, super_flag=False):
    #     r'''
    #     graded conjugation of GTensor

    #     Parameters
    #     ----------
    #     iso_dims: tuple[int], uncontracted free bonds
    #     side: int, 0 (left) or 1 (right) conjugation
    #     '''

    #     # reverse
    #     dims = list(range(self._ndim))
    #     dims.reverse()
    #     new_dual = [d ^ 1 for d in self._dual]
    #     new_dual.reverse()
    #     new_shape = list(self._shape)
    #     new_shape.reverse()
    #     # build new blocks
    #     new_blocks = {}
    #     for q, t in self._blocks.items():
    #         # possible super trace sign should be considered
    #         sgns = []
    #         for i in range(self._ndim):
    #             # contracted bonds
    #             if i not in iso_dims:
    #                 sgns.append(q[i]*(self._dual[i] ^ side))
    #             # uncontracted free bonds
    #             else:
    #                 sgns.append(q[i]*(self._dual[i] ^ (side ^ 1)))
    #         sign = (-1)**sum(sgns)
    #         new_q = list(q)
    #         new_q.reverse()
    #         new_blocks[tuple(new_q)] = sign*t.conj().permute(dims)

    #     # if super_flag:
    #         # pass

    #     return GTensor(dual=tuple(new_dual), shape=tuple(new_shape), blocks=new_blocks, info=self.info)

    def permute(self, dims: tuple):
        r'''
        permute GTensor

        Parameters
        ----------
        dims: tuple or list[int], the desired ordering of dimensions
            the j-th bond of the returned tensor corresponds to the dims[j] of the input
        '''
        
        # permute dual
        new_dual = tuple([self._dual[i] for i in dims])
        # permute shape
        new_shape = tuple([self._shape[i] for i in dims])
        # permute each blocks
        new_blocks = {}
        for q, t in self._blocks.items():
            new_q, sign = self.permute_qnums(q, dims)
            new_blocks[new_q] = sign*t.permute(dims)

        return GTensor(dual=new_dual, shape=new_shape, blocks=new_blocks, cflag=self._cflag, info=self._info)

    def fuse_bonds(self, bonds: tuple, pos=None, sub_dims=None):
        r'''
        fuse bonds of the GTensor to a new one

        Parameters
        ----------
        bonds: tuple[int], bonds to be fused, must share a same dual
        pos: int, position of the new fused bond to be placed; default: at the tail -1
        sub_dims: tuple[int], optional, permutations of inner bonds in the subspace
        '''

        def _fused_subspace(num: int, qnum: int):
            r'''
            construct the basis of a fused subspace

            Parameters
            ----------
            num: int, how many bonds are going to be fused
            qnum: int, 0 or 1, the fused quantum number

            Returns
            -------
            qs: list[tuple], all possible quantum numbers in this sector
            '''
            
            qs_list = [(0, 1)]*num
            qs_all = list(itertools.product(*qs_list))
            qs = [q for q in qs_all if qnum == sum(q) % 2]

            return qs

        num_fuse = len(bonds)
        num_left = self._ndim-num_fuse

        # permute the bonds together to the end
        dims = list(range(self._ndim))
        for b in bonds:
            assert self._dual[b] == self._dual[bonds[0]], 'fused bonds must share a same dual'
            dims.remove(b)
            dims.append(b)
        temp = self.permute(tuple(dims))
        # how many bonds which are not going to fused
        new_dual = temp.dual[:num_left]+(temp.dual[-1],)
        # background zero tensor
        # new blocks are build by fuse those Z2 quantum numbers
        # each block contains degenerated subpace
        # for example, if num_fuse=3, then dim_deg=4:
        # (0, 0, 0) -> 0
        # (1, 1, 0) -> 0
        # (1, 0, 1) -> 0
        # (0, 1, 1) -> 0
        deg_qs_e = _fused_subspace(num_fuse, 0)
        deg_qs_o = _fused_subspace(num_fuse, 1)
        dim_deg = len(deg_qs_e)

        temp_shape = temp.block_shape
        fuse_shape = temp_shape[:num_left]+(math.prod(temp_shape[num_left:]),)
        new_block_shape = fuse_shape[:len(fuse_shape)-1]+(fuse_shape[-1]*dim_deg,)
        bare = torch.zeros(fuse_shape)
        back_ground = torch.stack([bare]*dim_deg, dim=len(fuse_shape))
        back_ground = torch.reshape(back_ground, new_block_shape)
        # print(temp_shape, fuse_shape, new_block_shape)
        new_blocks = {}
        for q, t in temp.blocks().items():
            # quantum numbers to be fused
            fq = q[num_left:]
            new_q = q[:num_left]+(sum(fq) % 2,)
            # even
            if 0 == (sum(fq) % 2):
                # place block tensor in the right place in the subspace
                idx = deg_qs_e.index(fq)
                temp = [bare]*dim_deg
                temp[idx] = t.reshape(fuse_shape)
                fused_t = torch.stack(temp, dim=len(fuse_shape))
                sign = 1
                if sub_dims is not None:
                    qs, sign = self.permute_qnums(fq, sub_dims)
                new_t = sign*torch.reshape(fused_t, new_block_shape)
            # odd
            elif 1 == (sum(fq) % 2): 
                idx = deg_qs_o.index(fq)
                temp = [bare]*dim_deg
                temp[idx] = t.reshape(fuse_shape)
                fused_t = torch.stack(temp, dim=len(fuse_shape))
                sign = 1
                if sub_dims is not None:
                    qs, sign = self.permute_qnums(fq, sub_dims)
                new_t = sign*torch.reshape(fused_t, new_block_shape)
            # a new block
            new_blocks[new_q] = new_blocks.get(new_q, back_ground)+new_t

        gt = GTensor(dual=new_dual, blocks=new_blocks, info=self._info)

        # if permutation needed
        if pos is not None:
            dims = list(range(gt.ndim))
            dims.insert(pos, dims.pop(-1))
            gt = gt.permute(dims)
           
        return gt

    def parity_mat_qnums(self, group_dims: tuple, parity: int):
        r'''
        divide this GTensor into two parts I and J
        return all possible quantum number combinations in each parity sector
        as the basis to bulid parity matrix in each sector

        Parameters
        ----------
        group_dims: tuple[tuple], two tuples denote dims in I and J
        parity: which parity sector

        Returns
        -------
        qns_I,J: list[tuple], different quantum numbers for I and J
        '''

        assert list(range(self._ndim)) == sorted(group_dims[0]+group_dims[1]), 'input group_dims is not valid'

        qns_I, qns_J = [], []
        for q in self._blocks:
            qI, qJ = tuple([q[i] for i in group_dims[0]]), tuple([q[i] for i in group_dims[1]])
            # Z2 fusion rule does not depend on the dual: +1 or -1 gives the same result
            # but U1 or other point symmetries depend
            if parity == (sum(qI) % 2):
                qns_I.append(qI)
                qns_J.append(qJ)

        # reomve possible duplicates
        return tuple(dict.fromkeys(qns_I).keys()), tuple(dict.fromkeys(qns_J).keys())

    def parity_mat(self, qns: tuple, group_dims: tuple, parity: int):
        r'''
        build the parity symmetric matrix $C_{IJ}^{00, 11}$

        Parameters
        ----------
        qns: tuple, parity quantum numbers, as the basis
            refer to: parity_matrix_qnums()
        group_dims: tuple[tuple]
        parity: int, parity sector

        Returns
        -------
        mat: tensor
        '''

        qns_I, qns_J = qns[0], qns[1]
        # find the corresponding degeneracy dimensions
        dims_I, dims_J = [], []
        for q in qns_I:
            dims = [self._shape[d][q[i]] for i, d in enumerate(group_dims[0])]
            dims_I.append(math.prod(dims))
        for q in qns_J:
            dims = [self._shape[d][q[i]] for i, d in enumerate(group_dims[1])]
            dims_J.append(math.prod(dims))

        mat = torch.zeros(sum(dims_I), sum(dims_J), dtype=self._dtype)
        # reshape and push each block into the matrix
        for q, t in self._blocks.items():
            qI, qJ = tuple([q[l] for l in group_dims[0]]), tuple([q[l] for l in group_dims[1]])
            if parity == (sum(qI) % 2):
                # locate indices
                i, j = qns_I.index(qI), qns_J.index(qJ)
                # !reshaped tensor may not be contiguous, which cannot be operated by 'view()' or 'kron()'
                m = t.reshape(dims_I[i], dims_J[j]).contiguous()
                # locate the block
                sI, sJ = slice(sum(dims_I[:i]), sum(dims_I[:i+1])), slice(sum(dims_J[:j]), sum(dims_J[:j+1]))
                mat[sI, sJ] = m

        return mat

    @staticmethod
    def restore_blocks(mat, qns: tuple, shape: tuple, group_dims: tuple) -> dict:
        r'''
        restore GTensor blocks from a parity matrix

        Parameters
        ----------
        mat: tensor, a parity matrix
        qns: tuple[tuple], quantum numbers of two parts I and J
        shape: tuple[tuple], shape of the desired GTensor
        group_dims: tuple[tuple], two groups of dimensions for I and J
        '''

        assert len(shape) == (len(group_dims[0])+len(group_dims[1])), 'group_dims and shape are not consistent'

        qns_I, qns_J = qns[0], qns[1]
        # find the corresponding degeneracy dimensions
        dims_I, dims_J = [], []
        for q in qns_I:
            dims = [shape[d][q[i]] for i, d in enumerate(group_dims[0])]
            dims_I.append(math.prod(dims))
        for q in qns_J:
            dims = [shape[d][q[i]] for i, d in enumerate(group_dims[1])]
            dims_J.append(math.prod(dims))

        blocks = {}
        for c in itertools.product(qns_I, qns_J):
            qI, qJ = c[0], c[1]
            # restore the quantum number
            q = [None]*len(shape)
            for i, p in enumerate(group_dims[0]):
                q[p] = qI[i]
            for i, p in enumerate(group_dims[1]):
                q[p] = qJ[i]
            # locate the block
            i, j = qns_I.index(qI), qns_J.index(qJ)
            sI, sJ = slice(sum(dims_I[:i]), sum(dims_I[:i+1])), slice(sum(dims_J[:j]), sum(dims_J[:j+1]))
            m =  mat[sI, sJ].clone()
            # reshape the block
            block_shape = [shape[i][l] for i, l in enumerate(q)]
            blocks[tuple(q)] = m.reshape(block_shape)

        return blocks

    @classmethod
    def construct_from_parity_mats(cls, mats: tuple, qns: tuple, dual: tuple, shape: tuple, group_dims: tuple, cflag=False, info=None):
        r'''
        build a GTensor from two parity matrices

        Parameters
        ----------
        mats: tuple[Tensor], even and odd parity matrices
        qns: tuple[tuple], even and odd parity sector quantum numbers
        dual: tuple[int],
        shape: tuple[int], shape of block tensors
        group_dims: tuple[tuple], two groups of dimensions for I and J
        '''

        blocks_e = cls.restore_blocks(mats[0], qns[0], shape, group_dims)
        blocks_o = cls.restore_blocks(mats[1], qns[1], shape, group_dims)
        blocks = dict(blocks_e)
        blocks.update(blocks_o)

        return cls(dual=dual, shape=shape, blocks=blocks, cflag=cflag, info=info)

    @classmethod
    def extract_blocks(cls, dt, dual: tuple, shape: tuple, cflag=False, info=None):
        r'''
        build a GTensor from dense tensor
        
        Parameters
        ----------
        dt: torch.tensor
        dual: tuple[int], dual of the desired GTensor
        shape: tuple[tuple], shape of the desired GTensor
        '''

        whole_shape = tuple([sum(d) for d in shape])
        # print(dt.shape, whole_shape)
        assert tuple(dt.shape) == whole_shape, 'shape NOT consistent with the input dense tensor'

        qns_list = [(0, 1)]*len(dual)
        qns = tuple(itertools.product(*qns_list))
        blocks = {}
        for q in qns:
            # the structure tensor condition
            # Z2 parity conserved
            if 0 == sum(q) & 1:
                ss = []
                for i, p in enumerate(q):
                    if 0 == p:
                        ss.append(slice(0, shape[i][0]))
                    elif 1 == p:
                        ss.append(slice(shape[i][0], sum(shape[i])))

                blocks[q] = dt[tuple(ss)]

        return cls(dual=dual, shape=shape, blocks=blocks, cflag=cflag, info=info)

# ----------------------------------
# module-level functions for fermi tensor
# ----------------------------------

def z2gcontract(*args: any, info=None) -> Z2gTensor:
    r'''
    Z2gTensor contraction
    '''

    return Z2gTensor.contract(*args, info=info)

def gpermute(t: GTensor, dims: tuple) -> GTensor:
    return t.permute(dims)

def gcontract(*args: any, bosonic_dims=(), info=None) -> GTensor:
    r'''
    GTensor contraction
    '''

    return GTensor.contract(*args, bosonic_dims=bosonic_dims, info=info)
