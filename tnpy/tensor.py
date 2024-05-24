import itertools
from scipy.stats import unitary_group as ug
import math
import opt_einsum as oe

import torch
torch.set_default_dtype(torch.float64)

def contract(*args: any):
    return oe.contract(*args, backend='torch')

def random_isometric_tensor(shape: tuple, iso_axis: int):
    r'''
    return an isometric Tensor filled with random numbers

    Parameters
    ----------
    shape: tuple
    iso_axis: int, the free bond for the isometry condition;
              its dimension should be larger or equil to the sum of the rest

    Examples
    --------
    If T_{abcde} is isometric with 'iso_axis=3', then:
    sum_{abce} T_{abcde}^{*}T_{abcfe} = \delta_{df}
    '''

    temp_shape = list(shape)
    # move the 'iso_axis' to the end
    try:
        temp_shape.append(temp_shape.pop(iso_axis))
    except IndexError:
        raise IndexError('iso_axis=%s not existing!' % iso_axis) from None

    dim_0, dim_1 = math.prod(temp_shape[:-1]), temp_shape[-1]

    assert dim_0 >= dim_1

    temp_mat = ug.rvs(dim_0)[:, :dim_1]
    temp_ten = torch.reshape(torch.as_tensor(temp_mat), temp_shape)

    # permute back to the initial order
    back_axes = list(range(len(shape)))
    back_axes.insert(iso_axis, back_axes.pop(-1))

    return temp.permute(tuple(back_axes))

class U1Tensor(object):
    r'''
    U1 symmetric tensor
    '''

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
