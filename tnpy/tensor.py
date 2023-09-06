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
