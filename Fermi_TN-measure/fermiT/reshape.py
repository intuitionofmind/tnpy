from .core import FermiT
import numpy as np
from time import time

def _get_group_index_parameters(DS, DE):

    name = str(DS * 10000 + DE*10)
    if name in FermiT.all_index_parameters.keys():
        paras = FermiT.all_index_parameters[name]
        starts, ends, start_val, end_val, intervals = paras
        return starts, ends, start_val, end_val, intervals
    t0 = time()

    dim = DS.shape[0]
    paras = np.concatenate([np.zeros(len(DS)), DE, DS]).reshape(3,dim)
    # print(paras)
    
    starts = np.zeros((2**dim,dim), dtype=int)
    ends = np.zeros((2**dim,dim), dtype=int)
    start_val = np.zeros((2**dim,), dtype=int)
    end_val = np.zeros((2**dim,), dtype=int)
    intervals = np.zeros((2**dim,), dtype=int)

    begin = 0
    for i in range(0,2**dim):
        s = np.binary_repr(i, width=dim)
        if s.count("1") % 2 != 0: continue
        for j in range(dim):
            starts[i][j] = paras[int(s[j]),j]
            ends[i][j] = paras[int(s[j])+1,j]
        start_val[i] = begin   
        intervals[i] = np.prod(ends[i] - starts[i])
        begin = begin + intervals[i]
        end_val[i] = begin
        # print(i, starts[i], ends[i], start_val[i], end_val[i])

    for i in range(0,2**dim):
        s = np.binary_repr(i, width=dim)
        if s.count("1") % 2 == 0: continue
        for j in range(dim):
            starts[i][j] = paras[int(s[j]),j]
            ends[i][j] = paras[int(s[j])+1,j]
        start_val[i] = begin   
        intervals[i] = np.prod(ends[i] - starts[i])
        begin = begin + intervals[i]
        end_val[i] = begin
        # print(i, starts[i], ends[i], start_val[i], end_val[i])

    # print(DS, DE, name)
    FermiT.all_index_parameters[name] = [starts, ends, start_val, end_val, intervals]
    return starts, ends, start_val, end_val, intervals

def merge_axes(a: FermiT, shapeT: list[list[int]]):
    """
    If you want to group fermionic tensors, be careful that bonds, which will be
    coupled into one band, must be in the same dual space. Otherwise, there will 
    be error reporting!
    For example, the dual of the first band is 0 and the second is 1. It is wrong
    to couple them together.
    """
    order = sum(shapeT, [])
    a = a.transpose(*order)
    DS0 = a.DS
    DE0 = a.DE
    dual0 = a.dual
    
    DS = [None]*len(shapeT)
    DE = [None]*len(shapeT)
    dual = [None]*len(shapeT)

    val0 = a.val.reshape(np.prod(DS0))
    start_dim = 0
    for i in range(len(shapeT)):
        val = np.zeros(np.prod(DS0), dtype=complex)
        dim = len(shapeT[i])
        end_dim = start_dim + dim

        starts, ends, start_val, end_val, intervals =\
            _get_group_index_parameters(DS0[start_dim:end_dim], DE0[start_dim:end_dim])

        DS[i] = np.prod(DS0[start_dim:end_dim])
        if dim % 2 == 0:
            DE[i] = end_val[-1]
        else:
            DE[i] = end_val[-2]
            
        dual[i] = dual0[start_dim]

        if dim == 1: 
            start_dim = end_dim
            continue

        val0shape = [int(np.prod(DS0[:start_dim]))] + list(DS0[start_dim:end_dim]) + [int(np.prod(DS0[end_dim:]))]
        valshape = [val0shape[0], np.prod(val0shape[1:-1]), val0shape[-1]]
        val0 = val0.reshape(val0shape)
        val = val.reshape(valshape)
        if dim == 1:
            for i in range(2**dim):
                slice_shape = [val0shape[0], intervals[i], val0shape[-1]]
                val[:,start_val[i]:end_val[i],:] = val0[:, starts[i][0]:ends[i][0], :].reshape(slice_shape)
        elif dim == 2:
            for i in range(2**dim):
                slice_shape = [val0shape[0], intervals[i], val0shape[-1]]
                val[:,start_val[i]:end_val[i],:] = val0[:, starts[i][0]:ends[i][0], starts[i][1]:ends[i][1], :].reshape(slice_shape)
        elif dim == 3:
            for i in range(2**dim):
                slice_shape = [val0shape[0], intervals[i], val0shape[-1]]
                val[:,start_val[i]:end_val[i],:] = val0[:, starts[i][0]:ends[i][0], starts[i][1]:ends[i][1], starts[i][2]:ends[i][2], :].reshape(slice_shape)
        elif dim == 4:
            for i in range(2**dim):
                slice_shape = [val0shape[0], intervals[i], val0shape[-1]]
                val[:,start_val[i]:end_val[i],:] = val0[:, starts[i][0]:ends[i][0], starts[i][1]:ends[i][1], starts[i][2]:ends[i][2], starts[i][3]:ends[i][3], :].reshape(slice_shape)
        elif dim == 5:
            for i in range(2**dim):
                slice_shape = [val0shape[0], intervals[i], val0shape[-1]]
                val[:,start_val[i]:end_val[i],:] = val0[:, starts[i][0]:ends[i][0], starts[i][1]:ends[i][1], starts[i][2]:ends[i][2], starts[i][3]:ends[i][3], starts[i][4]:ends[i][4], :].reshape(slice_shape)
        else:
            raise ValueError("too big the dimension of shapeT")
        val0 = val
        start_dim = end_dim
    
    return FermiT(DS, DE, dual, val0.reshape(DS))

def _split_axes_once(fermiT, index, subDS, subDE):
    subDS = np.asarray(subDS)
    subDE = np.asarray(subDE)
    DS0 = fermiT.DS
    DE0 = fermiT.DE
    dual0 = fermiT.dual
    val0 = fermiT.val
    dim = subDS.shape[0]
    if dim == 1: return fermiT
    
    starts, ends, start_val, end_val, intervals = _get_group_index_parameters(subDS, subDE)

    valshape = [int(np.prod(DS0[:index]))] + list(subDS) + [int(np.prod(DS0[index+1:]))]
    val = np.zeros(valshape, dtype=complex)
    val0shape = [valshape[0], np.prod(valshape[1:-1]), valshape[-1]]
    val0 = val0.reshape(val0shape)
    
    if dim == 2:
        for i in range(2**dim):
            slice_shape = [valshape[0]] + list(ends[i] - starts[i]) + [valshape[-1]]
            val[:, starts[i][0]:ends[i][0], starts[i][1]:ends[i][1], :] = val0[:,start_val[i]:end_val[i], :].reshape(slice_shape)
    elif dim == 3:
        for i in range(2**dim):
            slice_shape = [valshape[0]] + list(ends[i] - starts[i]) + [valshape[-1]]
            val[:, starts[i][0]:ends[i][0], starts[i][1]:ends[i][1], starts[i][2]:ends[i][2], :] = val0[:,start_val[i]:end_val[i], :].reshape(slice_shape)
    elif dim == 4:
        for i in range(2**dim):
            slice_shape = [valshape[0]] + list(ends[i] - starts[i]) + [valshape[-1]]
            val[:, starts[i][0]:ends[i][0], starts[i][1]:ends[i][1], starts[i][2]:ends[i][2], starts[i][3]:ends[i][3], :] = val0[:,start_val[i]:end_val[i],:].reshape(slice_shape)
    elif dim == 5:
        for i in range(2**dim):
            slice_shape = [valshape[0]] + list(ends[i] - starts[i]) + [valshape[-1]]
            val[:, starts[i][0]:ends[i][0], starts[i][1]:ends[i][1], starts[i][2]:ends[i][2], starts[i][3]:ends[i][3], starts[i][4]:ends[i][4], :] = val0[:,start_val[i]:end_val[i],:].reshape(slice_shape)
    else:
        print("too big the dimension of shapeT")
    
    DS = np.concatenate([DS0[:index], subDS, DS0[index+1:]], axis=0)
    DE = np.concatenate([DE0[:index], subDE, DE0[index+1:]], axis=0)
    dual = list(dual0[:index]) + [dual0[index]]*len(subDS) + list(dual0[index+1:])
    dual = np.asarray(dual)
    val = val.reshape(DS)
    return FermiT(DS, DE, dual, val)

def split_axes(fermiT, shapeT, DS, DE):
    order0 = sum(shapeT, [])
    dims_subDS = [len(shapeT[i]) for i in range(len(shapeT))]
    indexs_subDS = [None] * (len(shapeT) + 1)
    indexs_subDS[0] = 0
    for i in range(len(shapeT)):
        indexs_subDS[i+1] = indexs_subDS[i] + dims_subDS[i]

    for i in range(len(shapeT)):
        subDS = DS[indexs_subDS[i]:indexs_subDS[i+1]]
        subDE = DE[indexs_subDS[i]:indexs_subDS[i+1]]
        fermiT = _split_axes_once(fermiT, indexs_subDS[i], subDS, subDE)
        # for i in range(len(shapeT))[::-1]:
        # fermiT = defgroup_once(fermiT, i, subDS, subDE)
    
    # order = np.zeros(len(order0), dtype=int)
    # for i, j in enumerate(order0): order[j] = i
    # return fermiT.transpose(order)
    return fermiT
