import torch
torch.set_default_dtype(torch.float64)

import math

from tnpy import GTensor
from tnpy.linalg import svd

def gtensor_qr(input_gt: GTensor, group_dims: tuple, qr_dims=None):
    r''' 
    QR decomposition of a GTensor like: T = Q -<- R
    with normal trace between Q and R

    Parameters
    ----------
    input_gtr: GTensor
    group_dims: tuple[tuple],
    qr_dims: tuple[int],

    Returns
    -------
    gt_q: GTensor, the isometric tensor
    gt_r: GTensor, the residual tensor
    '''

    # permute to the new order
    temp_gt = input_gt.permute(group_dims[0]+group_dims[1])
    split = len(group_dims[0])

    # build parity quantum numbers and matrices
    dims = tuple(range(temp_gt.ndim))
    split_dims = dims[:split], dims[split:]
    mat_qns_e = temp_gt.parity_mat_qnums(split_dims, parity=0)
    mat_qns_o = temp_gt.parity_mat_qnums(split_dims, parity=1)
    mat_e = temp_gt.parity_mat(mat_qns_e, split_dims, parity=0)
    mat_o = temp_gt.parity_mat(mat_qns_o, split_dims, parity=1)
    # QR in these two sectors, respectively
    qe, re = torch.linalg.qr(mat_e, mode='reduced')
    qo, ro = torch.linalg.qr(mat_o, mode='reduced')

    # new dual for Q and R
    dual_q, dual_r = temp_gt.dual[:split]+(1,), (0,)+temp_gt.dual[split:]
    # new quantum numbers for Q and R
    qns_qe = mat_qns_e[0], ((0,),)
    qns_qo = mat_qns_o[0], ((1,),)
    qns_re = ((0,),), mat_qns_e[1]
    qns_ro = ((1,),), mat_qns_o[1]
    # new dimensions from QR
    dim_e = min(mat_e.shape[0], mat_e.shape[1])
    dim_o = min(mat_o.shape[0], mat_o.shape[1])
    # new shape for Q and R
    shape_q = temp_gt.shape[:split]+((dim_e, dim_o),)
    shape_r = ((dim_e, dim_o),)+temp_gt.shape[split:]

    # construct new GTensors
    # pay attention to the new group_dims
    dims_q = list(range(len(shape_q)))
    dims_r = list(range(len(shape_r)))
    gt_q = GTensor.construct_from_parity_mats(
            mats=(qe, qo), qns=(qns_qe, qns_qo), dual=dual_q, shape=shape_q,
            group_dims=(tuple(dims_q[:-1]), (dims_q[-1],)))
    gt_r = GTensor.construct_from_parity_mats(
            mats=(re, ro), qns=(qns_re, qns_ro), dual=dual_r, shape=shape_r,
            group_dims=((0,), tuple(dims_r[1:])))

    # permute to the desired order if needed
    if qr_dims is not None:
        dims_q.insert(qr_dims[0], dims_q.pop(-1))
        dims_r.insert(qr_dims[1], dims_r.pop(0))
        gt_q, gt_r = gt_q.permute(dims_q), gt_r.permute(dims_r)

    return gt_q, gt_r

def gtensor_super_qr(input_gt: GTensor, group_dims: tuple, qr_dims=None):
    r''' 
    QR decomposition of a GTensor like: T = Q ->- R
    with super trace between Q and R

    Parameters
    ----------
    input_gt: GTensor
    group_dims: tuple[tuple],
    qr_dims: tuple[int],

    Returns
    -------
    gt_q: GTensor, the isometric tensor
    gt_r: GTensor, the residual tensor
    '''

    # permute to the new order
    temp_gt = input_gt.permute(group_dims[0]+group_dims[1])
    split = len(group_dims[0])

    # build parity quantum numbers and matrices
    dims = tuple(range(temp_gt.ndim))
    split_dims = dims[:split], dims[split:]
    mat_qns_e = temp_gt.parity_mat_qnums(split_dims, parity=0)
    mat_qns_o = temp_gt.parity_mat_qnums(split_dims, parity=1)
    mat_e = temp_gt.parity_mat(mat_qns_e, split_dims, parity=0)
    mat_o = temp_gt.parity_mat(mat_qns_o, split_dims, parity=1)
    # QR in these two sectors, respectively
    qe, re = torch.linalg.qr(mat_e, mode='reduced')
    qo, ro = torch.linalg.qr(mat_o, mode='reduced')

    # new dual for Q and R
    dual_q, dual_r = temp_gt.dual[:split]+(0,), (1,)+temp_gt.dual[split:]
    # new quantum numbers for Q and R
    qns_qe = mat_qns_e[0], ((0,),)
    qns_qo = mat_qns_o[0], ((1,),)
    qns_re = ((0,),), mat_qns_e[1]
    qns_ro = ((1,),), mat_qns_o[1]
    # new dimensions from QR
    dim_e = min(mat_e.shape[0], mat_e.shape[1])
    dim_o = min(mat_o.shape[0], mat_o.shape[1])
    # new shape for Q and R
    shape_q = temp_gt.shape[:split]+((dim_e, dim_o),)
    shape_r = ((dim_e, dim_o),)+temp_gt.shape[split:]

    # construct new GTensors
    # pay attention to the new group_dims
    dims_q = list(range(len(shape_q)))
    dims_r = list(range(len(shape_r)))
    gt_q = GTensor.construct_from_parity_mats(
            mats=(qe, qo), qns=(qns_qe, qns_qo), dual=dual_q, shape=shape_q,
            group_dims=(tuple(dims_q[:-1]), (dims_q[-1],)))
    gt_r = GTensor.construct_from_parity_mats(
            mats=(re, ro), qns=(qns_re, qns_ro), dual=dual_r, shape=shape_r,
            group_dims=((0,), tuple(dims_r[1:])))

    # supertrace sign is assigned to Q
    # unitary property of Q should be redefined
    for q, t in gt_q.blocks().items():
        if 1 == q[-1]:
            gt_q._blocks.update({q:-t})

    # permute to the desired order if needed
    if qr_dims is not None:
        dims_q.insert(qr_dims[0], dims_q.pop(-1))
        dims_r.insert(qr_dims[1], dims_r.pop(0))
        gt_q, gt_r = gt_q.permute(dims_q), gt_r.permute(dims_r)

    return gt_q, gt_r

def gtensor_svd(input_gt: GTensor, group_dims: tuple, svd_dims=None, cut_off=None):
    r'''
    SVD a GTensor in the direction: T = U -<- S -<- V
    with normal trace between them

    Parameters
    ----------
    input_gt: GTensor,
    group_dims: tuple[tuple], two tuple[int] consist of bonds of 'U' and 'V'
    svd_dims: tuple[int], optional, the SVD dims
    cut_off: tuple[int], optional, SVD dim truncation for even and odd sectors, respectively
    '''

    # permute to the new order
    temp_gt = input_gt.permute(group_dims[0]+group_dims[1])
    split = len(group_dims[0])

    # new duals for new tensors: U -<- S -<- V
    dual_u, dual_s, dual_v = temp_gt.dual[:split]+(1,), (0, 1), (0,)+temp_gt.dual[split:]
    # build parity quantum numbers and matrices
    dims = tuple(range(temp_gt.ndim))
    split_dims = dims[:split], dims[split:]
    mat_qns_e = temp_gt.parity_mat_qnums(split_dims, parity=0)
    mat_qns_o = temp_gt.parity_mat_qnums(split_dims, parity=1)
    mat_e = temp_gt.parity_mat(mat_qns_e, split_dims, parity=0)
    mat_o = temp_gt.parity_mat(mat_qns_o, split_dims, parity=1)

    # SVD in these two sectors, respectively
    ue, se, ve = svd(mat_e, full_matrices=False)
    uo, so, vo = svd(mat_o, full_matrices=False)
    se, so = se.diag(), so.diag()
    # new dims from SVD
    svd_dims = min(mat_e.shape), min(mat_o.shape)

    # new quanum numbers for U, S, V
    mat_qns_ue = mat_qns_e[0], ((0,),)
    mat_qns_uo = mat_qns_o[0], ((1,),)
    mat_qns_ve = ((0,),), mat_qns_e[1]
    mat_qns_vo = ((1,),), mat_qns_o[1]
    mat_qns_se = ((0,),), ((0,),)
    mat_qns_so = ((1,),), ((1,),)

    if cut_off is not None:
        svd_dims = min(cut_off[0], svd_dims[0]), min(cut_off[1], svd_dims[1])

    # block shape for new tensors
    shape_u = temp_tensor.block_shape[:split]+(dim_from_svd,)
    shape_s = (dim_from_svd, dim_from_svd)
    shape_v = (dim_from_svd,)+temp_tensor.block_shape[split:]

    # restore GTensor from parity matrices
    # pay attention to the divide
    mats_u = (u_e[:, :dim_from_svd], u_o[:, :dim_from_svd])
    mats_s = (s_e[:dim_from_svd, :dim_from_svd], s_o[:dim_from_svd, :dim_from_svd])
    mats_v = (v_e[:dim_from_svd, :], v_o[:dim_from_svd, :])
    gt_u = GTensor.restore_from_parity_matrices(
            mats=mats_u,
            qnums=(qnums_u_e, qnums_u_o),
            dual=dual_u, shape=shape_u, divide=-1)
    gt_s = GTensor.restore_from_parity_matrices(
            mats=mats_s,
            qnums=(qnums_s_e, qnums_s_o),
            dual=dual_s, shape=shape_s, divide=1)
    gt_v = GTensor.restore_from_parity_matrices(
            mats=mats_v,
            qnums=(qnums_v_e, qnums_v_o),
            dual=dual_v, shape=shape_v, divide=1)

    if svd_dims is not None:
        # permute to the desired order
        dims_u, dims_v = list(range(gt_u.ndim)), list(range(gt_v.ndim))
        dims_u.insert(svd_dims[0], dims_u.pop(-1))
        dims_v.insert(svd_dims[1], dims_v.pop(0))
        gt_u, gt_v = gpermute(gt_u, dims_u), gpermute(gt_v, dims_v)

    return gt_u, gt_s, gt_v

def gtensor_super_svd(input_tensor: GTensor, group_dims: tuple, svd_dims=None, cut_off=None, full_matrices=None):
    r'''
    SVD a GTensor T = USV in another decomposed direction
    supertrace contraction $|\alpha\rangle\langle\beta|$
    only valid the direction: U->-S->-V

    Parameters
    ----------
    input_tensor: GTensor,
    group_dims: tuple[tuple], two tuple[int] consist of bonds of 'U' and 'V'
    svd_dims: tuple[int], the SVD dims
    cut_off: int, optional, SVD truncation
    '''

    flag = False
    if full_matrices is not None:
        flag = full_matrices

    dims = group_dims[0]+group_dims[1]
    # permute to the new order
    temp_tensor = gpermute(input_tensor, dims)
    split = len(group_dims[0])

    # new duals for new tensors: U ->- S ->- V
    dual_u, dual_s, dual_v = temp_tensor.dual[:split]+(0,), (1, 0), (1,)+temp_tensor.dual[split:]
    # build the divided qunumber numbers based on 'split'
    qnums_e, qnums_o = temp_tensor.parity_matrix_qnums(divide=split)
    # fuse to even and odd sector matrices and obtain quantum numbers
    mat_e, mat_o = temp_tensor.parity_matrices(qns=(qnums_e, qnums_o), divide=split)
    # SVD in these two sectors
    u_e, s_e, v_e = svd(mat_e, full_matrices=flag)
    u_o, s_o, v_o = svd(mat_o, full_matrices=flag)
    s_e, s_o = torch.diag(s_e), torch.diag(s_o)
    # new quantum numbers in these two sectors
    qnums_u_e = qnums_e[0], ((0,),)
    qnums_s_e = ((0,),), ((0,),)
    qnums_v_e = ((0,),), qnums_e[1]
    qnums_u_o = qnums_o[0], ((1,),)
    qnums_s_o = ((1,),), ((1,),)
    qnums_v_o = ((1,),), qnums_o[1]
    # block shapes for new tensors
    dim_from_svd = min(mat_e.shape[0], mat_e.shape[1])
    # if SVD truncation is set
    if (cut_off is not None) and (cut_off < dim_from_svd):
        dim_from_svd = cut_off
    # block shape for new tensors
    shape_u = temp_tensor.block_shape[:split]+(dim_from_svd,)
    shape_s = (dim_from_svd, dim_from_svd)
    shape_v = (dim_from_svd,)+temp_tensor.block_shape[split:]
    # restore GTensor from parity matrices
    # pay attention to the divide
    mats_u = (u_e[:, :dim_from_svd], u_o[:, :dim_from_svd])
    mats_s = (s_e[:dim_from_svd, :dim_from_svd], s_o[:dim_from_svd, :dim_from_svd])
    mats_v = (v_e[:dim_from_svd, :], v_o[:dim_from_svd, :])
    gt_u = GTensor.restore_from_parity_matrices(
            mats=mats_u,
            qnums=(qnums_u_e, qnums_u_o),
            dual=dual_u, shape=shape_u, divide=-1)
    gt_s = GTensor.restore_from_parity_matrices(
            mats=mats_s,
            qnums=(qnums_s_e, qnums_s_o),
            dual=dual_s, shape=shape_s, divide=1)
    gt_v = GTensor.restore_from_parity_matrices(
            mats=mats_v,
            qnums=(qnums_v_e, qnums_v_o),
            dual=dual_v, shape=shape_v, divide=1)

    # pay attention to the possible sign arised in supertrace
    # the supertrace signs are assigned to U and V
    u_blocks = {}
    for q, t in gt_u._blocks.items():
        if 0 == q[-1]:
            u_blocks[q] = t
        else:
            u_blocks[q] = -t
    v_blocks = {}
    for q, t in gt_v._blocks.items():
        if 0 == q[0]:
            v_blocks[q] = t
        else:
            v_blocks[q] = -t
    gt_u = GTensor(dual=gt_u.dual, blocks=u_blocks, info=gt_u.info)
    gt_v = GTensor(dual=gt_v.dual, blocks=v_blocks, info=gt_v.info)

    # permute if needed
    if svd_dims is not None:
        # permute to the desired order
        dims_u, dims_v = list(range(gt_u.ndim)), list(range(gt_v.ndim))
        dims_u.insert(svd_dims[0], dims_u.pop(-1))
        dims_v.insert(svd_dims[1], dims_v.pop(0))
        gt_u, gt_v = gpermute(gt_u, dims_u), gpermute(gt_v, dims_v)

    return gt_u, gt_s, gt_v

def gtensor_factorize(gt: GTensor, group_dims: tuple, pos_dims: tuple) -> tuple:
    r'''
    factorize a GTensor into two GTensors
    T = A-<-B

    Parameters
    ----------
    gt: GTensor,
    group_dims: tuple[tuple]
    pos_dims: tuple[int]
    '''

    u, s, v = gtensor_svd(gt, group_dims)
    s_e = s.blocks()[(0, 0)].sqrt()
    s_o = s.blocks()[(1, 1)].sqrt()
    s_blocks = {(0, 0): s_e, (1, 1): s_o}
    s_sqrt = GTensor(dual=s.dual, blocks=s_blocks)

    oe_str = ''
    for i in range(gt.ndim+4):
        oe_str += oe.get_symbol(i)
    u_str, s_str, v_str = oe_str[:u.ndim], oe_str[u.ndim:u.ndim+s.ndim], oe_str[u.ndim+s.ndim:]
    u_str = u_str.replace(u_str[-1], s_str[0])
    v_str = v_str.replace(v_str[0], s_str[1])
    us, sv = gcontract(u_str+','+s_str, u, s_sqrt), gcontract(s_str+','+v_str, s_sqrt, v)
    
    # permute if needed
    if pos_dims is not None:
        # permute to the desired order
        dims_u, dims_v = list(range(us.ndim)), list(range(sv.ndim))
        dims_u.insert(pos_dims[0], dims_u.pop(-1))
        dims_v.insert(pos_dims[1], dims_v.pop(0))
        us, sv = gpermute(us, dims_u), gpermute(sv, dims_v)

    return us, sv

def gtensor_super_factorize(gt: GTensor, group_dims: tuple, pos_dims: tuple) -> tuple:
    r'''
    factorize a GTensor into two GTensors
    T = A->-B

    Parameters
    ----------
    gt: GTensor,
    group_dims: tuple[tuple]
    pos_dims: tuple[int]
    '''

    u, s, v = gtensor_super_svd(gt, group_dims)
    s_e = s.blocks()[(0, 0)].sqrt()
    s_o = s.blocks()[(1, 1)].sqrt()
    # an extra supertrace sign here
    ls_blocks = {(0, 0): s_e, (1, 1): s_o}
    rs_blocks = {(0, 0): s_e, (1, 1): -1.0*s_o}
    ls_sqrt = GTensor(dual=s.dual, blocks=ls_blocks)
    rs_sqrt = GTensor(dual=s.dual, blocks=rs_blocks)

    oe_str = ''
    for i in range(gt.ndim+4):
        oe_str += oe.get_symbol(i)
    u_str, s_str, v_str = oe_str[:u.ndim], oe_str[u.ndim:u.ndim+s.ndim], oe_str[u.ndim+s.ndim:]
    u_str = u_str.replace(u_str[-1], s_str[0])
    v_str = v_str.replace(v_str[0], s_str[1])
    us, sv = gcontract(u_str+','+s_str, u, ls_sqrt), gcontract(s_str+','+v_str, rs_sqrt, v)
    
    # permute if needed
    if pos_dims is not None:
        # permute to the desired order
        dims_u, dims_v = list(range(us.ndim)), list(range(sv.ndim))
        dims_u.insert(pos_dims[0], dims_u.pop(-1))
        dims_v.insert(pos_dims[1], dims_v.pop(0))
        us, sv = gpermute(us, dims_u), gpermute(sv, dims_v)

    return us, sv

def gpinv(gt: GTensor) -> GTensor:
    r'''
    compute the pseudoinverse of a GTensor 'gt' satisfying:
    inv_gt*gt = 1 and gt*inv_gt = 1

    Returns
    -------
    inv_gt: GTensor, same dual as the input
    '''

    assert isinstance(gt, GTensor) and 2 == gt.ndim, 'not a 2-dimensional GTensor(matrix)'

    flag = False
    if (1, 0) == gt.dual:
        temp_gt = gpermute(gt, (1, 0))
        flag = True
    else:
        temp_gt = gt
    # find pinvs block by block
    inv_blocks = {k:torch.linalg.pinv(v) for k, v in temp_gt.blocks().items()}
    inv_gt = GTensor(temp_gt.dual, blocks=inv_blocks)
    if flag:
        inv_gt = gpermute(inv_gt, (1, 0))

    return inv_gt

def ginv(gt: GTensor) -> GTensor:
    r'''
    compute the pseudoinverse of a GTensor 'gt' satisfying:
    inv_gt*gt = 1 and gt*inv_gt = 1

    Returns
    -------
    inv_gt: GTensor, same dual as the input
    '''

    assert isinstance(gt, GTensor) and 2 == gt.ndim, 'not a 2-dimensional GTensor(matrix)'

    flag = False
    if (1, 0) == gt.dual:
        temp_gt = gpermute(gt, (1, 0))
        flag = True
    else:
        temp_gt = gt
    # find pinvs block by block
    inv_blocks = {k:torch.linalg.inv(v) for k, v in temp_gt.blocks().items()}
    inv_gt = GTensor(temp_gt.dual, blocks=inv_blocks)
    if flag:
        inv_gt = gpermute(inv_gt, (1, 0))

    return inv_gt
