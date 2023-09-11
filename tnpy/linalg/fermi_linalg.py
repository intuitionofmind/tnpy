import torch
torch.set_default_dtype(torch.float64)

import math

from tnpy import GTensor
from tnpy.linalg import svd

def gtqr(input_gt: GTensor, group_dims: tuple, qr_dims=None) -> tuple:
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

def super_gtqr(input_gt: GTensor, group_dims: tuple, qr_dims=None) -> tuple:
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

def gtsvd(input_gt: GTensor, group_dims: tuple, svd_dims=None, cut_off=None):
    r'''
    SVD a GTensor in the direction: T = U -<- S -<- V
    with normal trace between them

    Parameters
    ----------
    input_gt: GTensor,
    group_dims: tuple[tuple], two tuple[int] consist of bonds of 'U' and 'V'
    svd_dims: tuple[int], optional, the SVD dims
    cut_off: optional,
        if: tuple[int], SVD truncation for even and odd sectors, respectively
        elif: int, SVD truncation overall combined for even and odd sectors
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

    # SVD in these two sectors, respectively
    ue, se, ve = svd(mat_e, full_matrices=False)
    uo, so, vo = svd(mat_o, full_matrices=False)

    # new duals for U -<- S -<- V
    dual_u, dual_s, dual_v = temp_gt.dual[:split]+(1,), (0, 1), (0,)+temp_gt.dual[split:]
    # new quanum numbers for U, S, V
    qns_ue = mat_qns_e[0], ((0,),)
    qns_uo = mat_qns_o[0], ((1,),)
    qns_ve = ((0,),), mat_qns_e[1]
    qns_vo = ((1,),), mat_qns_o[1]
    qns_se = ((0,),), ((0,),)
    qns_so = ((1,),), ((1,),)

    # shape of new dims from SVD
    svd_shape = min(mat_e.shape), min(mat_o.shape)
    if cut_off is not None:
        # separate truncation in even and odd sectors, respectively
        if isinstance(cut_off, tuple):
            svd_shape = min(cut_off[0], svd_shape[0]), min(cut_off[1], svd_shape[1])
        # overall truncation by putting even and odd sectors together
        elif isinstance(cut_off, int):
            s = torch.cat((se, so), dim=0)
            ss = torch.sort(s, descending=True, stable=True)
            remaining_indices = ss.indices[:cut_off]
            ne, no = 0, 0
            for d in remaining_indices:
                # count odd sector
                if d.item() >= len(se):
                    no += 1
                # count even sector
                else:
                    ne += 1
            svd_shape = ne, no
        else:
            raise TypeError('input cut_off type is not valid')

    # truncate matrices and spectrum
    mats_u = ue[:, :svd_shape[0]], uo[:, :svd_shape[1]]
    mats_v = ve[:svd_shape[0], :], vo[:svd_shape[1], :]
    mats_s = se[:svd_shape[0]].diag(), so[:svd_shape[1]].diag()

    # block shape for new tensors
    shape_u = temp_gt.shape[:split]+((svd_shape[0], svd_shape[1]),)
    shape_v = ((svd_shape[0], svd_shape[1]),)+temp_gt.shape[split:]
    shape_s = ((svd_shape[0], svd_shape[1]),)+((svd_shape[0], svd_shape[1]),)

    # restore GTensors from parity matrices
    # pay attention to the group_dims
    dims_u = list(range(len(shape_u)))
    dims_v = list(range(len(shape_v)))
    gt_u = GTensor.construct_from_parity_mats(
        mats=mats_u, qns=(qns_ue, qns_uo), dual=dual_u, shape=shape_u,
        group_dims=(tuple(dims_u[:-1]), (dims_u[-1],)))
    gt_v = GTensor.construct_from_parity_mats(
        mats=mats_v, qns=(qns_ve, qns_vo), dual=dual_v, shape=shape_v,
        group_dims=((0,), tuple(dims_v[1:])))
    gt_s = GTensor.construct_from_parity_mats(
        mats=mats_s, qns=(qns_se, qns_so), dual=dual_s, shape=shape_s,
        group_dims=((0,), (1,)), cflag=input_gt.cflag)

    if svd_dims is not None:
        # permute to the desired order
        dims_u.insert(svd_dims[0], dims_u.pop(-1))
        dims_v.insert(svd_dims[1], dims_v.pop(0))
        gt_u, gt_v = gt_u.permute(dims_u), gt_v.permute(dims_v)

    return gt_u, gt_s, gt_v

def super_gtsvd(input_gt: GTensor, group_dims: tuple, svd_dims=None, cut_off=None):
    r'''
    SVD a GTensor in the direction: T = U ->- S ->- V
    with super trace between them

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

    # new duals for U ->- S ->- V
    dual_u, dual_s, dual_v = temp_gt.dual[:split]+(0,), (1, 0), (1,)+temp_gt.dual[split:]
    # new quanum numbers for U, S, V
    qns_ue = mat_qns_e[0], ((0,),)
    qns_uo = mat_qns_o[0], ((1,),)
    qns_ve = ((0,),), mat_qns_e[1]
    qns_vo = ((1,),), mat_qns_o[1]
    qns_se = ((0,),), ((0,),)
    qns_so = ((1,),), ((1,),)

    # shape of new dims from SVD
    svd_shape = min(mat_e.shape), min(mat_o.shape)
    if cut_off is not None:
        # separate truncation in even and odd sectors, respectively
        if isinstance(cut_off, tuple):
            svd_shape = min(cut_off[0], svd_shape[0]), min(cut_off[1], svd_shape[1])
        # overall truncation by putting even and odd sectors together
        elif isinstance(cut_off, int):
            s = torch.cat((se, so), dim=0)
            ss = torch.sort(s, descending=True, stable=True)
            remaining_indices = ss.indices[:cut_off]
            ne, no = 0, 0
            for d in remaining_indices:
                # count odd sector
                if d.item() >= len(se):
                    no += 1
                # count even sector
                else:
                    ne += 1
            svd_shape = ne, no
        else:
            raise TypeError('input cut_off type is not valid')

    # truncate matrices and spectrum
    mats_u = ue[:, :svd_shape[0]], uo[:, :svd_shape[1]]
    mats_v = ve[:svd_shape[0], :], vo[:svd_shape[1], :]
    mats_s = se[:svd_shape[0]].diag(), so[:svd_shape[1]].diag()

    # block shape for new tensors
    shape_u = temp_gt.shape[:split]+((svd_shape[0], svd_shape[1]),)
    shape_v = ((svd_shape[0], svd_shape[1]),)+temp_gt.shape[split:]
    shape_s = ((svd_shape[0], svd_shape[1]),)+((svd_shape[0], svd_shape[1]),)

    # restore GTensors from parity matrices
    # pay attention to the group_dims
    dims_u = list(range(len(shape_u)))
    dims_v = list(range(len(shape_v)))
    gt_u = GTensor.construct_from_parity_mats(
        mats=mats_u, qns=(qns_ue, qns_uo), dual=dual_u, shape=shape_u,
        group_dims=(tuple(dims_u[:-1]), (dims_u[-1],)))
    gt_v = GTensor.construct_from_parity_mats(
        mats=mats_v, qns=(qns_ve, qns_vo), dual=dual_v, shape=shape_v,
        group_dims=((0,), tuple(dims_v[1:])))
    gt_s = GTensor.construct_from_parity_mats(
        mats=mats_s, qns=(qns_se, qns_so), dual=dual_s, shape=shape_s,
        group_dims=((0,), (1,)), cflag=input_gt.cflag)

    # supertrace signs are assigned to U and V
    # unitary property of U, V should be redefined
    for q, t in gt_u.blocks().items():
        if 1 == q[-1]:
            gt_u._blocks.update({q:-t})
    for q, t in gt_v.blocks().items():
        if 1 == q[0]:
            gt_v._blocks.update({q:-t})

    if svd_dims is not None:
        # permute to the desired order
        dims_u.insert(svd_dims[0], dims_u.pop(-1))
        dims_v.insert(svd_dims[1], dims_v.pop(0))
        gt_u, gt_v = gt_u.permute(dims_u), gt_v.permute(dims_v)

    return gt_u, gt_s, gt_v

def gpinv(gt: GTensor) -> GTensor:
    r'''
    compute the pseudoinverse of a GTensor satisfying: inv_gt*gt = 1 and gt*inv_gt = 1

    Returns
    -------
    inv_gt: GTensor, same dual as the input
    '''

    assert 2 == gt.ndim, 'not a 2-dimensional GTensor (matrix)'

    flag = False
    if (1, 0) == gt.dual:
        temp_gt = gt.permute(dims=(1, 0))
        flag = True
    else:
        temp_gt = gt

    # find pinvs block by block
    inv_blocks = {k:torch.linalg.pinv(v) for k, v in temp_gt.blocks().items()}
    inv_gt = GTensor(temp_gt.dual, shape=gt.shape[::-1], blocks=inv_blocks, info=gt.info)

    if flag:
        inv_gt = inv_gt.permute(dims=(1, 0))

    return inv_gt

def ginv(gt: GTensor) -> GTensor:
    r'''
    compute the inverse of a GTensor satisfying: inv_gt*gt = 1 and gt*inv_gt = 1

    Returns
    -------
    inv_gt: GTensor, same dual as the input
    '''

    assert 2 == gt.ndim, 'not a 2-dimensional GTensor (matrix)'

    flag = False
    if (1, 0) == gt.dual:
        temp_gt = gt.permute(dims=(1, 0))
        flag = True
    else:
        temp_gt = gt

    # find pinvs block by block
    inv_blocks = {k:torch.linalg.inv(v) for k, v in temp_gt.blocks().items()}
    inv_gt = GTensor(temp_gt.dual, shape=gt.shape[::-1], blocks=inv_blocks, info=gt.info)

    if flag:
        inv_gt = inv_gt.permute(dims=(1, 0))

    return inv_gt
