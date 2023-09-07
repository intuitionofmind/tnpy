import math
import torch
torch.set_default_dtype(torch.float64)
import opt_einsum as oe
import scipy

from tnpy import GTensor, gcontract, Z2gTensor
# import tnpy as tp

# ---------------------
# functions for Tensor
# ---------------------

def svd(mat, full_matrices=None):
    r'''
    robust SVD
    also refer to: https://tenpy.github.io/reference/tenpy.linalg.svd_robust.html
    '''

    flag = False 
    if full_matrices is not None:
        flag = full_matrices

    try:
        u, s, v = torch.linalg.svd(mat, full_matrices=flag)
    except torch._C._LinAlgError:
        u, s, v = scipy.linalg.svd(mat, full_matrices=flag, overwrite_a=True, check_finite=False, lapack_driver='gesvd')
        u, s, v = torch.tensor(u), torch.tensor(s), torch.tensor(v)

    return u, s, v

def tensor_svd(input_tensor, group_dims: tuple, svd_dims=None, cut_off=None) -> tuple:
    r'''
    SVD a tensor T to T = ASB

    Parameters
    ----------
    input_tensor: Tensor
    group_dims: tuple[tuple], tuple[0] bonds going to A; tuple[1] bonds going to B
    svd_dims: tuple[int], positions of new SVD bonds in A and B
            if svd_dims[0] > A.ndim, then it will be place at the end
            similar to B
    cut_off: int, optional, if a trunction on singular value spectrum is required

    Returns
    -------
    u_tensor: Tensor, isometric
    s: 1d Tensor, singular values
    v_tensor: Tensor, isometric

    Examples
    --------
    #    b         d            e
    #    |         |            |
    # a--*--d = c--*--P,P--Q,Q--*-b
    #   / \        |           
    #  d   c       a

    T_{abcde} -> \sum_{PQ} A_{cdPa} S_{PQ} B_{ebQ}

    Indeed we have three steps to accomplish this:
    step #0: T_{abcde} -> T_{cdaeb}
    step #1: T_{cdaeb} -> \sum_{PQ} A_{cdaP} S_{PQ} B_{Qeb}
    step #2: A_{cdaP} -> A_{cdPa}; B_{Qeb} -> B_{ebQ}

    Jupyter notebook:

    t = torch.rand(4, 8, 16, 32, 16)
    %timeit -r 5 -n 100 a, s, b = tensor_svd(t, group_dims=((2, 3, 0), (4, 1)), svd_dims=(2, 2))
    print(a.shape, s.shape, b.shape)
    res = pytn.contract('abcd,cg,efg->dfabe', a, s.diag(), b)
    print('Test tensor SVD:', torch.linalg.norm(t-res))
    '''

    # permute to the operation order
    new_tensor = torch.permute(input_tensor, group_dims[0]+group_dims[1])
    new_shape = new_tensor.shape
    # divide
    divide = len(group_dims[0])
    # reshape to a matrix
    # Python>=3.8
    mat_shape = math.prod(new_shape[:divide]), math.prod(new_shape[divide:])
    mat = torch.reshape(new_tensor, mat_shape)
    # SVD
    u, s, v = svd(mat)
    svd_dim = min(mat_shape)
    # if SVD truncation is needed
    if (cut_off is not None) and (cut_off < svd_dim):
        svd_dim = cut_off
    # new shapes appended with SVD indices
    # column vectors of U and rows of V are orthogonal
    u_shape, v_shape = new_shape[:divide]+(svd_dim,), (svd_dim,)+new_shape[divide:]
    u_tensor, v_tensor = torch.reshape(u[:, :svd_dim], u_shape), torch.reshape(v[:svd_dim, :], v_shape)

    if svd_dims is not None:
        u_dims, v_dims = list(range(u_tensor.ndim)), list(range(v_tensor.ndim))
        u_dims.insert(svd_dims[0], u_dims.pop(-1))
        v_dims.insert(svd_dims[1], v_dims.pop(0))
        u_tensor = torch.permute(u_tensor, u_dims)
        v_tensor = torch.permute(v_tensor, v_dims)

    return u_tensor, s[:svd_dim], v_tensor

def tensor_qr(input_tensor, group_dims: tuple, qr_dims=None) -> tuple:
    r''' 
    QR decomposition of a tensor
    T = QR
    Q is the isometric tensor, R is the residual tensor

    Parameters
    ----------
    input_tensor: Tensor
    group_dims: tuple[tuple], tuple[0]: bonds of Q; tuple[1]: bonds of R
    qr_dims: tuple[int], optional, where the new QR bonds in Q and R
        Defalut: put two new QR bonds at the tail of Q and head of R, respectively

    Returns
    -------
    q_tensor: Tensor, isometric tensor
    r_tensor: Tensor, residual tensor
    '''

    # permute to the operation order
    new_tensor = torch.permute(input_tensor, group_dims[0]+group_dims[1])
    new_shape = new_tensor.shape
    divide = len(group_dims[0])
    # Python>=3.8
    mat_shape = math.prod(new_shape[:divide]), math.prod(new_shape[divide:])
    mat = torch.reshape(new_tensor, mat_shape)
    q, r = torch.linalg.qr(mat, mode='reduced')
    qr_dim = min(mat_shape)
    q_shape, r_shape = new_shape[:divide]+(qr_dim,), (qr_dim,)+new_shape[divide:]
    q_tensor, r_tensor = torch.reshape(q, q_shape), torch.reshape(r, r_shape)

    # put the qr_dims in the desired order
    if qr_dims is not None:
        q_dims, r_dims = list(range(q_tensor.ndim)), list(range(r_tensor.ndim))
        q_dims.insert(qr_dims[0], q_dims.pop(-1))
        r_dims.insert(qr_dims[1], r_dims.pop(0))
        q_tensor, r_tensor = torch.permute(q_tensor, q_dims), torch.permute(r_tensor, r_dims)

    return q_tensor, r_tensor

# ---------------------
# functions for GTensor
# ---------------------

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

    dims = group_dims[0]+group_dims[1]
    # permute to the new order
    temp_gt = input_gt.permute(dims)
    cut = len(group_dims[0])

    # new dual for Q and R
    dual_q, dual_r = temp_gt.dual[:cut]+(1,), (0,)+temp_gt.dual[cut:]
    # build parity quantum numbers and matrices
    dims = tuple(range(temp_gt.ndim))
    gdims = dims[:cut], dims[cut:]
    mat_qns_e = temp_gt.parity_mat_qnums(gdims, parity=0)
    mat_qns_o = temp_gt.parity_mat_qnums(gdims, parity=1)
    mat_e = temp_gt.parity_mat(mat_qns_e, gdims, parity=0)
    mat_o = temp_gt.parity_mat(mat_qns_o, gdims, parity=1)
    # QR in these two sectors, respectively
    qe, re = torch.linalg.qr(mat_e, mode='reduced')
    qo, ro = torch.linalg.qr(mat_o, mode='reduced')
    # new quantum numbers for Q and R
    # even
    mat_qns_qe = mat_qns_e[0], ((0,),)
    mat_qns_re = ((0,),), mat_qns_e[1]
    # odd
    mat_qns_qo = mat_qns_o[0], ((1,),)
    mat_qns_ro = ((1,),), mat_qns_o[1]
    # new dimensions from QR
    dim_e = min(mat_e.shape[0], mat_e.shape[1])
    dim_o = min(mat_o.shape[0], mat_o.shape[1])
    # new shape for Q and R
    shape_q = temp_gt.shape[:cut]+((dim_e, dim_o),)
    shape_r = ((dim_e, dim_o),)+temp_gt.shape[cut:]

    # restore new GTensors
    # pay attention to the new group_dims
    dims_q = list(range(len(shape_q)))
    dims_r = list(range(len(shape_r)))
    gt_q = GTensor.construct_from_parity_mats(
            mats=(qe, qo), qns=(mat_qns_qe, mat_qns_qo), dual=dual_q, shape=shape_q,
            group_dims=(tuple(dims_q[:-1]), (dims_q[-1],)))
    gt_r = GTensor.construct_from_parity_mats(
            mats=(re, ro), qns=(mat_qns_re, mat_qns_ro), dual=dual_r, shape=shape_r,
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

    dims = group_dims[0]+group_dims[1]
    # permute to the new order
    temp_gt = input_gt.permute(dims)
    cut = len(group_dims[0])

    # new dual for Q and R
    dual_q, dual_r = temp_gt.dual[:cut]+(0,), (1,)+temp_gt.dual[cut:]
    # build parity quantum numbers and matrices
    dims = tuple(range(temp_gt.ndim))
    gdims = dims[:cut], dims[cut:]
    mat_qns_e = temp_gt.parity_mat_qnums(gdims, parity=0)
    mat_qns_o = temp_gt.parity_mat_qnums(gdims, parity=1)
    mat_e = temp_gt.parity_mat(mat_qns_e, gdims, parity=0)
    mat_o = temp_gt.parity_mat(mat_qns_o, gdims, parity=1)
    # QR in these two sectors, respectively
    qe, re = torch.linalg.qr(mat_e, mode='reduced')
    qo, ro = torch.linalg.qr(mat_o, mode='reduced')
    # new quantum numbers for Q and R
    # even
    mat_qns_qe = mat_qns_e[0], ((0,),)
    mat_qns_re = ((0,),), mat_qns_e[1]
    # odd
    mat_qns_qo = mat_qns_o[0], ((1,),)
    mat_qns_ro = ((1,),), mat_qns_o[1]
    # new dimensions from QR
    dim_e = min(mat_e.shape[0], mat_e.shape[1])
    dim_o = min(mat_o.shape[0], mat_o.shape[1])
    # new shape for Q and R
    shape_q = temp_gt.shape[:cut]+((dim_e, dim_o),)
    shape_r = ((dim_e, dim_o),)+temp_gt.shape[cut:]

    # restore new GTensors
    # pay attention to the new group_dims
    dims_q = list(range(len(shape_q)))
    dims_r = list(range(len(shape_r)))
    gt_q = GTensor.construct_from_parity_mats(
            mats=(qe, qo), qns=(mat_qns_qe, mat_qns_qo), dual=dual_q, shape=shape_q,
            group_dims=(tuple(dims_q[:-1]), (dims_q[-1],)))
    gt_r = GTensor.construct_from_parity_mats(
            mats=(re, ro), qns=(mat_qns_re, mat_qns_ro), dual=dual_r, shape=shape_r,
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

def gtensor_svd(input_tensor: GTensor, group_dims: tuple, svd_dims=None, cut_off=None, full_matrices=None):
    r'''
    SVD a GTensor in the direction: T = U-<-S-<-V

    Parameters
    ----------
    input_tensor: GTensor,
    group_dims: tuple[tuple], two tuple[int] consist of bonds of 'U' and 'V'
    svd_dims: tuple[int], optional, the SVD dims
    cut_off: int, optional, SVD truncation
    '''

    flag = False
    if full_matrices is not None:
        flag = full_matrices

    dims = group_dims[0]+group_dims[1]
    # permute to the new order
    temp_tensor = gpermute(input_tensor, dims)
    split = len(group_dims[0])

    # new duals for new tensors: U <-- S <-- V
    dual_u, dual_s, dual_v = temp_tensor.dual[:split]+(1,), (0, 1), (0,)+temp_tensor.dual[split:]
    # build the divided qunumber numbers based on 's'
    qnums_e, qnums_o = temp_tensor.parity_matrix_qnums(divide=split)
    # fuse to even and odd sector matrices and obtain quantum numbers
    mat_e, mat_o = temp_tensor.parity_matrices(qns=(qnums_e, qnums_o), divide=split)

    # SVD in these two sectors, respectively
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
