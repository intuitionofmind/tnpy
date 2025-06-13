import sys
sys.path.append("../../")
import numpy as np
from glob import glob
from utils import dir2param

def gather_dim(
    datadir: str, gname: str, mattype: str, 
    pltstep: int, fixvar: dict[str] = {}, grange=None
):
    """
    Return format
    ---------
    Each row: g, d0, d1, d2, ...
    
    Sorted according to the value of g
    """
    cftdata = []
    try:
        if grange is None: grange = []
    except:
        pass
    # determine folders to include
    dirs = glob(datadir + '/*/')
    for d in dirs:
        try:
            param = dir2param(d)
        except:
            continue
        # match fixvar
        if all(item in param.items() for item in fixvar.items()):
            # current parameter value
            g = param[gname]
            if len(grange) == 2 and (g > grange[1] or g < grange[0]):
                continue
            # scaling dimension file
            filename = "{}/{}.txt".format(d, mattype)
            current_g = np.loadtxt(filename)
            cftdata.append(np.insert(current_g[pltstep // 2 - 1, 1::], 0, g))
    cftdata = np.asarray(cftdata)
    # sort by parameter value
    cftdata = cftdata[np.argsort(cftdata[:,0])]
    # sort scaling dimension of each line
    for i in range(cftdata.shape[0]):
        cftdata[i, 1:] = cftdata[i,1:][np.argsort(cftdata[i,1:])]
    return cftdata

def combine_dim(cftdata1: np.ndarray, cftdata2: np.ndarray):
    """Combine two sectors of scaling dimension"""
    # combine two sectors
    allshape = (cftdata1.shape[0], cftdata1.shape[1]*2 - 1)
    cftdata = np.empty(allshape)
    for i in range(allshape[0]):
        assert cftdata1[i,0] == cftdata2[i,0]
        cftdata[i,:] = np.concatenate((cftdata1[i,:], cftdata2[i,1:]))
    # sort by parameter value
    cftdata = cftdata[np.argsort(cftdata[:,0])]
    # sort scaling dimension of each line
    for i in range(cftdata.shape[0]):
        cftdata[i, 1:] = cftdata[i,1:][np.argsort(cftdata[i,1:])]
    return cftdata

def cal_vel(rawdimv: np.ndarray, rawdimh: np.ndarray):
    """Calculate velocity from raw scaling dimension (assumed v = 1)"""
    return np.sqrt(rawdimv[1:] / rawdimh[1:])

def full_vlist(rawdimv: np.ndarray, rawdimh: np.ndarray, nev=20):
    """Calculate velocity from every scaling dimension"""
    return np.asarray([
        np.concatenate((np.array([rawdimh[i,0]]), cal_vel(rawdimv[i,1:], rawdimh[i,1:])[0:nev]))
        for i in range(rawdimh.shape[0])
    ])

def correct_scdim(
    rawdimv: np.ndarray, rawdimh: np.ndarray, 
    usev_num=4, return_v=False
):
    """Calculate velocity from raw scaling dimension and correct them"""
    vlist = np.asarray(list([
        rawdimh[i,0], 
        np.average(cal_vel(rawdimv[i,1:], rawdimh[i,1:])[0:usev_num])
    ] for i in range(rawdimh.shape[0])))
    # correct the scaling dimensions
    for i in range(rawdimh.shape[0]):
        rawdimh[i, 1:] *= vlist[i, 1]
        rawdimv[i, 1:] /= vlist[i, 1]
    if return_v == False:
        return rawdimv, rawdimh
    else:
        return rawdimv, rawdimh, vlist
        