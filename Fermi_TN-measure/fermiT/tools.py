
def get_slice(DS: list[int], DE: list[int], gIdx: tuple[int, ...]):
    """Get slice in the FermiT corresponding to the block
    labelled by the fermion index `gIdx`"""
    idxStart = [(0 if g == 0 else de) for g, de in zip(gIdx, DE)]
    idxEnd = [(de if g == 0 else ds) for g, de, ds in zip(gIdx, DE, DS)]
    return tuple(map(slice, idxStart, idxEnd))


def get_idx(gIdx: int, nIdx: int, DS: int, DE: int):
    """
    Convert index (gIdx, nIdx) (GTensor format) to idx (FermiT format) 
    """
    assert gIdx in (0, 1)
    DO = DS - DE
    if gIdx == 0: 
        assert 0 <= nIdx < DE
        return nIdx
    else:
        assert 0 <= nIdx < DO
        return DE + nIdx
    

def gen_gidx(ndim, parity=0):
    """Grassmann index generator (of given parity)"""
    if not parity % 1 == 0:
        raise ValueError('`parity` must be an integer')
    parity %= 2
    if ndim == 0:
        if parity != 0:
            raise ValueError('scalar must be parity even')
        yield tuple()
        return
    for i in range(2**ndim):
        binary = bin(i)[2:].rjust(ndim, '0')
        if binary.count('1') % 2 == parity:
            yield tuple(int(j) for j in binary)


def regularize_axes(axes: list[int], ndim: int) -> list[int]:
    """Deal with negative axes id"""
    axesRegularized = []
    # deal with negative numbers in axes
    for ax in axes:
        assert isinstance(ax, int)
        if ax < 0:
            axesRegularized.append(ndim + ax)
        else:
            axesRegularized.append(ax)
    return axesRegularized


def process_flip_axes(axes):
    if len(axes) == 0:
        axReg = ((), ())
    elif len(axes) == 2:
        try:
            axReg = tuple(tuple(ax) for ax in axes)
        except TypeError:
            axReg = ((axes[0],), (axes[1],))
    else:
        raise ValueError("`axes` is in wrong format")
    assert len(axReg[0]) == len(axReg[1])
    return axReg
