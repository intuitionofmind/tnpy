import re
from .plottools import *
from matplotlib.axes import Axes
from numpy import ndarray
import scipy.interpolate as spip
from scipy.stats import linregress

def scale_D(
    datas: dict[int, ndarray],
    x2s: ndarray, xlim: tuple[float, float] | None = None,
    mode="linear", minr2=0.5
):
    """
    Perform 1/D scaling

    Parameters
    ----
    datas: dict[int, ndarray]
        datas of different D.
        keys: D;
        values: [xs, ys] (as rows)
    x2s: ndarray
        values of x to be interpolated
    xlim: tuple[float, float] | None
        limit of x for input date
    xsamples: ndarray
        values of x to perform 1/D scaling
    """
    # interpolate data
    assert mode in ("linear", "spline")
    datas_ip: dict[int, ndarray] = {}
    for D, data in datas.items():
        order = np.argsort(data[0])
        xs, ys = data[0][order], data[1][order]
        if xlim is not None:
            assert xlim[1] > xlim[0]
            select = np.where((xs >= xlim[0]) & (xs <= xlim[1]))
            xs, ys = xs[select], ys[select]
        if mode == "linear":
            y2s = np.interp(x2s, xs, ys)
        else:
            ysip = spip.CubicSpline(xs, ys)
            y2s = ysip(x2s)
        datas_ip[D] = np.stack([x2s, y2s])

    # perform 1/D scaling
    x2s_, ks, bs, errs = [], [], [], []
    Ds = 1 / np.array([D for D in datas.keys()])
    for x in x2s:
        # find interpolated ys corresponding to x
        ys = np.zeros(Ds.size)
        for i, data in enumerate(datas_ip.values()):
            idx = int(np.where(data[0] == x)[0][0])
            ys[i] = data[1, idx]
        result = linregress(Ds, ys)
        if result.rvalue**2 > minr2:
            x2s_.append(x)
            ks.append(result.slope)
            bs.append(result.intercept)
            errs.append(result.intercept_stderr)
    scale_data = np.array([x2s_, ks, bs, errs])
    return datas_ip, scale_data

def draw_plot(
    axs: list[Axes], 
    datas: dict[int, ndarray], datas_ip: dict[int, ndarray], 
    scale_data: ndarray, color: None | str=None, 
    id0: int = 0, scaleplot=False, xsamples=None, draw_ip=False, 
):
    """
    Plot 1/D scaling result in provided matplotlib axis `ax`
    """
    markers = ("o", "s", "^", "v", "p")
    line_params = dict(markerfacecolor="none", linewidth=1, color=color)
    if scaleplot:
        assert xsamples is not None
        assert len(axs) == 2
    else:
        assert len(axs) == 1
    
    # plot scaling results and original data
    if scaleplot: 
        ax = axs[0]
    for (D, data), marker in zip(datas.items(), markers):
        ax.plot(*data, label=f"D = {D}", marker=marker, **line_params)
        if draw_ip:
            ax.plot(
                *datas_ip[D], label=f"D = {D} (fit)", 
                marker=marker, 
                alpha=(1.0 if color is None else 0.6),
                linestyle=":", **line_params
            )
    x2s_, ks, bs, errs = scale_data
    ax.errorbar(
        x2s_[id0::], bs[id0::], yerr=errs[id0::]/2, 
        color=("black" if color is None else color), 
        capsize=3, linewidth=1, marker='o', markersize=4,
        label="D" + r"$\to \infty$", 
    )
    ax.set_ylim(bottom=0)
    
    # plot 1/D linear fit
    if scaleplot:
        ax = axs[1]
        Ds = 1 / np.array(list(datas.keys()))
        tmp = np.array([0.0, np.max(Ds) * 1.2])
        line_params = dict(linewidth=1, linestyle="--", color=color)
        for x, alpha in zip(
            xsamples, np.linspace(1.0, 0.2, len(xsamples))
        ):
            alpha_ = 1.0 if color is None else alpha
            try:
                idx = int(np.where(x2s_ == x)[0][0])
                ys = np.array([data[1, idx] for data in datas_ip.values()])
                ax.scatter(Ds, ys, label=f"δ={x}", s=16, color=color, alpha=alpha_)
                k, b = ks[idx], bs[idx]
                ax.plot(tmp, k*tmp+b, alpha=alpha_, **line_params)
            except IndexError:
                continue
        ax.set_xlim(left=0, right=tmp[1])
        ax.set_xlabel(r"$1/D$")
    return axs

