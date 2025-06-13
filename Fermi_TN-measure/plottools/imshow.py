import numpy as np
from numpy import ndarray
from cmath import phase, pi
from .plottools import *
from matplotlib.colors import LinearSegmentedColormap

cmap = LinearSegmentedColormap.from_list(
    'custom', [
        (0,"lightsalmon"), (0.25,"violet"), (0.5,"lightblue"), 
        (0.75,"limegreen"), (1,"lightsalmon")]
)


def myphase(z: float|complex):
    """Return normalized phase angle x (0 to 1) 
    of a complex number z = exp(2*pi*x)"""
    arg = phase(z)
    if arg < 0: arg += 2*pi
    return arg / (2*pi)


def get_linewidth(bondval: complex, smax: float):
    if smax == 0:
        return 5.0
    else:
        return np.abs(bondval) / smax * 20


def get_color(bondval: complex):
    if bondval == 0:
        return "lightgray"
    else:
        return cmap(myphase(bondval))


def imshow_config(
    N1: int, N2: int, e0: float,
    holes: ndarray, mags: ndarray|None,
    bondVs: ndarray, bondHs: ndarray,
    figname: str|None,
):
    """
    Plot PEPS configuration on 1st neighbor bonds
    """
    unit_size = 3
    fig = plt.figure(figsize=(N2 * unit_size, N1 * unit_size))
    ax = fig.add_subplot(111)

    py = [i + 1 for i in range(N1) for j in range(N2)]
    px = [j + 1 for i in range(N1) for j in range(N2)]

    iterators = [(i, j) for j in range(N2) for i in range(N1)]
    smax = max(np.max(np.abs(bondHs)), np.max(np.abs(bondVs)))
    for i, j in iterators:
        ax.plot(
            [j + 1, j + 1], [i + 1, i + 2],
            color=get_color(bondVs[i, j]),
            linewidth=get_linewidth(bondVs[i, j], smax),
        )
        ax.plot(
            [j + 1, j + 2], [i + 1, i + 1],
            color=get_color(bondHs[i, j]),
            linewidth=get_linewidth(bondHs[i, j], smax),
        )
        if j == N2 - 1:
            ax.plot(
                [j + 1 - N2, j + 2 - N2], [i + 1, i + 1],
                color=get_color(bondHs[i, j]),
                linewidth=get_linewidth(bondHs[i, j], smax),
            )
        if i == N1 - 1:
            ax.plot(
                [j + 1, j + 1], [i + 1 - N1, i + 2 - N1],
                color=get_color(bondVs[i, j]),
                linewidth=get_linewidth(bondVs[i, j], smax),
            )
        ax.text(j + 1, i + 1.4, "{:.4}".format(bondVs[i, j]), zorder=30)
        ax.text(j + 1.4, i + 1.06, "{:.4}".format(bondHs[i, j]), zorder=30)
        ax.text(j + 1 + 0.1, i + 1 + 0.15, "h  {:.4}".format(holes[i, j]), zorder=30)
        if mags is not None:
            magsnorm = np.linalg.norm(mags, axis=0)
            ax.text(j + 1 + 0.1, i + 1 - 0.15, "|m| {:.4}".format(magsnorm[i, j]), zorder=30)
            # magnetization direction is determined by sign of Re(Sz)
            signs = np.sign(mags[2].real)
            mag = signs[i,j] * magsnorm[i, j]
            ax.arrow(
                j + 1, i + 1 - mag / 4, 0, mag / 2, width=0.02, fc="black", zorder=20
            )

    ax.scatter(px, py, holes.real * 30000, "lightgreen", zorder=10)

    ax.set_xlim([0.5, N2 + 0.5])
    ax.set_ylim([0.5, N1 + 0.5])
    ax.set_title(
        "N1={}, N2={}, hole0={:.4f}, e0={:.4f}".format(N1, N2, np.mean(holes.real), e0)
    )
    ax.set_xticks([])
    ax.set_yticks([])
    if figname is not None:
        fig.savefig(figname, dpi=144)
        plt.close()
    else:
        plt.show()
    return fig, ax

