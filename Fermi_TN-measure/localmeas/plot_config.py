"""
Plot the configuration of doping, 
singlet SC parameter and magnetization
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from math import sqrt, pi
from cmath import phase
from matplotlib import rc
from update_ftps.sutools import get_tswts
from .local_measure2 import wtkey_bonds
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


def plot_config(
    *, e_site: float, dope: np.ndarray, 
    scorder: dict[str, float|complex], 
    mag: np.ndarray, mag_norm: np.ndarray, **kwargs
):
    """
    Draw doping, magnetization and singlet SC parameter
    """
    rc('font', **{'size': 10})
    t4 = (len(dope) == 4)
    assert mag.shape[0] == 3
    # unit cell size
    Nx, Ny = 2, 2
    # if t4 is False, convert input to t4 format
    if t4 is False:
        assert len(dope) == mag.shape[1] == 2
        dope = np.tile(dope, 2)
        mag = np.tile(mag, (1, 2))
        mag_norm = np.tile(mag_norm, 2)
        keys = [key for key in scorder.keys()]
        for key in keys:
            key2 = key[0:2] + "2"
            scorder[key2] = scorder[key]
    # create figure
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    plt.subplots_adjust(
        # left=0.1, bottom=0.15, right=0.97, 
        top=0.85, # wspace=0.3, hspace=0.3
    )
    ax.set_ylim(bottom=-0.5, top=0.5)
    ax.set_xlim(left=-0.5, right=0.5)
    ax.set_aspect(1)
    ax.axis("off")
    # show energy per site
    ax.text(
        0.5, 0.1, 
        r"$e_0$ = {:.3f}".format(e_site),
        # + "\n" + r"$h$ = {:.3f}".format(np.mean(dopings)), 
        horizontalalignment='center', verticalalignment='center',
        transform=ax.transAxes
    )
    # location of the 4 sites (A, B, C, D)
    artists = []
    ts_xy = {
        "Ta": np.array([-0.25, -0.25]),
        "Tb": np.array([0.25, -0.25]),
        "Tc": np.array([0.25, 0.25]),
        "Td": np.array([-0.25, 0.25])
    }
    ts_n = {
        "Ta": [0,0], "Tb": [1,0], 
        "Tc": [1,1], "Td": [0,1]
    }
    # draw 1st neighbor singlet pairing
    bond_len = 0.25
    for idx, (tname, xy) in enumerate(ts_xy.items()):
        # (right and top of each tensor)
        # SC order parameter on 4 axes of current tensor
        sc_amps = [
            scorder[wtkey_bonds[wtkey]] 
            for wtkey in get_tswts(tname, t4=True)
        ]
        n = ts_n[tname]
        for tsax, amp in enumerate(sc_amps):
            # beginning and end of the bond line
            xdata = [xy[0]] * 2
            ydata = [xy[1]] * 2
            if tsax == 0:   xdata[1] += bond_len
            elif tsax == 1: ydata[1] += bond_len
            elif tsax == 2: xdata[1] -= bond_len
            else:           ydata[1] -= bond_len
            artists.append(Line2D(
                xdata, ydata, color = cmap(myphase(amp)), zorder = 0,
                linewidth=(abs(amp) * 400 if amp != 0.0 else 4.0),
            ))
            # label SC order on the right and top
            if tsax in (0, 1):
                label_xy = [xdata[1], ydata[1]]
                if n[0] == Nx - 1 and tsax == 0:
                    label_xy[0] -= bond_len * 0.4
                if n[1] == Ny - 1 and tsax == 1:
                    label_xy[1] -= bond_len * 0.3
                ax.text(
                    *label_xy, r"$\Delta$: {:.3f}".format(amp),
                    horizontalalignment='center', verticalalignment='center'
                )
        # draw doping
        rad = dope[idx]
        artists.append(Circle(
            xy, radius=sqrt(rad/24), color='lightgreen', zorder=1
        ))
        ax.text(
            xy[0]-0.02, xy[1]+0.06, r"$h$: {:.3f}".format(rad),
            horizontalalignment='right', verticalalignment='center'
        )
        # draw magnetization (z-sign * magnitude)
        msign = 1 if (mag[2, idx].real >= 0) else -1
        m = msign * mag_norm[idx]
        ax.arrow(
            *(xy - np.asarray([0, m/16])), *np.asarray([0, m/8]), 
            linewidth=2.4, head_width=0.015, color="black", zorder=2
        )
        ax.text(
            xy[0]-0.02, xy[1]-0.06, r"$|m|$: {:.3f}".format(m),
            horizontalalignment='right', verticalalignment='center'
        )
    for art in artists:
        ax.add_artist(art)
    fig.tight_layout(rect=[0., 0., 1., 0.95])
    return fig, ax
