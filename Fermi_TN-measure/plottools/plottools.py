"""
Common modules to be imported to plot figures
"""

import numpy as np
from pprint import pprint
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
from itertools import cycle, product
from matplotlib.ticker import MultipleLocator
rc('font', **{'family': 'serif'})
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.size'] = 5  # length of major ticks (default is 3.5)
plt.rcParams['xtick.minor.size'] = 2.5  # length of minor ticks (if any)
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['ytick.minor.size'] = 2.5
plt.rcParams['xtick.major.width'] = 0.8  # width of major ticks (default is 0.8)
plt.rcParams['ytick.major.width'] = 0.8
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.right'] = True

def create_fig(
    xlabel: str, ylabel: str, 
    xrange: tuple[float, float], yrange: tuple[float, float],
    figsize: tuple[int, int] = (5, 5)
):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plt.subplots_adjust(
        left=0.18, bottom=0.15, right=0.95, top=0.9, 
        wspace=0.3, hspace=0.3
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(left=xrange[0], right=xrange[1])
    ax.set_ylim(bottom=yrange[0], top=yrange[1])
    return fig, ax
