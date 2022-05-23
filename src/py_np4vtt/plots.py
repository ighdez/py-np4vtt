#  Copyright 2021 Technische Universiteit Delft
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
from typing import Optional
from enum import Enum, unique, auto
from math import ceil, floor, sqrt

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# Histogram can only be produced for Logit and ANN
def run_plot_hist(vtts: np.ndarray,
        min_bvtt_arg: Optional[float] = None, max_bvtt_arg: Optional[float] = None,
        bin_width_arg: Optional[float] = None):

    # Set min, max and bin width for automatic mode
    auto_min_bvtt = floor(np.min(vtts));
    auto_max_bvtt = ceil(np.max(vtts));
    min_bvtt = auto_min_bvtt if min_bvtt_arg is None else min_bvtt_arg
    max_bvtt = auto_max_bvtt if max_bvtt_arg is None else max_bvtt_arg

    spec_tolerance = max_bvtt - min_bvtt;

    N = vtts.size()
    n_bins = ceil(sqrt(N));

    auto_bin_width = round(floor(spec_tolerance) / n_bins, 2);
    bin_width = auto_bin_width if bin_width_arg is None else bin_width_arg

    pass


@unique
class PlotMethodLogitOrANN(Enum):
    LOGIT = auto()
    ANN = auto()

    def __str__(self):
        if self.value == PlotMethodLogitOrANN.LOGIT: return 'Logit'
        else: return 'ANN'


def run_plot_ecdf_logit_ann(
        plot_method: PlotMethodLogitOrANN,
        vtts: np.ndarray, probs: np.ndarray,
        min_bvtt: Optional[float] = None, max_bvtt: Optional[float] = None,
        ):
    # Grid of subfigures
    fig = plt.figure()
    axs = fig.add_subplot(1, 1, 1)  # Single subfigure

    # Decorations
    axs.set_title(f"{plot_method} ECDF")
    axs.set_xlabel('VTT')
    axs.set_ylabel('Cumulative density')

    # This is because the cumulative prob. when x=0 should be 0.0
    data_x = np.append([0.0], vtts[1:])
    data_y = np.append([0.0], probs)

    # The point (0,0) should be exactly on the bottom left, no margins
    axs.margins(0.0)

    # formatting just a parameter or...?
    format_points_ecdf = 'bo-'
    axs.plot(data_x, data_y, format_points_ecdf)

    # Axis limits and ticks
    ylim_min = 0.0   # Must be like this because it's a probabilty
    ylim_max = 1.0
    yticks_num = 11
    axs.set_xlim(left=min_bvtt, right=max_bvtt)
    axs.set_ylim(bottom=ylim_min, top=ylim_max)
    axs.set_yticks(np.linspace(ylim_min, ylim_max, yticks_num))

    # TODO: check that matplotlib is smart to not show ALL THE POINTS
    axs.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.show()


@unique
class PlotMethodRWOrLL(Enum):
    ROUWENDAL = auto()
    LOC_LOGIT = auto()

    def __str__(self):
        if self.value == PlotMethodRWOrLL.ROUWENDAL: return 'Rouwendal'
        else: return 'Local logit'


def run_plot_ecdf_rouwendal_loclogit(
        plot_method: PlotMethodRWOrLL,
        vtts: np.ndarray, probs: np.ndarray,
        min_bvtt: Optional[float] = None, max_bvtt: Optional[float] = None,
        ):
    # Grid of subfigures
    fig = plt.figure()
    axs = fig.add_subplot(1, 1, 1)  # Single subfigure

    # Decorations
    axs.set_title(f"{plot_method} ECDF")
    axs.set_xlabel('VTT')
    axs.set_ylabel('Cumulative density')

    # This is because the cumulative prob. when x=0 should be 0.0
    data_x = np.append([0.0], vtts[1:])
    data_y = np.append([0.0], probs)

    # The point (0,0) should be exactly on the bottom left, no margins
    axs.margins(0.0)

    # formatting just a parameter or...?
    format_points_ecdf = 'bo-'
    axs.plot(data_x, data_y, format_points_ecdf)

    # Axis limits and ticks
    ylim_min = 0.0   # Must be like this because it's a probabilty
    ylim_max = 1.0
    yticks_num = 11
    axs.set_xlim(left=min_bvtt, right=max_bvtt)
    axs.set_ylim(bottom=ylim_min, top=ylim_max)
    axs.set_yticks(np.linspace(ylim_min, ylim_max, yticks_num))

    # TODO: check that matplotlib is smart to not show ALL THE POINTS
    axs.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.show()

