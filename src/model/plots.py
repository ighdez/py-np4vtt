#  Copyright 2021 Technische Universiteit Delft
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def run_plot_hist():
    # % Set min, max and bin width for automatic mode
    # app.auto_min_bvtt = floor(min(app.CallingApp.arrays.BVTT(:)));
    # app.auto_max_bvtt = ceil(max(app.CallingApp.arrays.BVTT(:)));
    #
    # % Step 1: get No.of data points and min - max of BVTT
    # N = app.CallingApp.arrays.NP;
    #
    # % Step 2: No.of bins is sqrt(N)
    # n_bins = ceil(sqrt(N));
    #
    # % Step 3: Calculate bin width
    # spec_tolerance = app.auto_max_bvtt - app.auto_min_bvtt;
    # app.auto_binwidth = round(floor(spec_tolerance) / n_bins, 2);
    pass


def run_plot_ecdf(vtts: np.ndarray, probs: np.ndarray):
    # Grid of subfigures
    fig = plt.figure()
    axs = fig.add_subplot(1, 1, 1)  # Single subfigure

    # Decorations
    axs.set_title('Local logit ECDF')
    axs.set_xlabel('VTT')
    axs.set_ylabel('Cumulative density')

    axs.xaxis.grid()
    axs.yaxis.grid()

    # TODO: check that adding these points actually is correct
    # This is because the cumulative prob. when x=0 should be 0.0
    data_x = np.append([0.0], vtts[1:])
    data_y = np.append([0.0], probs)

    # TODO: formatting just a parameter or is
    format_points_ecdf = 'bo-'
    axs.plot(data_x, data_y, format_points_ecdf)

    # TODO: The point (0,0) should be on the bottom left

    # Must be like this because it's a probabilty
    yticks_start = 0.0
    yticks_stop = 1.0

    # TODO: just make that the labels are presented with rounder decimals
    yticks_num = 11
    axs.set_yticks(np.linspace(yticks_start, yticks_stop, yticks_num))

    # TODO: check that matplotlib is smart to not show ALL THE POINTS
    axs.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.show()
