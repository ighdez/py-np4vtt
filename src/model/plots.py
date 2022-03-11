#  Copyright 2021 Technische Universiteit Delft
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
import numpy as np
import matplotlib.pyplot as plt


def run_plot(vtts: np.ndarray, probs: np.ndarray):
    # Grid of subfigures
    fig = plt.figure()
    axs = fig.add_subplot(1, 1, 1)  # Single subfigure

    # Decorations
    axs.set_title('Local logit ECDF')
    axs.set_xlabel('VTT')
    axs.set_ylabel('Cumulative density')

    axs.xaxis.grid()
    axs.yaxis.grid()

    data_x = vtts
    # TODO: what exactly shall we append here to make the x and y data arrays have equal size?
    data_y = np.append(probs, [1.0])

    # TODO: formatting just a parameter or is
    format_points_ecdf = 'bo-'
    axs.plot(data_x, data_y, format_points_ecdf)

    yticks_start = 0.0
    yticks_stop = 1.0
    yticks_num = 10  # TODO: how many subdivisions do we want? calculate automatically?
    axs.set_yticks(np.linspace(yticks_start, yticks_stop, yticks_num))

    plt.show()
