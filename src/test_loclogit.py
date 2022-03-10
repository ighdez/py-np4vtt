#  Copyright 2021 Technische Universiteit Delft
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from model.data_format import StudyVar
from model.model_loclogit import ModelLocLogit, ConfigLocLogit
from model.data_import import make_modelarrays, compute_descriptives

from helpers import check_in_range


def run_test():
    # Step 1: read CSV file
    columnarrays = {
        StudyVar.Id: 'RespID',
        StudyVar.ChosenAlt: 'Chosen',
        StudyVar.Cost1: 'CostL',
        StudyVar.Cost2: 'CostR',
        StudyVar.Time1: 'TimeL',
        StudyVar.Time2: 'TimeR',
    }

    curscript_dir = Path(__file__).resolve().parent
    df = pd.read_table(curscript_dir.parent / 'data' / 'Norway09_data_v5.txt')

    model_arrays = make_modelarrays(df, columnarrays)

    # Step 2: Do descriptives
    _descriptives = compute_descriptives(model_arrays)

    # Step 3: Make config
    config = ConfigLocLogit(minimum=0, maximum=18, supportPoints=19)

    # Step 4: Call model
    loclogit = ModelLocLogit(config, model_arrays)
    p, fval, vtt_grid = loclogit.run()

    # Check if the model reached the expected results
    f_final_expected = -27596.28

    # Is it within an interval with margin?
    if check_in_range(f_final_expected, fval, margin_proportion=0.1):
        print('Final F-value: PASS')
    else:
        print(f'Final F-value: FAIL too far from expected. Expected={f_final_expected}, Actual={fval}')

    return vtt_grid, p


# Plot here quickly, we will make this nice into the library
def run_plot(vtts, probs):
    fig = plt.figure()
    axs = fig.add_subplot(1, 1, 1)  # Single subfigure

    axs.set_title('Local logit ECDF')
    axs.set_xlabel('VTT')
    axs.set_ylabel('Cumulative density')

    axs.xaxis.grid()
    axs.yaxis.grid()

    data_x = vtts
    data_y = np.append(probs, [1.0])

    axs.plot(data_x, data_y, 'bo-')

    axs.set_yticks(np.linspace(0.0, 1.0, 10))

    plt.show()


if __name__ == '__main__':
    run_test()
