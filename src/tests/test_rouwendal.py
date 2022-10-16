#  Copyright 2021 Technische Universiteit Delft
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
import pandas as pd
from pathlib import Path

from py_np4vtt.data_format import Vars
from py_np4vtt.model_rouwendal import ConfigRouwendal, ModelRouwendal
from py_np4vtt.data_import import make_modelarrays, compute_descriptives

from tests.test_helpers import check_in_range

def run_test():
    # Step 1: read CSV file
    columnarrays = {
        Vars.Id: 'RespID',
        Vars.ChosenAlt: 'Chosen',
        Vars.Cost1: 'CostL',
        Vars.Cost2: 'CostR',
        Vars.Time1: 'TimeL',
        Vars.Time2: 'TimeR',
    }

    curscript_dir = Path(__file__).resolve().parent
    reporoot_dir = curscript_dir.parent.parent
    df = pd.read_table(reporoot_dir / 'data' / 'Norway09_data_v5.txt')

    model_arrays = make_modelarrays(df, columnarrays)

    # Step 2: Do descriptives
    _descriptives = compute_descriptives(model_arrays)

    # Step 3: Make config
    config = ConfigRouwendal(0, 17, 18, 0.9)

    # Step 4: Call model
    rouwendal = ModelRouwendal(config, model_arrays)
    initialArgs, initialVal = rouwendal.setupInitialArgs()
    q_prob, q_est, q_se, par, se, fvtt, cumsum_fvtt, vtt_grid, fval, exitflag, output = rouwendal.run(initialArgs)

    # Check if the model reached the expected results
    f_initial_expected = -29058.2263  # TODO: Grab actual value from MATLAB code
    f_final_expected = -23335.63
    q_prob_expected = 0.90069

    # TODO: check the initialValue
    print('Rouwendal method checks:')
    if not check_in_range(f_initial_expected, initialVal, margin_proportion=0.1):
        print('Initial F-value: too far from expected. Expected={f_initial_expected}, Actual={initialVal}')
    else:
        print('Initial F-value: PASS')

    if not check_in_range(f_final_expected, fval, margin_proportion=0.1):
        print('Final F-value: too far from expected. Expected={f_final_expected}, Actual={fval}')
    else:
        print('Final F-value: PASS')

    if not check_in_range(q_prob_expected, q_prob, margin_proportion=0.01):
        print('Q Prob: too far from expected. Expected={q_prob_expected}, Actual={q_prob}')
    else:
        print('Q Prob: PASS')


if __name__ == '__main__':
    run_test()
