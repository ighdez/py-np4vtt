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
from py_np4vtt.model_ann import ModelANN, ConfigANN
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
    df = pd.read_table(reporoot_dir / 'data' / 'Norway2009VTT_demodata.txt')

    model_arrays = make_modelarrays(df, columnarrays)

    # Step 2: Do descriptives
    _descriptives = compute_descriptives(model_arrays)

    # Step 3: Make config
    config = ConfigANN([10, 10], 20, 50, None)

    # Step 4: Call model
    ann = ModelANN(config, model_arrays)
    initialArgs = ann.setupInitialArgs()
    ll, r2, clf, vtt = ann.run(initialArgs,verbose=True)

    # Export to excel
    # Check if the model reached the expected results
    ll_expected = 107864.1944
    r2_expected = 0.466340404

    ll_mean = -ll.mean()
    r2_mean = r2.mean()

    print('ANN checks:')
    if not check_in_range(ll_expected, ll_mean, margin_proportion=0.15):
        print('Log-likelihood: too far from expected. Expected={ll_expected}, Actual={ll_mean}')
    else:
        print('Log-likelihood: PASS')

    if not check_in_range(r2_expected, r2_mean, margin_proportion=0.1):
        print('Rho-squared: too far from expected. Expected={r2_expected}, Actual={r2_mean}')
    else:
        print('Rho-squared: PASS')


if __name__ == '__main__':
    run_test()
