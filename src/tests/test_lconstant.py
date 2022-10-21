#  Copyright 2021 Technische Universiteit Delft
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
from pathlib import Path
import pandas as pd

from py_np4vtt.data_format import Vars
from py_np4vtt.model_lconstant import ModelLConstant, ConfigLConstant
from py_np4vtt.data_import import make_modelarrays, compute_descriptives

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
    config = ConfigKSpady(minimum=0, maximum=18, supportPoints=19)

    # Step 4: Call model
    kspady = ModelKSpady(config, model_arrays)
    initialArgs = kspady.setupInitialArgs()
    p, vtt_grid = kspady.run(initialArgs)

    return vtt_grid, p

if __name__ == '__main__':
    run_test()
