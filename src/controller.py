#  Copyright 2021 Technische Universiteit Delft
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#
from typing import List, Optional
from pathlib import Path
import csv

import pandas as pd

from model.data_format import StudyVarMapping, DescriptiveStatsBasic, StudiedArrays, ModelArrays
from model.data_import import make_modelarrays, compute_descriptives


dataset_frame: Optional[pd.DataFrame] = None
dataset_varmapping: StudyVarMapping = {}

model_arrays: Optional[ModelArrays] = None


def openDataset(chosenPath: Path) -> List[str]:
    fullPath = chosenPath.resolve()
    dialect = csv.excel_tab if fullPath.suffix == '.txt' else csv.excel

    with open(fullPath) as file:
        reader = csv.reader(file, dialect)
        rows = list(reader)

    # We assume the data file has a header on first line
    column_names = rows[0]
    rows_numeric = [map(int, row) for row in rows[1:]]
    data_dicts = [dict(zip(column_names, row)) for row in rows_numeric]

    # Make the Pandas dataframe
    global dataset_frame
    dataset_frame = pd.DataFrame(data_dicts)

    return column_names


def importMappedDataset(mapping: StudyVarMapping) -> DescriptiveStatsBasic:
    global dataset_varmapping
    dataset_varmapping = mapping

    global model_arrays
    model_arrays = make_modelarrays(dataset_frame, dataset_varmapping)

    return compute_descriptives(model_arrays)


def modelConfig_loclogit():
    pass

def modelConfig_rouwendal():
    pass

def modelConfig_logit():
    pass

def modelConfig_ann():
    pass

