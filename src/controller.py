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
from PyQt5.QtCore import QThread

from gui.progress import EstimationWorker, EstimationProgress
from py_np4vtt.data_format import StudyVarMapping, DescriptiveStatsBasic, ModelArrays
from py_np4vtt.data_import import make_modelarrays, compute_descriptives
from py_np4vtt.model_ann import ConfigANN
from py_np4vtt.model_loclogit import ConfigLocLogit
from py_np4vtt.model_logit import ConfigLogit
from py_np4vtt.model_rouwendal import ConfigRouwendal


dataset_frame: Optional[pd.DataFrame] = None
dataset_varmapping: StudyVarMapping = {}

model_arrays: Optional[ModelArrays] = None

modelcfg_logit: Optional[ConfigLogit] = None
modelcfg_loclogit: Optional[ConfigLocLogit] = None
modelcfg_rouwendal: Optional[ConfigRouwendal] = None
modelcfg_ann: Optional[ConfigANN] = None

__estimation_thread: Optional[QThread] = None


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


def modelConfig_loclogit(minimum: float, maximum: float, numPoints: int):
    global modelcfg_loclogit
    modelcfg_loclogit = ConfigLocLogit(
        minimum=minimum,
        maximum=maximum,
        supportPoints=numPoints,
    )
    print(modelcfg_loclogit)  # TODO: debug, remove this

def modelConfig_logit(intercept: float, parameter: float, scale: float, iterations: int, seed: int):
    global modelcfg_logit
    modelcfg_logit = ConfigLogit(
        mleIntercept=intercept,
        mleParameter=parameter,
        mleScale=scale,
        mleMaxIterations=iterations,
        seed=seed,
    )
    print(modelcfg_logit)  # TODO: debug, remove this

def modelConfig_rouwendal(minimum: float, maximum: float, numPoints: int, probConsistent: float, maxIterations: int):
    global modelcfg_rouwendal
    modelcfg_rouwendal = ConfigRouwendal(
        minimum=minimum,
        maximum=maximum,
        supportPoints=numPoints,
        startQ=probConsistent,
        # TODO maxIterations?
    )
    print(modelcfg_rouwendal)  # TODO: debug, remove this

def modelConfig_ann(hiddenLayers: List[int], numRepeats: int, numShuffles: int, seed: int):
    global modelcfg_ann
    modelcfg_ann = ConfigANN(
        hiddenLayerNodes=hiddenLayers,
        trainingRepeats=numRepeats,
        shufflesPerRepeat=numShuffles,
        seed=seed,
    )
    print(modelcfg_ann)  # TODO: debug, remove this

def modelEstimate_rouwendal():
    pass

def modelEstimate_ann():
    pass

def modelEstimate_loclogit(progressWorker: EstimationWorker, progressDialog: EstimationProgress):
    global __estimation_thread
    __estimation_thread = QThread()
    progressWorker.moveToThread(__estimation_thread)

    __estimation_thread.started.connect(progressWorker.run)
    progressWorker.finished.connect(__estimation_thread.quit)
    progressWorker.finished.connect(progressWorker.deleteLater)
    __estimation_thread.finished.connect(__estimation_thread.deleteLater)
    progressWorker.progress.connect(progressDialog.reportStatus)

    __estimation_thread.start()

def modelEstimate_logit():
    pass
