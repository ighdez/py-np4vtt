#  Copyright 2021 Technische Universiteit Delft
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
import pandas as pd

from model.data_format import VarMapping, StudyDescriptives, StudyArrays


def validate_dataset(dataset_frame: pd.DataFrame, dataset_varmapping: VarMapping) -> StudyArrays:
    varID = dataset_frame.get(dataset_varmapping.varId)
    varChoice = dataset_frame.get(dataset_varmapping.varChosenAlt)

    varCost1 = dataset_frame.get(dataset_varmapping.varCost1)
    varCost2 = dataset_frame.get(dataset_varmapping.varCost2)
    varTime1 = dataset_frame.get(dataset_varmapping.varTime1)
    varTime2 = dataset_frame.get(dataset_varmapping.varTime2)

def compute_descriptives(arrays_study: StudyArrays) -> StudyDescriptives:
    # Reshape and re-header DataFrame to contain only mapped columns

    # TODO
    return StudyDescriptives()
