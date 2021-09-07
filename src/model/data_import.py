#  Copyright 2021 Technische Universiteit Delft
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
import pandas as pd

from model.data_format import StudyVar, StudyVarMapping, StudyDescriptives, StudyArrays


class VarMappingException(Exception):
    def __init__(self, missingVar: StudyVar, colName: str):
        self.missingVar = missingVar
        self.colName = colName

    def __str__(self):
        return f"The study variable '{self.missingVar}' (mapped to column '{self.colName}') is missing from the dataset"


def make_studyarrays(dataset_frame: pd.DataFrame, dataset_varmapping: StudyVarMapping) -> StudyArrays:
    study_arrays = {}

    for v in StudyVar:
        colName = dataset_varmapping[v]
        arr = dataset_frame.get(colName)
        if arr is None:
            raise VarMappingException(v, colName)
        else:
            study_arrays[v] = arr

    return study_arrays

def compute_descriptives(arrays_study: StudyArrays) -> StudyDescriptives:
    # Reshape and re-header DataFrame to contain only mapped columns

    # TODO
    return StudyDescriptives()
