#  Copyright 2021 Technische Universiteit Delft
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
import pandas as pd
import numpy as np

from model.data_format import StudyVar, StudyVarMapping, DescriptiveStatsBasic, ModelArrays


class VarMappingException(Exception):
    def __init__(self, missingVar: StudyVar, colName: str):
        self.missingVar = missingVar
        self.colName = colName

    def __str__(self):
        return f"The study variable '{self.missingVar}' (mapped to column '{self.colName}') is missing from the dataset"


def make_modelarrays(dataset_frame: pd.DataFrame, dataset_varmapping: StudyVarMapping) -> ModelArrays:
    study_arrays = {}

    for v in StudyVar:
        colName = dataset_varmapping[v]
        arr = dataset_frame.get(colName)
        if arr is None:
            raise VarMappingException(v, colName)
        else:
            study_arrays[v] = arr

    id_all = study_arrays[StudyVar.Id]
    id_uniq = pd.unique(id_all)
    npar = id_uniq.size
    t = id_all.size / id_uniq.size

    cost1 = study_arrays[StudyVar.Cost1]
    cost2 = study_arrays[StudyVar.Cost2]
    cheap_alt = pd.Series(np.zeros(study_arrays[StudyVar.Cost1].size))
    cheap_alt.loc[cost1 < cost2] = 1
    cheap_alt.loc[cost2 < cost1] = 2

def compute_descriptives(arrs: ModelArrays) -> DescriptiveStatsBasic:
    """
        chosen_BVTT = app.arrays.Choice .* app.arrays.BVTT;
        chosen_FnEx = sum(app.arrays.Choice,2);
        nt_CheapnSl = sum(chosen_FnEx==0);
        nt_FastnExp = sum(chosen_FnEx==app.arrays.T);

        textline1 = ['No. individuals: ',' ',num2str(app.arrays.NP)];
        textline2 = ['Sets per indiv.: ',' ',num2str(app.arrays.T)];

        textline3 = 'Number of non-traders:';
        textline4 = ['Fast-exp. alt.: ','  ',num2str(nt_FastnExp)];
        textline5 = ['Slow-cheap alt.: ',' ',num2str(nt_CheapnSl)];

        textline6 = 'BVTT statistics:';
        textline7 = ['Mean chosen BVTT:',' ',num2str(mean(chosen_BVTT(:)))];
        textline8 = ['Minimum of BVTT:','  ',num2str(min(app.arrays.BVTT(:)))];
        textline9 = ['Maximum of BVTT:','  ',num2str(max(app.arrays.BVTT(:)))];

        dataset_info = {'Dataset information:';...
                        '';...
                        textline1;...
                        textline2;...
                        ' ';...
                        textline3;...
                        textline4;...
                        textline5;...
                        ' ';...
                        textline6;...
                        textline7;...
                        textline8;...
                        textline9};
    """

    # TODO
    return DescriptiveStatsBasic()
