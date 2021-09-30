#  Copyright 2021 Technische Universiteit Delft
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
from os import error
from typing import List, Tuple

import pandas as pd
import numpy as np

from model.data_format import StudyVar, StudyVarMapping, DescriptiveStatsBasic, ModelArrays, StudiedArrays


class VarMappingException(Exception):
    def __init__(self, missingVar: StudyVar, colName: str):
        self.missingVar = missingVar
        self.colName = colName

    def __str__(self):
        return f"The study variable '{self.missingVar}' (mapped to column '{self.colName}') is missing from the dataset"


def make_studiedarrays(dataset_frame: pd.DataFrame, dataset_varmapping: StudyVarMapping) -> StudiedArrays:
    studied_arrays = {}

    for v in StudyVar:
        colName = dataset_varmapping[v]
        arr = dataset_frame.get(colName)
        if arr is None:
            raise VarMappingException(v, colName)
        else:
            studied_arrays[v] = arr

    return studied_arrays


def validate_modeldata(id_all, t, cost1, cost2, time1, time2, slow_alt, cheap_alt, choice) -> Tuple[bool, List[str]]:

    # Create errormessage list
    errorMessages = []

    # id_all values must be finite.
    idIsFinite = np.isfinite(id_all).all()

    if not idIsFinite:
        errorMessages.append('There are either NAs or (minus) infinite values in ID Variable')

    # T must be integer
    tIsInteger = (int(t) == t)

    if not tIsInteger:
        errorMessages.append('Number of choice situations must be equal for all individuals.')

    # Costs must be finite
    cost1IsFinite = np.isfinite(cost1).all()

    if not cost1IsFinite:
        errorMessages.append('There are either NAs or (minus) infinite values in Cost of alternative 1.')

    cost2IsFinite = np.isfinite(cost2).all()

    if not cost2IsFinite:
        errorMessages.append('There are either NAs or (minus) infinite values in Cost of alternative 2.')

    # Time must be finite
    time1IsFinite = np.isfinite(time1).all()

    if not time1IsFinite:
        errorMessages.append('There are either NAs or (minus) infinite values in Time of alternative 1.')

    time2IsFinite = np.isfinite(time2).all()

    if not time2IsFinite:
        errorMessages.append('There are either NAs or (minus) infinite values in Time of alternative 2.')

    # Cheap and fast cannot be the same alternative
    nonDominantAlt = (cheap_alt == slow_alt).all()

    if not nonDominantAlt:
        errorMessages.append('At least one choice situation have either a cheap-fast or expensive-slow alternative.')

    # Choice variable must be either 1 or 2
    choiceOneOrTwo = (choice == 1 | choice == 2)

    if not choiceOneOrTwo:
        errorMessages('Chosen alternative variable must be either 1 or 2.')
    
    # Compile all integrity checks in one list
    integrityCheckList = [  idIsFinite,tIsInteger,
                            cost1IsFinite,cost2IsFinite,
                            time1IsFinite,time2IsFinite,
                            nonDominantAlt,choiceOneOrTwo]

    # Test if all statements are true
    integrityCheck = all(integrityCheckList)

    # Return True if all OK, otherwise return False and a message.
    return integrityCheck, errorMessages


def make_modelarrays(dataset_frame: pd.DataFrame, dataset_varmapping: StudyVarMapping) -> ModelArrays:
    study_arrays = make_studiedarrays(dataset_frame, dataset_varmapping)

    id_all = study_arrays[StudyVar.Id]
    id_uniq = pd.unique(id_all)
    npar = id_uniq.size
    t = id_all.size / id_uniq.size

    # Copy to avoid changing the original data imported
    cost1 = study_arrays[StudyVar.Cost1].to_numpy(copy=True)
    cost2 = study_arrays[StudyVar.Cost2].to_numpy(copy=True)
    time1 = study_arrays[StudyVar.Time1].to_numpy(copy=True)
    time2 = study_arrays[StudyVar.Time2].to_numpy(copy=True)
    choice = study_arrays[StudyVar.ChosenAlt].to_numpy(copy=True)

    # Identify the cheap alternative
    cheap_alt = np.zeros(study_arrays[StudyVar.Cost1].size, dtype=int)
    cheap_alt[cost1 < cost2] = 1
    cheap_alt[cost1 > cost1] = 2

    # Identify the slow alternative
    slow_alt = np.zeros(study_arrays[StudyVar.Time1].size, dtype=int)
    slow_alt[time1 > time2] = 1
    slow_alt[time1 < time2] = 2

    integrityCheck, errorMessages = validate_modeldata(id_all, t, cost1, cost2, time1, time2, slow_alt, cheap_alt, choice)

    # Reshape cost and time matrices, each row contains the entries for ONE participant
    cost1 = np.reshape(cost1, (npar, t), order='C')
    cost2 = np.reshape(cost2, (npar, t), order='C')
    time1 = np.reshape(time1, (npar, t), order='C')
    time2 = np.reshape(time2, (npar, t), order='C')

    # Create BVTT matrix (floating point)
    dtime = np.abs(time2 - time1)
    dcost = np.abs(cost2 - cost1)
    bvtt = dcost / dtime

    # FBE = "Fast But Expensive"
    fbe_chosen = choice != cheap_alt
    fbe_chosen = np.reshape(fbe_chosen, (npar, t), order='C')

    # The number of times a DM accepted the 'FBE' alt. Sum accross columns
    accepts = np.sum(fbe_chosen.astype(int), 1)

    return ModelArrays(
        BVTT=bvtt,
        Choice=fbe_chosen,
        Accepts=accepts,
        ID=id_uniq,
        NP=npar,
        T=t,
    )

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
