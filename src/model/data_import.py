#  Copyright 2021 Technische Universiteit Delft
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
import math
from typing import List

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


def validate_modeldata(id_all, t, cost1, cost2, time1, time2, slow_alt, cheap_alt, choice) -> List[str]:
    # Create errormessage list
    errorList = []

    if not np.isfinite(id_all).all():
        errorList.append('There are either NAs or (minus) infinite values in ID Variable')

    if not (int(t) == t):
        errorList.append('Number of choice situations must be equal for all individuals.')

    if not np.isfinite(cost1).all():
        errorList.append('There are either NAs or (minus) infinite values in Cost of alternative 1.')

    if not np.isfinite(cost2).all():
        errorList.append('There are either NAs or (minus) infinite values in Cost of alternative 2.')

    if not np.isfinite(time1).all():
        errorList.append('There are either NAs or (minus) infinite values in Time of alternative 1.')

    if not np.isfinite(time2).all():
        errorList.append('There are either NAs or (minus) infinite values in Time of alternative 2.')

    if not (cheap_alt == slow_alt).all():
        errorList.append('At least one choice situation have either a cheap-fast or expensive-slow alternative.')

    if not np.logical_or((choice == 1), (choice == 2)).all():
        errorList.append('Chosen alternative variable must be either 1 or 2.')

    # Whoever calls this validator knows that empty errorList means validator success
    return errorList


def make_modelarrays(dataset_frame: pd.DataFrame, dataset_varmapping: StudyVarMapping) -> ModelArrays:
    study_arrays = make_studiedarrays(dataset_frame, dataset_varmapping)

    # Copy to avoid changing the original data imported
    cost1 = study_arrays[StudyVar.Cost1].to_numpy(copy=True)
    cost2 = study_arrays[StudyVar.Cost2].to_numpy(copy=True)
    time1 = study_arrays[StudyVar.Time1].to_numpy(copy=True)
    time2 = study_arrays[StudyVar.Time2].to_numpy(copy=True)
    choice = study_arrays[StudyVar.ChosenAlt].to_numpy(copy=True)

    # Identify the cheap alternative
    cheap_alt = np.zeros(study_arrays[StudyVar.Cost1].size, dtype=int)
    cheap_alt[cost1 < cost2] = 1
    cheap_alt[cost1 > cost2] = 2

    # Identify the slow alternative
    slow_alt = np.zeros(study_arrays[StudyVar.Time1].size, dtype=int)
    slow_alt[time1 > time2] = 1
    slow_alt[time1 < time2] = 2

    id_all = study_arrays[StudyVar.Id]
    id_uniq = pd.unique(id_all)
    npar = id_uniq.size
    t = id_all.size / id_uniq.size

    errorList = validate_modeldata(id_all, t, cost1, cost2, time1, time2, slow_alt, cheap_alt, choice)

    t_int = math.floor(t)

    # Reshape cost and time matrices, each row contains the entries for ONE participant
    cost1 = np.reshape(cost1, (npar, t_int), order='C')
    cost2 = np.reshape(cost2, (npar, t_int), order='C')
    time1 = np.reshape(time1, (npar, t_int), order='C')
    time2 = np.reshape(time2, (npar, t_int), order='C')

    # Create BVTT matrix (floating point)
    dtime = np.abs(time2 - time1)
    dcost = np.abs(cost2 - cost1)
    bvtt = dcost / dtime

    # FBE = "Fast But Expensive"
    fbe_chosen = choice != cheap_alt
    fbe_chosen = np.reshape(fbe_chosen, (npar, t_int), order='C')

    # The number of times a DM accepted the 'FBE' alt. Sum accross columns
    accepts = np.sum(fbe_chosen.astype(int), 1)

    return ModelArrays(
        BVTT=bvtt,
        Choice=fbe_chosen,
        Accepts=accepts,
        ID=id_uniq,
        NP=npar,
        T=t_int,
    )


def compute_descriptives(arrs: ModelArrays) -> DescriptiveStatsBasic:
    fbe_units = arrs.Choice.astype(int)

    chosenBVTT = fbe_units * arrs.BVTT
    # noinspection PyTypeChecker
    chosenBVTT_mean: int = np.mean(chosenBVTT)

    chosen_fastexp = np.sum(fbe_units, 1)
    nt_cheapslow = np.count_nonzero(chosen_fastexp == 0)
    nt_fastext = np.count_nonzero(chosen_fastexp == arrs.T)

    return DescriptiveStatsBasic(
        NP=arrs.NP,
        T=arrs.T,
        NT_FastExp=nt_fastext,
        NT_CheapSlow=nt_cheapslow,
        ChosenBVTT_Mean=chosenBVTT_mean,
        BVTT_min=np.amin(arrs.BVTT),
        BVTT_max=np.amax(arrs.BVTT),
    )
