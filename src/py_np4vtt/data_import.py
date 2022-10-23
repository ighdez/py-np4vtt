#  Copyright 2021 Technische Universiteit Delft
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""Methods to create model arrays and generate descriptive statistics."""

import math
from typing import List

import pandas as pd
import numpy as np

from py_np4vtt.data_format import Vars, VarsMapping, DescriptiveStatsBasic, ModelArrays, Arrays


class VarMappingException(Exception):
    def __init__(self, missingVar: Vars, colName: str):
        self.missingVar = missingVar
        self.colName = colName

    def __str__(self):
        return f"The variable '{self.missingVar}' (mapped to column '{self.colName}') is missing from the dataset"


def make_arrays(dataset_frame: pd.DataFrame, dataset_varmapping: VarsMapping) -> Arrays:
    studied_arrays = {}

    for v in Vars:
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


def make_modelarrays(dataset_frame: pd.DataFrame, dataset_varmapping: VarsMapping) -> ModelArrays:
    """Create model arrays.

    This function takes a Pandas `DataFrame` and a dictionary that contains 
    the mapping between the necessary variables and the variable names in 
    the dataset.

    Parameters
    ----------
    dataset_frame : pandas.DataFrame
        A Pandas `DataFrame` that contains the dataset.

    dataset_varmapping: Dict[Vars,str]
        A dictionary file that maps the necessary variables required by NP4VTT 
        with the variable names in the dataset. Each key is of the format 
        `Vars.variablename` where `variablename` is one of the required 
        variables (see the documentation of `py_np4vtt.data_format.Vars` for 
        more details about the required variables). Each value is a string 
        that contains the name of the variable corresponding to the necessary 
        variable as it appears in `dataset_frame`.

    Returns
    -------
    ModelArrays
        An object containing the model arrays used by the nonparametric models.
    """
    study_arrays = make_arrays(dataset_frame, dataset_varmapping)

    # Copy to avoid changing the original data imported
    cost1 = study_arrays[Vars.Cost1].to_numpy(copy=True)
    cost2 = study_arrays[Vars.Cost2].to_numpy(copy=True)
    time1 = study_arrays[Vars.Time1].to_numpy(copy=True)
    time2 = study_arrays[Vars.Time2].to_numpy(copy=True)
    choice = study_arrays[Vars.ChosenAlt].to_numpy(copy=True)

    # Identify times and costs
    t1 = np.c_[time1,time2].max(axis=1)   # Higher time
    t2 = np.c_[time1,time2].min(axis=1)   # Lower time
    c1 = np.c_[cost1,cost2].min(axis=1)   # Lower cost
    c2 = np.c_[cost1,cost2].max(axis=1)   # Higher cost


    # Identify expensive and slow alternative
    cheap_alt = np.c_[cost1,cost2].argmin(axis=1) + 1
    slow_alt = np.c_[time1,time2].argmax(axis=1) + 1

    # Create scalars and ID variables
    id_all = study_arrays[Vars.Id]
    id_uniq = pd.unique(id_all)
    npar = id_uniq.size
    t = id_all.size / id_uniq.size

    errorList = validate_modeldata(id_all, t, c1, c2, t1, t2, slow_alt, cheap_alt, choice)
    t_int = math.floor(t)

    # BVTT
    bvtt = (- (c1-c2)/(t1-t2)).reshape((npar,t_int))

    # FBE = "Fast But Expensive"
    fbe_chosen = (choice != cheap_alt).reshape((npar, t_int))

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
    """Compute descriptive statistics
    
    It takes the object that contains the model arrays and returns a set of 
    relevant descriptive statistics regarding respondents, their choices and 
    about the BVTT.
    
    Parameters
    ----------
    arrs : ModelArrays
        The model arrays object

    Returns
    -------

    DescriptiveStatsBasic
        An object that contains the relevant descriptive statistics. Can be 
        accessed using `print()`.
    """
    fbe_units = arrs.Choice.astype(int)

    chosenBVTT = (fbe_units * arrs.BVTT)

    # noinspection PyTypeChecker
    chosenBVTT_mean: int = np.sum(chosenBVTT)/np.sum(fbe_units)

    chosen_fastexp = np.sum(fbe_units, 1)
    nt_cheapslow = np.count_nonzero(chosen_fastexp == 0)
    nt_fastext = np.count_nonzero(chosen_fastexp == arrs.T)

    return DescriptiveStatsBasic(
        NP=arrs.NP,
        T=arrs.T,
        NT_FastExp=nt_fastext,
        NT_CheapSlow=nt_cheapslow,
        ChosenBVTT_Mean=np.round(chosenBVTT_mean,4),
        BVTT_min=np.round(np.amin(arrs.BVTT),4),
        BVTT_max=np.round(np.amax(arrs.BVTT),4),
    )
