#  Copyright 2021 Technische Universiteit Delft
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
from enum import Enum, auto

import pandas as pd
import numpy as np
import numpy.typing as npt

from dataclasses import dataclass
from typing import Dict


class StudyVar(Enum):
    Id = auto()
    ChosenAlt = auto()
    Cost1 = auto()
    Time1 = auto()
    Cost2 = auto()
    Time2 = auto()


StudyVarMapping = Dict[StudyVar, str]

StudiedArrays = Dict[StudyVar, pd.Series]


@dataclass
class ModelArrays:
    BVTT: npt.NDArray[np.float64]  # pd.Series[float]
    Choice: npt.NDArray[np.bool_]
    Accepts: npt.NDArray[np.int_]
    ID: npt.NDArray[np.int_]  # Unique participant IDs
    NP: int  # Number of participants
    T: int  # Number of choice situations per participant


@dataclass
class DescriptiveStatsBasic:
    def __str__(self) -> str:
        # TODO Maybe some rich text in here?
        return "StudyDescriptives()"
