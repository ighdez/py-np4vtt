#  Copyright 2021 Technische Universiteit Delft
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""Data format classes"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict
from inspect import cleandoc

import pandas as pd
import numpy as np
import numpy.typing as npt


class Vars(Enum):
    Id = auto()
    ChosenAlt = auto()
    Cost1 = auto()
    Time1 = auto()
    Cost2 = auto()
    Time2 = auto()


VarsMapping = Dict[Vars, str]

Arrays = Dict[Vars, pd.Series]


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
    NP: int  # Number of participants
    T: int  # Number of choice situations per participant
    NT_FastExp: int  # Number of non-traders (fast-expensive alt.)
    NT_CheapSlow: int  # Number of non-traders (cheap-slow alt.)
    ChosenBVTT_Mean: float  # Mean chosen BVTT
    BVTT_min: float  # Minimum of BVTT
    BVTT_max: float  # Maximum of BVTT

    def __str__(self) -> str:
        formatted_descriptives = cleandoc(f"""
            No. individuals: {self.NP}
            Sets per indiv.: {self.T}

            Number of non-traders:
            Fast-exp. alt.: {self.NT_FastExp}
            Slow-cheap alt.: {self.NT_CheapSlow}

            BVTT statistics:
            Mean chosen BVTT: {self.ChosenBVTT_Mean}
            Minimum of BVTT: {self.BVTT_min}
            Maximum of BVTT: {self.BVTT_max}
        """)

        return formatted_descriptives
