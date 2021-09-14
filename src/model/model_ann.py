#  Copyright 2021 Technische Universiteit Delft
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ConfigANN:
    hiddenLayerNodes: List[int]

    trainingRepeats: int
    shufflesPerRepeat: int

    seed: Optional[int]

    def validate(self):

        # Create errormessage list
        errorMessage = []

        # Number of hidden nodes must be positive
        isHiddenLayersPositive = (self.hiddenLayerNodes > 0)

        if not isHiddenLayersPositive:
            errorMessage.append('Number of hidden nodes must be positive.')

        # Number of repeats must be positive
        isPositiveRepeats = (self.isPositiveRepeats > 0)

        if not isPositiveRepeats:
            errorMessage.append('Number of repeats must be positive.')

        # Number of shuffles per repeats must be positive
        isPositiveShuffles = (self.isPositiveShuffles > 0)

        if not isPositiveShuffles:
            errorMessage.append('Number of shuffles per repeats must be positive.')

        # Create integrity check list
        integrityCheckList = [isHiddenLayersPositive,isPositiveRepeats,isPositiveShuffles]

        # Test if all statements are true
        integrityCheck = all(integrityCheckList)

        # Return True if all OK, otherwise return False and a message.
        return(integrityCheck, errorMessage)
