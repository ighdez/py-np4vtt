#  Copyright 2021 Technische Universiteit Delft
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
from dataclasses import dataclass
from typing import Optional


@dataclass
class ConfigLogit:
    mleIntercept: float
    mleParameter: float
    mleScale: float

    mleMaxIterations: int

    seed: Optional[int]

    def validate(self):

        # Create errormessage list
        errorMessage = []

        # Scale starting value must be positive.
        isPositiveScale = (self.mleScale > 0)

        if not isPositiveScale:
            errorMessage.append('Scale starting value must be positive.')

        # Max iterations must be greater than zero
        isPositiveMaxIter = (self.mleMaxIterations > 0)

        if not isPositiveMaxIter:
            errorMessage.append('Max iterations must be greater than zero.')

        # Seed must be non-negative
        isNonNegativeSeed = (self.seed >= 0)

        if not isNonNegativeSeed:
            errorMessage.append('Seed must be non-negative.')

        # Create integrity check list
        integrityCheckList = [isPositiveScale,isPositiveMaxIter,isNonNegativeSeed]

        # Test if all statements are true
        integrityCheck = all(integrityCheckList)

        # Return True if all OK, otherwise return False and a message.
        return(integrityCheck, errorMessage)