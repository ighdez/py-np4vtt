#  Copyright 2021 Technische Universiteit Delft
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
import numpy as np
from random import uniform

# Create VTT midpoints
def vtt_midpoints(vtt_grid):
    
    # Calculate midpoints
    vtt_grid_midpoints = ((np.append(vtt_grid, 0) + np.append(0,vtt_grid)))/2
    
    # Replace last value of the VTT midpoints by the last value of the VTT array
    vtt_grid_midpoints[-1] = vtt_grid[-1]

    return vtt_grid_midpoints

# Create predicted VTT for each respondent, based in the choice probability and a VTT grid.
def predicted_vtt(ecdf,grid,NP):
    pp = ((np.append(ecdf, 0)-np.append(0,ecdf)))
    count_data = []
    for n in range(1,len(grid)):
        dat = [uniform(grid[n-1], grid[n]) for p in range(0, np.round((pp[n]*NP)).astype(int))]
        count_data = np.append(count_data,dat)
    return count_data