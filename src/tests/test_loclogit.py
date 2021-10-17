# Import modules
from model.data_format import ModelArrays, StudyVar
from model.model_loclogit import ModelLocLogit, ConfigLocLogit
from model.data_import import make_modelarrays, compute_descriptives

import pandas as pd

# Step 1: read CSV file
columnarrays = {StudyVar.Id: 'RespID',
                StudyVar.ChosenAlt: 'Chosen',
                StudyVar.Cost1: 'CostL',
                StudyVar.Cost2: 'CostR',
                StudyVar.Time1: 'TimeL',
                StudyVar.Time2: 'TimeR',
}

df = pd.read_table('../data/Norway09_data_v5.txt')

model_arrays = make_modelarrays(df,columnarrays)

# Step 2: Do descriptives
descriptives = compute_descriptives(model_arrays)

# Step 3: Make config
config = ConfigLocLogit(0,18,19)

# Step 4: Call model
loclogit = ModelLocLogit(config,model_arrays)

p, fval, vtt_grid = loclogit.run()

print(p)
print(fval)
print(vtt_grid)