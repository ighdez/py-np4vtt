# Import modules
from model.data_format import ModelArrays, StudyVar
from model.model_logit import ModelLogit, ConfigLogit
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

# Step 2: Make config
config = ConfigLogit(1,0.1,1,10000,12345)

# Step 3: Call model
logit = ModelLogit(config,model_arrays)

x, fval, exitflag, output = logit.run()

print(x, fval, exitflag)
print(output)