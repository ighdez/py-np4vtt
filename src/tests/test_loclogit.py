# Import modules
import pandas as pd
from pathlib import Path

from model.data_format import StudyVar
from model.model_loclogit import ModelLocLogit, ConfigLocLogit
from model.data_import import make_modelarrays, compute_descriptives


# Step 1: read CSV file
columnarrays = {
    StudyVar.Id: 'RespID',
    StudyVar.ChosenAlt: 'Chosen',
    StudyVar.Cost1: 'CostL',
    StudyVar.Cost2: 'CostR',
    StudyVar.Time1: 'TimeL',
    StudyVar.Time2: 'TimeR',
}

curscript_dir = Path(__file__).resolve().parent
df = pd.read_table(curscript_dir.parent.parent / 'data' / 'Norway09_data_v5.txt')

model_arrays = make_modelarrays(df, columnarrays)

# Step 2: Do descriptives
descriptives = compute_descriptives(model_arrays)

# Step 3: Make config
config = ConfigLocLogit(0, 18, 19)

# Step 4: Call model
loclogit = ModelLocLogit(config,model_arrays)

p, fval, vtt_grid = loclogit.run()

# Check if the model reached the expected results
f_final_expected = 27596.28
pass_f = (fval < f_final_expected*1.1)

if pass_f:
    print('Check!')
else:
    print('F-value too far from expected.')