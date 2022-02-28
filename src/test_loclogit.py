import pandas as pd
from pathlib import Path

from model.data_format import StudyVar
from model.model_loclogit import ModelLocLogit, ConfigLocLogit
from model.data_import import make_modelarrays, compute_descriptives

from helpers import check_in_range


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
df = pd.read_table(curscript_dir.parent / 'data' / 'Norway09_data_v5.txt')

model_arrays = make_modelarrays(df, columnarrays)

# Step 2: Do descriptives
descriptives = compute_descriptives(model_arrays)

# Step 3: Make config
config = ConfigLocLogit(minimum=0, maximum=18, supportPoints=19)

# Step 4: Call model
loclogit = ModelLocLogit(config, model_arrays)
initialArgs = loclogit.setupInitialArgs()
p, fval, vtt_grid = loclogit.run(initialArgs)


# Check if the model reached the expected results
f_final_expected = -27596.28

# Is it within an interval with margin?
if check_in_range(f_final_expected, fval, margin_proportion=0.1):
    print('Final F-value: PASS')
else:
    print(f'Final F-value: FAIL too far from expected. Expected={f_final_expected}, Actual={fval}')
