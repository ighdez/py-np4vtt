import pandas as pd
from pathlib import Path

from model.data_format import StudyVar
from model.model_logit import ModelLogit, ConfigLogit
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
config = ConfigLogit(mleIntercept=1, mleParameter=0.1, mleScale=1, mleMaxIterations=10000, seed=12345)

# Step 4: Call model
logit = ModelLogit(config, model_arrays)
initialArgs, initialVal = logit.setupInitialArgs()
x, fval, exitflag, output = logit.run(initialArgs)


# Check if the model reached the expected results
f_initial_expected = 0.  # TODO: Grab actual value from MATLAB code
f_final_expected = -2387.1224

# TODO: check the initialValue
print('Logistic regression checks:')
if not check_in_range(f_initial_expected, initialVal, margin_proportion=0.1):
    print(f'Initial F-value: FAIL too far from expected. Expected={f_initial_expected}, Actual={initialVal}')
else:
    print('Initial F-value: PASS')

if not check_in_range(f_final_expected, fval, margin_proportion=0.1):
    print(f'Final F-value: FAIL too far from expected. Expected={f_final_expected}, Actual={fval}')
else:
    print('Final F-value: PASS')
