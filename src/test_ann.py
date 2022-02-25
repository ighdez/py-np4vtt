# Import modules
import pandas as pd
from pathlib import Path

from model.data_format import StudyVar
from model.model_ann import ModelANN, ConfigANN
from model.data_import import make_modelarrays, compute_descriptives

from helpers import check_in_range


# Step 1: read CSV file
columnarrays = {StudyVar.Id: 'RespID',
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
config = ConfigANN([10,10],20,50,None)

# Step 4: Call model
ann = ModelANN(config,model_arrays)
initialArgs = ann.setupInitialArgs()
ll, r2, clf, vtt = ann.run(initialArgs)

# Export to excel
# Check if the model reached the expected results
ll_expected = 107864.1944
r2_expected = 0.466340404

ll_mean = -ll.mean()
r2_mean = r2.mean()

pass_ll = (ll_expected*0.85 < ll_mean < ll_expected*1.15)
pass_r2 = (r2_expected*0.9 < r2_mean < r2_expected*1.1)

print('ANN checks:')
if not pass_ll:
    print('F-value: too far from expected.')
else:
    print('F-value: OK')

if not pass_r2:
    print('Rho-squared:   too far from expected.')
else:
    print('Rho-squared:   OK')