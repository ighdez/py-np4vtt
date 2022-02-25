# Import modules
import pandas as pd
from pathlib import Path

from model.data_format import ModelArrays, StudyVar
from model.model_rouwendal import ConfigRouwendal, ModelRouwendal
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

model_arrays = make_modelarrays(df,columnarrays)

# Step 2: Do descriptives
descriptives = compute_descriptives(model_arrays)

# Step 3: Make config
config = ConfigRouwendal(0,17,18,0.9)

# Step 4: Call model
rouwendal = ModelRouwendal(config,model_arrays)
initialArgs, initialVal = rouwendal.setupInitialArgs()
q_prob, q_est, par, fvtt, cumsum_fvtt, vtt_grid, fval, exitflag, output = rouwendal.run(initialArgs)

# Check if the model reached the expected results
f_initial_expected = 0. # TODO: Grab actual value from MATLAB code
f_final_expected = 23335.63
q_prob_expected = 0.90069

pass_f_initial = (f_initial_expected*0.9 < initialVal < f_initial_expected*1.1)
pass_f_final = (fval < f_final_expected*1.1)
pass_q = (q_prob_expected*0.99 < q_prob < q_prob_expected*1.01)

# TODO: check the initialValue
print('Rouwendal method checks:')
if not pass_f_initial:
    print('Initial F-value: too far from expected. Expected={f_initial_expected}, Actual={initialVal}')
else:
    print('Initial F-value: OK')

if not pass_f_final:
    print('Final F-value: too far from expected. Expected={f_final_expected}, Actual={fval}')
else:
    print('Final F-value: OK')

if not pass_q:
    print('Q Prob: too far from expected. Expected={q_prob_expected}, Actual={q_prob}')
else:
    print('Q Prob: OK')
