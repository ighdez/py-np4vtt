# Import modules
from model.data_format import ModelArrays, StudyVar
from model.model_rouwendal import ConfigRouwendal, ModelRouwendal
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
config = ConfigRouwendal(0,17,18,0.9)

# Step 4: Call model
rouwendal = ModelRouwendal(config,model_arrays)

q_prob, q_est, par, fvtt, cumsum_fvtt, vtt_grid, fval, exitflag, output = rouwendal.run()

# Check if the model reached the expected results
f_final_expected = 23335.63
q_prob_expected = 0.90069

pass_f = (fval < f_final_expected*1.1)
pass_q = (q_prob < q_prob_expected*1.01) | (q_prob > q_prob_expected*0.99)

if (pass_f & pass_q):
    print('Check!')
elif not pass_f:
    print('F-value too far from expected.')
elif not pass_q:
    print('Q prob too far from expected.')
