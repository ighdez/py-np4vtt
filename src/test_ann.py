# Import modules
import pandas as pd
from pathlib import Path

from model.data_format import StudyVar
from model.model_ann import ModelANN, ConfigANN
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

# curscript_dir = Path(__file__).resolve().parent
df = pd.read_table('../data/Norway09_data_v5.txt')
#pd.read_table(curscript_dir.parent.parent / 'data' / 'Norway09_data_v5.txt')

model_arrays = make_modelarrays(df, columnarrays)

# Step 2: Do descriptives
descriptives = compute_descriptives(model_arrays)

# Step 3: Make config
config = ConfigANN([10,10],20,3,None)

# Step 4: Call model
ann = ModelANN(config,model_arrays)
initialArgs = ann.setupInitialArgs()
# ll, r2, y_predict, clf = ann.run(initialArgs)
ll, r2, clf = ann.run(initialArgs)
