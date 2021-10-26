# Import modules
import pandas as pd
from pathlib import Path

from model.data_format import StudyVar
from model.model_logit import ModelLogit, ConfigLogit
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
config = ConfigLogit(1, 0.1, 1, 10000, 12345)

# Step 4: Call model
logit = ModelLogit(config, model_arrays)

x, fval, exitflag, output = logit.run()

print(x, fval, exitflag)
print(output)
