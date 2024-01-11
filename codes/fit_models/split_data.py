import sys
import os
from pathlib import Path
aux = os.path.join(Path(__file__).parents[1])
if str(aux) not in sys.path:
    sys.path.append(aux)
import os
import pandas as pd 

from utils.preprocessing import rename_aper, preprocessing, split_data

from utils.crossvalidation import  save_folds

from settings.paths import match_path, validation_path

# Read data
table = pd.read_table(os.path.join(match_path,"STRIPE82_DR4_DR16Q1a_unWISE2a_GALEXDR672a.csv"), sep=",")
table = rename_aper(table)

# Preprocessing
table = preprocessing(table)

# Split samples
train, test = split_data(table)

# Save cross-validation folds (k = 5)
save_folds(train, train.Zclass)

# Save test set
test.index.name = 'index'
test.to_csv(os.path.join(validation_path, "test.csv"))

# Save train set
train.index.name = 'index'
train.to_csv(os.path.join(validation_path, "train.csv"))