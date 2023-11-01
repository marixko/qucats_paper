from pathlib import Path
import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split    
from sklearn.model_selection import StratifiedKFold
from utils.crossvalidation import xval
from settings.paths import rf_path
from sklearn.ensemble import RandomForestRegressor

aux = os.path.join(Path.cwd(), "codes")
if str(aux) not in sys.path:
    sys.path.append(aux)

from settings.paths import match_path, validation_path
from settings.columns import list_feat, create_colors, calculate_colors, specz
from utils.preprocessing import rename_aper, prep_data, create_bins
from utils.correct_extinction import correction

# Read data
table = pd.read_table(os.path.join(match_path,"STRIPE82_DR2_DR16Q1a_unWISE2a_GALEXDR672a.csv"), sep=",")
table = rename_aper(table)
aper = "PStotal"
feat_mag = list_feat(aper=aper, broad = True, narrow = True, galex = False, wise = False)
# feat = create_colors(broad = True, narrow=True, wise=True, galex=True, aper=aper)
data = table.copy(deep=True)

# Preprocessing data
data = prep_data(data, rmax = 22, rmin = "None", zmax = 7, zmin = 0, val_mb = 99, aper=aper)
data = calculate_colors(data, broad = True, narrow= True, wise = True, galex= True, aper=aper)
data, bins, itvs = create_bins(data = data, bin_size=0.5, return_data = True, var = specz)

# Train/test split
train, test = train_test_split(data, random_state=47, stratify=data['Zclass'])
train = correction(train)
test = correction(test)
zclass_train = train["Zclass"]
# test.to_csv(os.path.join(validation_path,"test_orig.csv"), sep=",")    

# List of features
feat = {}
feat["broad"] =  create_colors(broad = True, narrow=False, wise=False, galex=False, aper=aper)
feat["broad+narrow"]=  create_colors(broad = True, narrow=True, wise=False, galex=False, aper=aper)
feat["broad+GALEX+WISE"]=  create_colors(broad = True, narrow=False, wise=True, galex=True, aper=aper)
feat["broad+GALEX+WISE+narrow"]=  create_colors(broad = True, narrow=True, wise=True, galex=True, aper=aper)

# Fine tuning 

param_grid = { 
    'n_estimators': [100, 200, 400],
    'bootstrap': [True, False],
    'max_depth' : [10,20,30, None],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10]
}
model = RandomForestRegressor(random_state = 47)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=47)
CV = GridSearchCV(estimator=model, param_grid=param_grid, cv=skf)
CV.fit(train[feat["broad+GALEX+WISE+narrow"]], zclass_train)
print(CV.best_params_)


# Crossvalidation
save_data = False
save_model = False
save_result = False
for key, value in feat.items():
    xval(train, zclass_train, feat[key], key, aper, save_data=save_data, save_model=save_model, save_result=save_result)

# Test
save_model = True
save_result = True

model = RandomForestRegressor(random_state = 47, bootstrap= True, max_depth= 20, min_samples_leaf= 2, min_samples_split= 2, n_estimators= 400)
model.fit(train[feat["broad+GALEX+WISE+narrow"]],train.Z)
if save_model:
    filename = os.path.join(rf_path,'RF_train.sav')
    pickle.dump(model, open(filename, 'wb'))
mag = pd.DataFrame(test["r_"+aper], columns={"r_"+aper})
mag["g_"+aper] = test["g_"+aper]
mag["g-r"] = mag["g_"+aper] - mag["r_"+aper]
z_p = pd.DataFrame(model.predict(test[feat["broad+GALEX+WISE+narrow"]]), index=test.index, columns=["z_pred"])
z_test = pd.DataFrame(test.Z)
z = pd.concat([z_p,z_test, mag], axis = 1)
if save_result:
    z.to_csv(os.path.join(rf_path,"test_z_broad+GALEX+WISE+narrow.csv"))
