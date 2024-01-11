import sys
import os
from pathlib import Path

aux = os.path.join(Path(__file__).parents[1])

if str(aux) not in sys.path:
    sys.path.append(aux)

import argparse
import pickle
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

from utils.crossvalidation import xval_results

from settings.paths import rf_path, validation_path
from settings.columns import  create_colors,  aper

def parse_args():
    parser = argparse.ArgumentParser(description="Random Forest")
    parser.add_argument("--save_model", action="store_true", default=False)
    parser.add_argument("--save_result", action="store_true", default=False)
    args = parser.parse_args()
    return args

# List of features
feat = {}

flags = ["flag_WISE", "flag_GALEX"]

feat["broad"] =  create_colors(broad = True, narrow=False, wise=False, galex=False, aper=aper)

feat["broad+narrow"]=  create_colors(broad = True, narrow=True, wise=False, galex=False, aper=aper)

feat["broad+GALEX+WISE"]=  create_colors(broad = True, narrow=False, wise=True, galex=True, aper=aper)
feat["broad+GALEX+WISE+flags"]=  feat["broad+GALEX+WISE"]+flags

feat["broad+WISE+narrow"]=  create_colors(broad = True, narrow=True, wise=False, galex=True, aper=aper)
feat["broad+WISE+narrow+flags"]=  feat["broad+WISE+narrow"]+["flag_WISE"]

feat["broad+GALEX+WISE+narrow"]=  create_colors(broad = True, narrow=True, wise=True, galex=True, aper=aper)
feat["broad+GALEX+WISE+narrow+flags"]=  feat["broad+GALEX+WISE+narrow"]+flags

# Read data
train = pd.read_csv(os.path.join(validation_path,"train.csv"), index_col="index")
test = pd.read_csv(os.path.join(validation_path,"test.csv"), index_col="index") 


# Fine tuning 
    
run_gridsearch = False

if run_gridsearch == False:
    dict_gridsearch = {}
    dict_gridsearch["broad+GALEX+WISE+narrow+flags"] = pickle.load(open(os.path.join(rf_path,'GridSearch_broad+GALEX+WISE+narrow+flags.sav'), 'rb')).best_params_
    dict_gridsearch["broad+GALEX+WISE+flags"] = pickle.load(open(os.path.join(rf_path,'GridSearch_broad+GALEX+WISE+flags.sav'), 'rb')).best_params_

else:
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
    CV.fit(train[feat["broad+GALEX+WISE+narrow+flags"]], train.Zclass)
    filename = os.path.join(rf_path,'GridSearch_broad+GALEX+WISE+narrow+flags.sav')
    pickle.dump(CV, open(filename, 'wb'))

    # CV = GridSearchCV(estimator=model, param_grid=param_grid, cv=skf)
    # CV.fit(train[feat["broad+GALEX+WISE+flags"]], train.Zclass)
    # pickle.dump(CV, open(filename, 'wb'))


# Crossvalidation
run_crossvalidation = True
save_model = False
save_result = True
z={}

if run_crossvalidation:
    for key, value in feat.items():
        print(key)
        z[key] = xval_results(feat[key], key, dict_gridsearch["broad+GALEX+WISE+narrow+flags"], save_model=save_model, save_result=save_result)

# Test
        
# save_model = False
# save_result = True

# mag = pd.DataFrame(test["r_"+aper], columns=["r_"+aper])
# mag["g_"+aper] = test["g_"+aper]
# mag["g-r"] = mag["g_"+aper] - mag["r_"+aper]

# for key in ["broad+GALEX+WISE+narrow+flags", "broad+GALEX+WISE+flags"]:
#     print(key)
#     model = RandomForestRegressor(**dict_gridsearch[key])
#     model.fit(train[feat[key]], train.Z)

#     if save_model:
#         filename = os.path.join(rf_path,'RF_train_'+key+'.sav')
#         pickle.dump(model, open(filename, 'wb'))

#     z_p = pd.DataFrame(model.predict(test[feat[key]]), index=test.index, columns=["z_pred"])
#     z_test = pd.DataFrame(test.Z)
#     z = pd.concat([z_p,z_test, mag], axis = 1)
#     if save_result:
#         z.to_csv(os.path.join(rf_path,"test_z_"+key+".csv"))

# Final model for production:
        
# model = RandomForestRegressor(**dict_gridsearch["broad+GALEX+WISE+narrow+flags"])
# spec = pd.concat([train, test])
# model.fit(spec[feat["broad+GALEX+WISE+narrow+flags"]], spec.Z)
# filename = os.path.join(rf_path,'RF_final_'+key+'.sav')
# pickle.dump(model, open(filename, 'wb'))