
import pandas as pd
import numpy as np
import os
import pickle
from utils.preprocessing import create_bins
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
from settings.paths import rf_path, validation_path
from settings.columns import specz


def count_bins(z, itvs):
    n = []
    # for i in sorted(itvs.unique().dropna()):
    for i in range(len(itvs.unique().value_counts())):
        z_cut = z[specz][itvs==i]
        try:
            n.append(len(z_cut))
        except:
            n.append(np.nan)
    return n


def metric_per_bin(metric, z:pd.core.frame.DataFrame, itvs:pd.core.frame.Series, column="z_pred"): 
    metrics_bin = []
    # for i in sorted(itvs.unique().dropna()):
    for i in range(len(itvs.unique().value_counts())):
        zp_cut = z[column][itvs == i]
        z_cut = z[specz][itvs == i]
        # try:
        metric_bin = metric(z_cut, zp_cut)
        # except:
        #     metric_bin = np.nan
        metrics_bin.append(metric_bin)
    return np.array(metrics_bin)


def xval(train, zclass_train, feat, filename, aper, save_data=False, save_model=False, save_result=True):
    mag_train_cv = {}
    mag_val_cv = {}
    z_train_cv = {}
    z_val_cv = {}

    i=0
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=47)
    for train_index, val_index in skf.split(train, zclass_train):
        mag_train_cv["fold"+str(i)], mag_val_cv["fold"+str(i)] = train.iloc[train_index], train.iloc[val_index]
        z_train_cv["fold"+str(i)], z_val_cv["fold"+str(i)] = train.iloc[train_index].Z, train.iloc[val_index].Z
        if save_data:
            mag_train_cv["fold"+str(i)].index.name = "index"
            mag_val_cv["fold"+str(i)].index.name = "index"
            mag_train_cv["fold"+str(i)].to_csv(os.path.join(validation_path,"trainf"+str(i)+".csv"), sep=",")
            mag_val_cv["fold"+str(i)].to_csv(os.path.join(validation_path,"valf"+str(i)+".csv"), sep=",")
    
        i=i+1  
    z = pd.DataFrame()

    i=0
    for fold in mag_train_cv:
        mag = pd.DataFrame(mag_val_cv[fold]["r_"+aper], columns={"r_"+aper})

        model = RandomForestRegressor(random_state = 47, bootstrap= True, max_depth= 20, min_samples_leaf= 2, min_samples_split= 2, n_estimators= 400)
        # model = RandomForestRegressor(random_state = 47)
        model.fit(mag_train_cv[fold][feat], z_train_cv[fold])
        if save_model:
            file = os.path.join(rf_path,'RF_'+filename+'fold_'+str(i)+'.sav')
            pickle.dump(model, open(file, 'wb'))
    
        z_p = pd.DataFrame(model.predict(mag_val_cv[fold][feat]), index=mag_val_cv[fold].index, columns=["z_pred"])
        z_val = pd.DataFrame(z_val_cv[fold])
        
        z_aux = pd.concat([z_p,z_val, mag], axis = 1)
        z_aux["fold"] = i
        z = pd.concat([z,z_aux], axis = 0)
        i=i+1

    if save_result:
        z.to_csv(os.path.join(rf_path,"z_"+filename+".csv"), index=True)
    return z


def calculate_single(model, metric, rmax = None, rmin = None, zmax = None, zmin=None, aper="PStotal"):
    df = pd.read_table(os.path.join(rf_path,"z_"+model), index_col="index", sep=',')
    output = []
    if rmax:
        df = df.query('r_'+aper+'<='+str(rmax))
    if rmin:
        df = df.query('r_'+aper+'>'+str(rmin))

    if zmax:
        df = df.query(specz+'<='+str(zmax))
    if zmin:
        df = df.query(specz+'>'+str(zmin))

    for i in [0,1,2,3,4]:
        aux = df.query('fold == '+str(i))
        output.append(metric(aux.z_pred,aux[specz]))
    # print(metric.__name__, np.round(np.mean(output)*100,2), np.round(np.std(output)*100,2))

    return np.mean(output), np.std(output)


def print_metrics(z, cutoff=None, xval=True):
    rmse_list = []
    sigma_list = []
    bias_list = []
    n30_list = []
    n15_list = []

    if cutoff>0:
        z = z.query("Z<=5")

    if xval:
        for fold in z.fold.unique():
            rmse_list.append(rmse(z[z["fold"]==fold].Z, z[z["fold"]==fold].z_pred))
            sigma_list.append(sigma(z[z["fold"]==fold].Z, z[z["fold"]==fold].z_pred))
            bias_list.append(bias(z[z["fold"]==fold].Z, z[z["fold"]==fold].z_pred))
            n30_list.append(out_frac(z[z["fold"]==fold].Z, z[z["fold"]==fold].z_pred, 0.3))
            n15_list.append(out_frac(z[z["fold"]==fold].Z, z[z["fold"]==fold].z_pred, 0.15))
        print('RMSE', np.round(np.mean(rmse_list),4), np.round(np.std(rmse_list),4))
        print('NMAD', np.round(np.mean(sigma_list),4),  np.round(np.std(sigma_list),4))
        print('bias', np.round(np.mean(bias_list),4),  np.round(np.std(bias_list),4))
        print('n30', np.round(np.mean(n30_list),4),  np.round(np.std(n30_list),4))
        print('n15', np.round(np.mean(n15_list),4),  np.round(np.std(n15_list),4))
    else:
        print('RMSE', np.round(rmse(z_aux.Z,z_aux.z_pred),4), np.round(rmse(z_aux.Z,z_aux.z_pred),4))
        print('NMAD', np.round(sigma(z_aux.Z,z_aux.z_pred),4),  np.round(sigma(z_aux.Z,z_aux.z_pred),4))
        print('bias', np.round(bias(z_aux.Z,z_aux.z_pred),4),  np.round(bias(z_aux.Z,z_aux.z_pred),4))
        print('n30', np.round(out_frac(z_aux.Z,z_aux.z_pred, 0.30),4),  np.round(out_frac(z_aux.Z,z_aux.z_pred, 0.30),4))
        print('n15', np.round(out_frac(z_aux.Z,z_aux.z_pred, 0.15),4),  np.round(out_frac(z_aux.Z,z_aux.z_pred,0.15),4))
        
    return
