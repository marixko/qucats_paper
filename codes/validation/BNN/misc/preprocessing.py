import os
from itertools import combinations
from warnings import simplefilter

import numpy as np
import pandas as pd

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler
import matplotlib.pyplot as plt

from utils.preprocessing import mag_redshift_selection, prep_wise, create_bins
from utils.correct_extinction import correction
from settings.columns import (specz, broad, narrow, splus, wise, galex,
                              list_feat, create_colors, calculate_colors, create_ratio, calculate_ratio)
from settings.paths import match_path as data_path, bnn_path as results_path


def select_magnitudes(Aper:str, magnitudes:list, prt=True) -> tuple:
    
    dir_list = []
    
    if 'broad' in magnitudes and 'narrow' in magnitudes:
        SPLUS = ['u', 'J0378', 'J0395', 'J0410', 'J0430', 'g', 'J0515', 'r', 'J0660', 'i', 'J0861', 'z']
        dir_list.append('Sbn')
    
    elif 'broad' in magnitudes:
        SPLUS = ['u', 'g', 'r', 'i', 'z']
        dir_list.append('Sb')
    
    elif 'narrow' in magnitudes:
        SPLUS = ['J0378', 'J0395', 'J0410', 'J0430', 'J0515', 'J0660', 'J0861']
        dir_list.append('Sn')
        
    if 'wise' in magnitudes:
        Magnitudes_WISE = ['W1','W2']
        dir_list.append('W')
    else:
        Magnitudes_WISE = []
    
    if 'galex' in magnitudes:
        Magnitudes_GALEX = ['FUVmag', 'NUVmag']
        dir_list.append('G')
    else:
        Magnitudes_GALEX = []
    
    Magnitudes_SPLUS = [item+'_'+Aper for item in SPLUS]
    Errors_SPLUS = ['e_'+item for item in Magnitudes_SPLUS]
    
    if prt: print('# Magnitudes:\n#', magnitudes)
    
    dir_str = ' + '.join(dir_list)
    Output_Dir = os.path.join(results_path, dir_str)
    
    return Magnitudes_SPLUS, Magnitudes_WISE, Magnitudes_GALEX, Errors_SPLUS, Output_Dir


def calc_colors(Dataframe, Aper:str, Magnitudes:list) -> list:

    Reference_Mag  = 'r_'+Aper
    Reference_Idx = Magnitudes.index(Reference_Mag)
    MagnitudesToLeft = Magnitudes[:Reference_Idx]
    MagnitudesToRight = Magnitudes[(Reference_Idx+1):]

    for feature in MagnitudesToLeft: # of Reference_Mag
        Dataframe[feature+'-'+Reference_Mag] = Dataframe[feature] - Dataframe[Reference_Mag]

    for feature in MagnitudesToRight: # of Reference_Mag
        Dataframe[Reference_Mag+'-'+feature] = Dataframe[Reference_Mag] - Dataframe[feature]
    
    Colors = []
    for s in Dataframe.columns.values:
        if '-' in s:
            if s.split('-')[0] in Magnitudes and s.split('-')[1] in Magnitudes:
                Colors.append(s)
    
    return Colors


def calc_ratios(Dataframe, Features:list) -> list:
    
    combs = list(combinations(Features, 2))
    Ratios = []
    
    for comb in combs:
        name = comb[0] + '/' + comb[1]
        Dataframe[name] = Dataframe[comb[0]] / Dataframe[comb[1]]
        Ratios.append(name)

    return Ratios


def split(dataframe, test_frac:float, seed:int):
    
    print('# Dataframe Size:', len(dataframe))
    
    if test_frac == 0:
        train_sample = dataframe
        test_sample = pd.DataFrame()

    else:
        dataframe, _, _ = create_bins(data=dataframe, bin_size=0.5, return_data=True, var='Z')
        train_sample, test_sample = train_test_split(
            dataframe, test_size=test_frac, stratify=dataframe['Zclass'], random_state=seed)   
        
    print('# Train Size:    ', len(train_sample))
    print('# Test Size:     ', len(test_sample))

    return train_sample, test_sample


def plot_features(feature_list:list, scaled_train, output_dir:str):
    
    fig, _ = plt.subplots(figsize=(20, 20))
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    features_to_plot = feature_list

    plt_idx = 1
    for feature in features_to_plot:
        plt.subplot(10, 6, plt_idx)
        plt.hist(scaled_train[feature], lw=2, bins=20, histtype='step')
        plt.yscale('log')
        plt.xlabel(feature)
        plt.grid(lw=.5)
        plt_idx += 1

    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'features.png'), bbox_inches='tight')
    plt.close()


def Process_Split_old(filename:str, mags:list, configs:dict, test_frac:float, seed:int, output_dir:str,
                  aper='PStotal', save_df=True):
    pd.options.mode.chained_assignment = None  # default='warn'
    
    if 'broad' in mags and 'narrow' in mags:
        splus = ['u', 'J0378', 'J0395', 'J0410', 'J0430', 'g', 'J0515', 'r', 'J0660', 'i', 'J0861', 'z']
    elif 'broad' in mags:
        splus = ['u', 'g', 'r', 'i', 'z']
    elif 'narrow' in mags:
        splus = ['J0378', 'J0395', 'J0410', 'J0430', 'J0515', 'J0660', 'J0861']
    Magnitudes_WISE = ['W1_MAG','W2_MAG'] if 'wise' in mags else []
    Magnitudes_GALEX = ['FUVmag', 'NUVmag'] if 'galex' in mags else []

    Base_Columns = ['ID', 'RA_1', 'DEC_1']
    Magnitudes_SPLUS = [item+'_'+aper for item in splus]
    Errors_SPLUS = ['e_'+item for item in Magnitudes_SPLUS]
    magnitudes = Magnitudes_GALEX + Magnitudes_SPLUS + Magnitudes_WISE  # It's important to keep this order
    
    file_path = os.path.join(data_path, filename)
    cols = Base_Columns + magnitudes + Errors_SPLUS + ['Z']
    if 'broad' not in mags:
        cols.append('r_'+aper)
    data = pd.read_csv(file_path, usecols=cols)
    
    data = mag_redshift_selection(data, rmax=22, zmax=7)  # cuts
    #data = prep_wise(data)  # wise flux to magnitude

    # Non detected/observed objects (splus 99 and unwise -1)
    for mag in magnitudes:
        data[mag][~data[mag].between(10, 50)] = np.nan
    
    # Extinction correction
    data = correction(data)

    # Replace S-PLUS missing features with the upper magnitude limit (the value in the error column)
    for mag, error in zip(Magnitudes_SPLUS, Errors_SPLUS):
        data[mag].fillna(data[error], inplace=True)
    
    feature_list = []
    
    if configs['mag']:
        feature_list += magnitudes

    if configs['col']:

        if 'broad' not in mags:
            # Inserting r band in the correct position only to calculate colors
            j0660_idx = magnitudes.index('J0660_'+aper)
            colors_magnitudes = magnitudes[:j0660_idx] + ['r_'+aper] + magnitudes[j0660_idx:]
        else:
            colors_magnitudes = magnitudes
        
        Colors = calc_colors(data, aper, colors_magnitudes)
        feature_list += Colors
    
    if configs['rat']:
        Ratios = calc_ratios(data, magnitudes)
        feature_list += Ratios
    
    print(f'# {len(feature_list)} Features:\n{feature_list}')
    
    train_sample, test_sample = split(data, test_frac, seed)
    train_sample['weights'] = 1
    
    if save_df: data.to_csv(os.path.join(output_dir, 'dataframe.csv'), index=False)
    
    scaler_1 = QuantileTransformer(output_distribution='normal')
    scaler_2 = MinMaxScaler((0, 1))
    scaled_train = scaler_1.fit_transform(train_sample[feature_list])
    scaled_train = scaler_2.fit_transform(scaled_train)
    scaled_train = pd.DataFrame(scaled_train, columns=feature_list)
    plot_features(feature_list, scaled_train, output_dir)
    
    train_mask = train_sample[feature_list].isna().reset_index(drop=True)
        
    if test_frac != 0:
        test_mask = test_sample[feature_list].isna().reset_index(drop=True)
        return train_sample, test_sample, feature_list, scaler_1, scaler_2, train_mask, test_mask
    
    if test_frac == 0:
        return train_sample, feature_list, scaler_1, scaler_2, train_mask


def Process_Split(filename:str, mags:list, configs:dict, test_frac:float, seed:int, output_dir:str,
                  aper='PStotal', save_df=True):
    pd.options.mode.chained_assignment = None  # default='warn'
    
    # List with magnitude names    
    magnitudes = []
    broad_bool, narrow_bool, wise_bool, galex_bool = False, False, False, False
    
    if 'broad' in mags and 'narrow' in mags:
        broad_bool, narrow_bool = True, True
        magnitudes.extend(splus)
    elif 'broad' in mags:
        broad_bool = True
        magnitudes.extend(broad)
    elif 'narrow' in mags:
        narrow_bool = True
        magnitudes.extend(narrow)
    magnitudes_splus = magnitudes.copy()
    if 'wise' in mags:
        wise_bool = True
        magnitudes.extend(wise)
    if 'galex' in mags:
        galex_bool = True
        magnitudes.extend(galex)
    
    base_columns = ['ID', 'RA_1', 'DEC_1']
    errors_splus = ['e_'+item for item in magnitudes_splus]
    magnitudes_r = magnitudes
    if 'broad' not in mags:
        magnitudes_r += ['r_'+aper]
    
    # Reading data
    file_path = os.path.join(data_path, filename)
    cols = base_columns + magnitudes_r + errors_splus + [specz]
    data = pd.read_csv(file_path, usecols=cols)
    
    # Preparing data
    data = mag_redshift_selection(data, rmax=22, zmax=7)  # cuts
    #data = prep_wise(data)  # wise flux to magnitude

    # Non detected/observed objects (splus 99 and unwise -1)
    for mag in magnitudes:
        data[mag][~data[mag].between(10, 50)] = np.nan
    
    # Extinction correction
    data = correction(data, magnitudes_r)

    # Replace S-PLUS missing features with the upper magnitude limit (the value in the error column)
    for mag, error in zip(magnitudes_splus, errors_splus):
        data[mag].fillna(data[error], inplace=True)
    
    # Defining features    
    feature_list = []
    
    if configs['mag']:
        feature_list += magnitudes

    if configs['col']:
        feature_list += create_colors(broad_bool, narrow_bool, wise_bool, galex_bool, aper)
        data = calculate_colors(data, broad_bool, narrow_bool, wise_bool, galex_bool, aper)
        
    if configs['rat']:
        feature_list += create_ratio(broad_bool, narrow_bool, wise_bool, galex_bool, aper)
        data = calculate_ratio(data, broad_bool, narrow_bool, wise_bool, galex_bool, aper)
    
    feature_list = ['FUVmag-r_PStotal', 'NUVmag-r_PStotal', 'u_PStotal-r_PStotal', 'J0378_PStotal-r_PStotal',
                    'J0395_PStotal-r_PStotal', 'J0410_PStotal-r_PStotal', 'J0430_PStotal-r_PStotal',
                    'g_PStotal-r_PStotal', 'J0515_PStotal-r_PStotal', 'r_PStotal-J0660_PStotal', 'r_PStotal-i_PStotal',
                    'r_PStotal-J0861_PStotal', 'r_PStotal-z_PStotal', 'r_PStotal-W1_MAG', 'r_PStotal-W2_MAG']
    print(f'# {len(feature_list)} Features:\n{feature_list}')
    
    # Splitting sample
    train_sample, test_sample = split(data, test_frac, seed)
    train_sample['weights'] = 1
    
    if save_df: data.to_csv(os.path.join(output_dir, 'dataframe.csv'), index=False)
    
    # Fitting scalars    
    scaler_1 = QuantileTransformer(output_distribution='normal')
    scaler_2 = MinMaxScaler((0, 1))
    scaled_train = scaler_1.fit_transform(train_sample[feature_list])
    scaled_train = scaler_2.fit_transform(scaled_train)
    scaled_train = pd.DataFrame(scaled_train, columns=feature_list)
    plot_features(feature_list, scaled_train, output_dir)
    
    # Missing band mask
    train_mask = train_sample[feature_list].isna().reset_index(drop=True)
        
    if test_frac != 0:
        test_mask = test_sample[feature_list].isna().reset_index(drop=True)
        return train_sample, test_sample, feature_list, scaler_1, scaler_2, train_mask, test_mask
    
    if test_frac == 0:
        return train_sample, feature_list, scaler_1, scaler_2, train_mask


def Process_Final(dataframe, mags:list, configs:dict, aper='PStotal'):
    '''CHECK LATER'''
    
    if 'broad' in mags and 'narrow' in mags:
        splus = ['u', 'J0378', 'J0395', 'J0410', 'J0430', 'g', 'J0515', 'r', 'J0660', 'i', 'J0861', 'z']
    elif 'broad' in mags:
        splus = ['u', 'g', 'r', 'i', 'z']
    elif 'narrow' in mags:
        splus = ['J0378', 'J0395', 'J0410', 'J0430', 'J0515', 'J0660', 'J0861']
    Magnitudes_WISE = ['W1_MAG','W2_MAG'] if 'wise' in mags else []
    Magnitudes_GALEX = ['FUVmag', 'NUVmag'] if 'galex' in mags else []

    Magnitudes_SPLUS = [item+'_'+aper for item in splus]
    Errors_SPLUS = ['e_'+item for item in Magnitudes_SPLUS]
    magnitudes = Magnitudes_GALEX + Magnitudes_SPLUS + Magnitudes_WISE  # It's important to keep this order
    
    # Non detected/observed objects
    for mag in magnitudes:
        dataframe[mag][~dataframe[mag].between(10, 50)] = np.nan

    # Replace S-PLUS missing features with the upper magnitude limit (the value in the error column)
    for mag, error in zip(Magnitudes_SPLUS, Errors_SPLUS):
        dataframe[mag].fillna(dataframe[error], inplace=True)
    
    feature_list = []
    
    if configs['mag']:
        feature_list += magnitudes

    if configs['col']:
        if 'broad' not in mags:
            # Inserting r band in the correct position only to calculate colors
            j0660_idx = magnitudes.index('J0660_'+aper)
            colors_magnitudes = magnitudes[:j0660_idx] + ['r_'+aper] + magnitudes[j0660_idx:]
        else:
            colors_magnitudes = magnitudes
        Colors = calc_colors(dataframe, aper, colors_magnitudes)
        feature_list += Colors
    
    if configs['rat']:
        Ratios = calc_ratios(dataframe, magnitudes)
        feature_list += Ratios
    
    # Mask
    mask = dataframe[feature_list].isna().reset_index(drop=True)

    return dataframe, feature_list, mask
