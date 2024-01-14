import os
from warnings import simplefilter

import pandas as pd; simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler
import matplotlib.pyplot as plt

from utils.preprocessing import mag_redshift_selection, prep_wise, flag_observation, create_bins, split_data
from utils.correct_extinction import correction
from settings.columns import (aper, specz, splus, error_splus, wise_flux, galex,
                              list_feat, create_colors, calculate_colors, create_ratio, calculate_ratio)
from settings.paths import match_path as data_path


def split(dataframe, test_frac:float, seed:int):
    
    print('# Dataframe Size:', len(dataframe))
    
    if test_frac == 0:
        train_sample = dataframe
        test_sample = pd.DataFrame()

    else:
        dataframe, _, _ = create_bins(data=dataframe, bin_size=0.5, return_data=True, var='Z')
        
        # Hard-coded
        train_sample, test_sample = split_data(dataframe)
        
        # train_sample, test_sample = train_test_split(
        #     dataframe, test_size=test_frac, stratify=dataframe['Zclass'], random_state=seed)
        
    print('# Train Size:    ', len(train_sample))
    print('# Test Size:     ', len(test_sample))

    return train_sample, test_sample


def plot_features(feature_list:list, scaled_train, output_dir:str, save_stuff:bool):
    
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
    if save_stuff:
        plt.savefig(os.path.join(output_dir, 'features.png'), bbox_inches='tight')
    plt.close()


def Process_Split(filename:str, mags:list, configs:dict, test_frac:float, seed:int, output_dir:str,
                  aper=aper, save_stuff=True):
    pd.options.mode.chained_assignment = None  # default='warn'
    
    flag_obs = False
    
    # Reading data
    file_path = os.path.join(data_path, filename)
    base_columns = ['ID', 'RA_1', 'DEC_1', 'SDSS_NAME']
    flag_columns = ['name', 'objID_x'] if flag_obs else []
    cols = base_columns + flag_columns + splus + error_splus + wise_flux + galex + [specz]
    data = pd.read_csv(file_path, usecols=cols)
    
    # Preparing data
    data = mag_redshift_selection(data, rmax=22, zmax=5)  # sample cuts
    data = prep_wise(data)  # wise flux to magnitude
    data = correction(data)  # extinction correction
    if flag_obs: data = flag_observation(data).drop(columns=flag_columns)  # WISE & GALEX observation flag
    
    # Replace S-PLUS missing features with the upper magnitude limit (the value in the error column)
    for mag, error in zip(splus, error_splus):
        data.loc[data[mag]==99, mag] = data.loc[data[mag]==99, error]
    
    # Processed dataframe
    if save_stuff: data.to_csv(os.path.join(output_dir, 'dataframe.csv'), index=False)
    
    # Defining features    
    feature_list = []
    broad_bool = 'broad' in mags
    narrow_bool = 'narrow' in mags
    wise_bool = 'wise' in mags
    galex_bool = 'galex' in mags
    
    if configs['mag']:
        feature_list += list_feat(aper, broad_bool, narrow_bool, wise_bool, galex_bool)

    if configs['col']:
        feature_list += create_colors(broad_bool, narrow_bool, wise_bool, galex_bool, aper)
        data = calculate_colors(data, broad_bool, narrow_bool, wise_bool, galex_bool, aper)
        
    if configs['rat']:
        feature_list += create_ratio(broad_bool, narrow_bool, wise_bool, galex_bool, aper)
        data = calculate_ratio(data, broad_bool, narrow_bool, wise_bool, galex_bool, aper)
    
    if flag_obs:
        feature_list += ['flag_WISE', 'flag_GALEX']
    
    print(f'# {len(feature_list)} Features:\n{feature_list}')
    
    # Dataframe with only base_columns and features
    if save_stuff: data[base_columns+feature_list].to_csv(os.path.join(output_dir, 'features.csv'), index=False)
    
    # Splitting sample
    train_sample, test_sample = split(data, test_frac, seed)
    train_sample['weights'] = 1
    
    # Fitting scalars    
    scaler_1 = QuantileTransformer(output_distribution='normal')
    scaler_2 = MinMaxScaler((0, 1))
    scaled_train = scaler_1.fit_transform(train_sample[feature_list])
    scaled_train = scaler_2.fit_transform(scaled_train)
    scaled_train = pd.DataFrame(scaled_train, columns=feature_list)
    plot_features(feature_list, scaled_train, output_dir, save_stuff)
    
    # Missing band mask
    train_mask = train_sample[feature_list].isna().reset_index(drop=True)
        
    if test_frac != 0:
        test_mask = test_sample[feature_list].isna().reset_index(drop=True)
        return train_sample, test_sample, feature_list, scaler_1, scaler_2, train_mask, test_mask
    
    if test_frac == 0:
        return train_sample, feature_list, scaler_1, scaler_2, train_mask
