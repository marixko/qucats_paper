import os
from itertools import combinations
from warnings import simplefilter

import numpy as np
import pandas as pd

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler
import matplotlib.pyplot as plt
import extinction

from utils_bnn import sfdmap
from settings.paths import (match_path as data_path,
                            bnn_path as results_path,
                            dust_path)


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
        Magnitudes_WISE = ['W1_MAG','W2_MAG']
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


def create_zclass(Dataframe, Bin_Size:float):
    zmin = np.floor(np.min(Dataframe['Z']))

    zmax = np.ceil(np.max(Dataframe['Z']))
    if zmax - np.max(Dataframe['Z']) > Bin_Size:
        zmax = zmax - Bin_Size

    bins = np.arange(zmin, zmax+Bin_Size, Bin_Size)
    Z_class = pd.cut(Dataframe['Z'], bins=bins, labels=range(len(bins)-1))

    Dataframe['Zclass'] = Z_class
    return bins


def split(Dataframe, Test_Frac:float, Stratify:bool, Data_Seed:float):
    
    print('# Dataframe Size:', len(Dataframe))
    
    if Test_Frac == 0:
        TrainingSample = Dataframe
        TestingSample = pd.DataFrame()

    else:
        
        if Stratify:
            TrainingSample, TestingSample = train_test_split(
                Dataframe, test_size=Test_Frac, stratify=Dataframe['Zclass'], random_state=Data_Seed)
        else:
            TrainingSample, TestingSample = train_test_split(
                Dataframe, test_size=Test_Frac, random_state=Data_Seed)   
        
    print('# Train Size:    ', len(TrainingSample))
    print('# Test Size:     ', len(TestingSample))

    return TrainingSample, TestingSample


def scale_features(TrainingSample, Training_Features:list):
    
    Scaler_X_1 = QuantileTransformer(output_distribution='normal')
    Scaled_Train_X = Scaler_X_1.fit_transform(TrainingSample[Training_Features])
    Scaler_X_2 = MinMaxScaler((0,1))
    Scaled_Train_X = Scaler_X_2.fit_transform(Scaled_Train_X)
    Scaled_Train_X = pd.DataFrame(Scaled_Train_X, columns=Training_Features)
        
    return Scaler_X_1, Scaler_X_2, Scaled_Train_X


def plot_features(Training_Features:list, Scaled_Train_X):
    
    fig, _ = plt.subplots(figsize=(20, 20))
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    Features_to_plot = Training_Features

    plt_idx = 1
    for feature in Features_to_plot:
        plt.subplot(10, 6, plt_idx)
        plt.hist(Scaled_Train_X[feature], lw=2, range=(0, 1), bins=20, histtype='step')
        plt.yscale('log')
        plt.xlabel(feature)
        plt.grid(lw=.5)
        plt_idx = plt_idx+1

    fig.tight_layout()
    plt.show()


def plot_split(Dataframe, TrainingSample, TestingSample, bins):
    
    plt.figure(figsize=(10, 8))
    plt.hist(Dataframe['Z'], bins=bins, label='all')
    plt.hist(TrainingSample['Z'], bins=bins, label='train sample')
    plt.hist(TestingSample['Z'], bins=bins, label='test sample')
    plt.legend()
    plt.tight_layout()
    plt.show()


def Correct_Extinction(Dataframe, magnitudes:list, Aper:str, Extinction_Maps=None):
    '''
    Correct the magnitudes for extinction using the CCM89 Law

    Keyword arguments:
    Dataframe         -- Dataframe containing the data to be corrected
    Extinction_Maps   -- SFD Maps
    '''
    print('# Correcting magnitudes for extinction...')
    Corrected_Dataframe = Dataframe.copy().reset_index(drop=True)
    
    Magnitudes_SPLUS, Magnitudes_WISE, Magnitudes_GALEX, _, _ = select_magnitudes(Aper, magnitudes, prt=False)
    Magnitudes = Magnitudes_GALEX + Magnitudes_SPLUS + Magnitudes_WISE
    
    if not Extinction_Maps:
        Extinction_Maps = sfdmap.SFDMap(dust_path)

    # Obtaining E(B-V) and Av in a given RA, DEC position
    AV = Extinction_Maps.ebv(Dataframe['RA'], Dataframe['DEC'])*3.1

    # Calculating the extinction on the S-PLUS bands using the Cardelli, Clayton & Mathis law.
    lambdas_dict = {'FUVmag': 1549.02, 'NUVmag': 2304.74,
                    'u_PStotal': 3536, 'J0378_PStotal': 3770, 'J0395_PStotal': 3940, 'J0410_PStotal': 4094,
                    'J0430_PStotal': 4292, 'g_PStotal': 4751, 'J0515_PStotal': 5133, 'r_PStotal': 6258,
                    'J0660_PStotal': 6614, 'i_PStotal': 7690, 'J0861_PStotal': 8611, 'z_PStotal': 8831,
                    'W1_MAG': 33526, 'W2_MAG': 46028}
    lambdas = []
    for name, lamb in lambdas_dict.items():
        if name in Magnitudes:
            lambdas.append(lamb)
    lambdas = np.array(lambdas, dtype=float)

    Extinctions = []
    for i in range(len(AV)):
        Extinctions.append(extinction.ccm89(lambdas, AV[i], 3.1))

    Extinction_DF = pd.DataFrame(Extinctions, columns=Magnitudes)
    Corrected_Dataframe[Magnitudes] = Corrected_Dataframe[Magnitudes].sub(Extinction_DF)

    return Corrected_Dataframe


def Process_Split(Filename:str, Aper:str, magnitudes:list, Test_Frac:float, Bin_Size:float, Plot:int, Configs:dict,
                  Conds:list, Seed:float, correct_ext:bool, Output_Dir:str, save_df:bool):
    pd.options.mode.chained_assignment = None  # default='warn'
    
    if 'broad' in magnitudes and 'narrow' in magnitudes:
        splus = ['u', 'J0378', 'J0395', 'J0410', 'J0430', 'g', 'J0515', 'r', 'J0660', 'i', 'J0861', 'z']
    elif 'broad' in magnitudes:
        splus = ['u', 'g', 'r', 'i', 'z']
    elif 'narrow' in magnitudes:
        splus = ['J0378', 'J0395', 'J0410', 'J0430', 'J0515', 'J0660', 'J0861']
    Magnitudes_WISE = ['W1_MAG','W2_MAG'] if 'wise' in magnitudes else []
    Magnitudes_GALEX = ['FUVmag', 'NUVmag'] if 'galex' in magnitudes else []

    Base_Columns = ['ID', 'RA', 'DEC', 'PhotoFlagDet', 'nDet_'+Aper]
    Magnitudes_SPLUS = [item+'_'+Aper for item in splus]
    Errors_SPLUS = ['e_'+item for item in Magnitudes_SPLUS]
    Magnitudes = Magnitudes_GALEX + Magnitudes_SPLUS + Magnitudes_WISE  # It's important to keep this order
    
    File_path = os.path.join(data_path, Filename)
    cols = Base_Columns+Magnitudes+Errors_SPLUS+['Z']
    if 'broad' not in magnitudes:
        cols.append('r_'+Aper)
    Dataframe = pd.read_csv(File_path, usecols=cols)
    
    for C in Conds:
        Dataframe = Dataframe.query(C)

    # Non detected/observed objects
    for feature in Magnitudes:
        Dataframe[feature][~Dataframe[feature].between(10, 50)] = np.nan
    
    if correct_ext:
        Dataframe = Correct_Extinction(Dataframe, magnitudes, Aper)

    # Replace S-PLUS missing features with the upper magnitude limit (the value in the error column)
    for feature, error in zip(Magnitudes_SPLUS, Errors_SPLUS):
        Dataframe[feature].fillna(Dataframe[error], inplace=True)
    
    Training_Features = []
    
    if Configs['mag']:
        Training_Features += Magnitudes

    if Configs['col']:

        if 'broad' not in magnitudes:
            # Inserting r band in the correct position only to calculate colors
            j0660_idx = Magnitudes.index('J0660_'+Aper)
            Colors_Magnitudes = Magnitudes[:j0660_idx] + ['r_'+Aper] + Magnitudes[j0660_idx:]
        else:
            Colors_Magnitudes = Magnitudes
        
        Colors = calc_colors(Dataframe, Aper, Colors_Magnitudes)
        Training_Features += Colors
    
    if Configs['rat']:
        Ratios = calc_ratios(Dataframe, Magnitudes)
        Training_Features += Ratios
    
    print(f'# {len(Training_Features)} Features:\n{Training_Features}')
    
    if Bin_Size:
        Bins = create_zclass(Dataframe, Bin_Size)
    else:
        Bins = np.arange(0, np.max(Dataframe['Z'])+0.25, 0.25)
    TrainingSample, TestingSample = split(Dataframe, Test_Frac, Bin_Size, Seed)
    TrainingSample['weights'] = 1
    
    if save_df: Dataframe.to_csv(os.path.join(Output_Dir, 'Dataframe.csv'), index=False)
    
    Scaler_X_1, Scaler_X_2, Scaled_Train_X = scale_features(TrainingSample, Training_Features)
    
    TrainingMask = TrainingSample[Training_Features].isna().reset_index(drop=True)
    if Test_Frac != 0:
        TestingMask  = TestingSample[Training_Features].isna().reset_index(drop=True)

    if Plot == 1:
        plot_features(Training_Features, Scaled_Train_X)
    elif Plot == 2:
        plot_split(Dataframe, TrainingSample, TestingSample, Bins)
    elif Plot == 3:
        plot_features(Training_Features, Scaled_Train_X)
        plot_split(Dataframe, TrainingSample, TestingSample, Bins)
        
    if Test_Frac != 0:
        return TrainingSample, TestingSample, Training_Features, \
            Scaler_X_1, Scaler_X_2, TrainingMask, TestingMask
    
    if Test_Frac == 0:
        return TrainingSample, Training_Features, Scaler_X_1, Scaler_X_2, TrainingMask


def Process_Final(Dataframe, Aper:str, magnitudes:list, Configs:dict):
    
    if 'broad' in magnitudes and 'narrow' in magnitudes:
        splus = ['u', 'J0378', 'J0395', 'J0410', 'J0430', 'g', 'J0515', 'r', 'J0660', 'i', 'J0861', 'z']
    elif 'broad' in magnitudes:
        splus = ['u', 'g', 'r', 'i', 'z']
    elif 'narrow' in magnitudes:
        splus = ['J0378', 'J0395', 'J0410', 'J0430', 'J0515', 'J0660', 'J0861']
    Magnitudes_WISE = ['W1_MAG','W2_MAG'] if 'wise' in magnitudes else []
    Magnitudes_GALEX = ['FUVmag', 'NUVmag'] if 'galex' in magnitudes else []

    Magnitudes_SPLUS = [item+'_'+Aper for item in splus]
    Errors_SPLUS = ['e_'+item for item in Magnitudes_SPLUS]
    Magnitudes = Magnitudes_GALEX + Magnitudes_SPLUS + Magnitudes_WISE  # It's important to keep this order
    
    # Non detected/observed objects
    for feature in Magnitudes:
        Dataframe[feature][~Dataframe[feature].between(10, 50)] = np.nan

    # Replace S-PLUS missing features with the upper magnitude limit (the value in the error column)
    for feature, error in zip(Magnitudes_SPLUS, Errors_SPLUS):
        Dataframe[feature].fillna(Dataframe[error], inplace=True)
    
    Training_Features = []
    
    if Configs['mag']:
        Training_Features += Magnitudes

    if Configs['col']:
        if 'broad' not in magnitudes:
            # Inserting r band in the correct position only to calculate colors
            j0660_idx = Magnitudes.index('J0660_'+Aper)
            Colors_Magnitudes = Magnitudes[:j0660_idx] + ['r_'+Aper] + Magnitudes[j0660_idx:]
        else:
            Colors_Magnitudes = Magnitudes
        Colors = calc_colors(Dataframe, Aper, Colors_Magnitudes)
        Training_Features += Colors
    
    if Configs['rat']:
        Ratios = calc_ratios(Dataframe, Magnitudes)
        Training_Features += Ratios
    
    # Mask
    Mask = Dataframe[Training_Features].isna().reset_index(drop=True)

    return Dataframe, Training_Features, Mask
