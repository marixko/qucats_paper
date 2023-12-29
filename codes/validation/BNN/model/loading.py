import os
import sys

import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import joblib

from model.preprocessing import Process_Split


def load(Filename:str, Aper:str, magnitudes:list, Test_Frac:float, Bin_Size:float, Plot_Preprocessing:int, Configs:dict,
         Conds:list, Seed:float, correct_ext:bool, Dir:str, Load_Samples=True):
    
    Seed = int(os.listdir(os.path.join(Dir, 'SavedModels', 'Fold0'))[2].split('_')[0][4:])
    print(f'# Loading models from {Dir} and using the seed {Seed}')

    Scaler_1 = joblib.load(Dir+'/Scaler_1_Quantile.joblib')
    Scaler_2 = joblib.load(Dir+'/Scaler_2_MinMax.joblib')

    # Data is loaded first due to an Train/Test ID check. If the sample is different, the script is interrupted
    if Load_Samples:
        Dataset = {}
        print('# Make sure that this function is the same as the one used for training!')
        Dataset['Train'], Dataset['Test'], Train_Var, _, _, TrainMask, TestMask \
            = Process_Split(Filename, Aper, magnitudes, Test_Frac, Bin_Size, Plot_Preprocessing, Configs, Conds, Seed,
                            correct_ext, Dir, False)

        # Checking if the loaded data is the same as the one saved during training
        Train_IDs = pd.read_csv(Dir+'Train_IDs.csv')
        Test_IDs = pd.read_csv(Dir+'Test_IDs.csv')
        # The condition return false if any value is different between the lists
        if not (Dataset['Train']['ID'].reset_index(drop=True) == Train_IDs['ID'].reset_index(drop=True)).all():
            sys.exit("# Train samples are different, exiting...")
        if not (Dataset['Test']['ID'].reset_index(drop=True) == Test_IDs['ID'].reset_index(drop=True)).all():
            sys.exit("# Test samples are different, exiting...")

        Training_Data_Features = Scaler_2.transform(Scaler_1.transform(Dataset['Train'][Train_Var].values))
        Training_Data_Features[TrainMask] = 0
        Testing_Data_Features = Scaler_2.transform(Scaler_1.transform(Dataset['Test'][Train_Var].values))
        Testing_Data_Features[TestMask] = 0

    Folds = [int(s[4:]) for s in os.listdir(Dir+'SavedModels/')]
    Model = {}

    Progress_Bar = tqdm(Folds)
    for fold in Progress_Bar:
        Progress_Bar.set_description(f'# Loading fold {fold}')
        Model[fold] = tf.keras.models.load_model(Dir + f'SavedModels/Fold{fold}', compile=False)

    return Model, Dataset, Testing_Data_Features
