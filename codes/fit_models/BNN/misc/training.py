import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import tensorflow as tf
callbacks = tf.keras.callbacks
from tqdm.keras import TqdmCallback
import joblib

from misc.preprocessing import Process_Split
from misc.network import Dense_Variational, Epochs, Batch_Size, Activation, Opt


def config_model(filename:str, magnitudes:list, configs:dict, test_frac:float, seed:int, output_dir:str, scheme:str):
    
    Dataset = {}
    
    if scheme == 'AllTrain':
        Dataset['Train'], Train_Var, Scaler_1, Scaler_2, TrainMask \
            = Process_Split(filename, magnitudes, configs, test_frac, seed, output_dir)
        Save_IDs = pd.DataFrame()
        Dataset['Train']['ID'].to_csv(output_dir+'Train_IDs.csv')
        Testing_Data_Features = np.array([])
    
    else:
        Dataset['Train'], Dataset['Test'], Train_Var, Scaler_1, Scaler_2, TrainMask, TestMask \
            = Process_Split(filename, magnitudes, configs, test_frac, seed, output_dir)
        Save_IDs = pd.DataFrame()
        Save_IDs['ID'] = Dataset['Train']['ID'].reset_index(drop=True)
        Dataset['Test']['ID'].to_csv(output_dir+'Test_IDs.csv')
        Testing_Data_Features = Scaler_2.transform(Scaler_1.transform(Dataset['Test'][Train_Var].values))
        Testing_Data_Features[TestMask] = 0
        
    Training_Data_Features = Scaler_2.transform(Scaler_1.transform(Dataset['Train'][Train_Var].values))
    Training_Data_Features[TrainMask] = 0
    Training_Data_Target = np.squeeze(Dataset['Train']['Z'].values)

    joblib.dump(Scaler_1, output_dir+'/Scaler_1_Quantile.joblib')
    joblib.dump(Scaler_2, output_dir+'/Scaler_2_MinMax.joblib')
    
    return Dataset, Training_Data_Features, Training_Data_Target, Testing_Data_Features, Train_Var, Save_IDs


def plot_kfold(skfold, Training_Data_Features, Testing_Data_Features, Train_Var:list, Training_Data_Zclass, output_dir:str):
    
    fig, _ = plt.subplots(figsize=(20,20))
    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    plt_idx = 1
    for feat_idx in range(np.shape(Training_Data_Features)[1]):
        plt.subplot(10, 6, plt_idx)

        Feature_min = 0
        Feature_max = 1

        for train, validation in skfold.split(Training_Data_Features, Training_Data_Zclass):
            plt.hist(Training_Data_Features[train][:,feat_idx], lw=2, range=(Feature_min, Feature_max),
                    bins=20, histtype='step')
            plt.hist(Training_Data_Features[validation][:,feat_idx], lw=2, range=(Feature_min, Feature_max),
                    bins=20, histtype='step')
        plt.hist(Testing_Data_Features[:,feat_idx], lw=2, range=(Feature_min, Feature_max),
                bins=20, histtype='step')
        
        plt.yscale('log')
        plt.xlabel(Train_Var[feat_idx])
        plt.grid(lw=.5)
        plt_idx += 1

    fig.tight_layout()
    plt.savefig(output_dir+'KFold_Distributions.pdf', bbox_inches='tight')
    plt.close()


def all_train(Dataset:dict, Training_Data_Features, Training_Data_Target, Train_Var:list, seed:int, output_dir:str):
    
    TimeNow = datetime.now().strftime('%d-%m-%Y/%Hh%Mm')
    Model = {}
    Model_Fit = {}
    fold_number = 0

    train = Training_Data_Features
    print('##################################################################')
    print('# Starting single training')
    print(f'# Training start time: {TimeNow}')
    print(f'# Training Sample Objects = {len(train)} ({(100*len(train)/len(train)):.3g}%)')
    print()

    # Custom metrics
    CheckpointFolderEpoch = output_dir + f'Checkpoints/Fold{fold_number}' + '/Epoch{epoch:02d}'
    CheckpointEpoch = callbacks.ModelCheckpoint(CheckpointFolderEpoch, verbose=0, period=300)

    Model[fold_number] = Dense_Variational(
        Train_Sample_Size=len(Training_Data_Features), Number_of_Features=len(Train_Var))

    # Fitting the model
    Model_Fit[fold_number] = Model[fold_number].fit(Training_Data_Features, Training_Data_Target, epochs=Epochs,
                                                    batch_size=Batch_Size, verbose=0,
                                                    callbacks=[CheckpointEpoch, TqdmCallback(verbose=0)],
                                                    sample_weight=Dataset['Train']['weights'].values)

    # Save the model
    Save_Dir = output_dir + f'SavedModels/Fold{fold_number}'
    Model[fold_number].save(Save_Dir, overwrite=True)
    pd.DataFrame(Model_Fit[fold_number].history).to_csv(Save_Dir + '/Seed' + str(seed) + f'_Fold{fold_number}.csv')
    print()
    
    return Model, Model_Fit, len(train)


def kfold(Dataset:dict, Training_Data_Features, Training_Data_Target, Testing_Data_Features, Train_Var:list, Save_IDs,
          seed:int, output_dir:str):
    
    Training_Data_Zclass = Dataset['Train']['Zclass']
    skfold = sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_number = 0

    plot_kfold(skfold, Training_Data_Features, Testing_Data_Features, Train_Var, Training_Data_Zclass, output_dir)
    
    TimeNow = datetime.now().strftime('%d-%m-%Y/%Hh%Mm')
    Model = {}
    Model_Fit = {}

    # Train and save each fit separately
    for train, validation in skfold.split(Training_Data_Features, Training_Data_Zclass):
        t = len(train)+len(validation)+len(Testing_Data_Features)
        
        print('##################################################################')
        print(f'# Starting fold number {fold_number}')
        print(f'# Training start time: {TimeNow}')
        print(f'# Training Sample Objects   = {len(train)} ({(100*len(train)/t):.3g}%)')
        print(f'# Validation Sample Objects = {len(validation)} ({(100*len(validation)/t):.3g}%)')
        print(f'# Testing Sample Objects    = {len(Testing_Data_Features)} ({(100*len(Testing_Data_Features)/t):.3g}%)')
        print()

        Save_IDs[f'Fold_{fold_number}'] = 'V'
        Save_IDs[f'Fold_{fold_number}'][train] = 'T'
        Save_IDs.to_csv(output_dir+'Train_IDs.csv', index=False)

        # Custom metrics
        CheckpointFolderEpoch = output_dir + f'Checkpoints/Fold{fold_number}' + '/Epoch{epoch:02d}'
        CheckpointEpoch = callbacks.ModelCheckpoint(CheckpointFolderEpoch, verbose=0, period=500)

        # Compiling new model for each fold
        Model[fold_number] = Dense_Variational(Train_Sample_Size=len(train), Number_of_Features=len(Train_Var))

        # Fitting the model
        Model_Fit[fold_number] = Model[fold_number].fit(Training_Data_Features[train], Training_Data_Target[train],
                                                        sample_weight=Dataset['Train']['weights'].values[train],
                                                        validation_data=(Training_Data_Features[validation],
                                                                         Training_Data_Target[validation],
                                                                         Dataset['Train']['weights'].values[validation]), 
                                                        epochs=Epochs, batch_size=Batch_Size, verbose=0,
                                                        callbacks=[TqdmCallback(verbose=0)])

        # Save the model and loss
        Save_Dir = output_dir + f'SavedModels/Fold{fold_number}'
        Model[fold_number].save(Save_Dir, overwrite=True)
        pd.DataFrame(Model_Fit[fold_number].history).to_csv(Save_Dir + '/Seed' + str(seed) + f'_Fold{fold_number}.csv')
        fold_number += 1
        print()
        
        # Saving training IDs + Folds
        Save_IDs.to_csv(output_dir+'Train_IDs.csv', index=False)
        
    return Model, Model_Fit, len(train)
    
    
def save_model(scheme:str, filename:str, Dataset:dict, Training_Data_Features, Train_Var:list, Model:dict, len_train:int,
               output_dir:str):
    
    # Save model summary to file with some extra info
    #https://stackoverflow.com/questions/45199047/how-to-save-model-summary-to-file-in-keras
    with open(output_dir+'Model_Summary.txt', 'w') as f:
        Model[0].summary(print_fn=lambda x: f.write(x + '\n'))
        f.write(f'Output dir: {output_dir}\n')
        f.write(f'Epochs: {Epochs}, Batch_Size: {Batch_Size}\n')
        
        if scheme == 'AllTrain':
            f.write(f'Activation: {Activation.name}, kl_weight: 1/{len(Training_Data_Features)}\n')
        else:
            f.write(f'Activation: {Activation.name}, kl_weight: 1/{len_train}\n')
            
        f.write(f'Optimizer: {Opt.get_config()}\n')
        f.write(f'Loss: {Model[0].loss.__name__}\n')
        f.write(f'Input file: {filename}\n')
        if scheme == 'AllTrain':
            f.write(f'Total len: {len(Dataset["Train"])}\n')
        else:
            f.write(f'Total len: {len(Dataset["Train"])+len(Dataset["Test"])}\n')
        f.write(f'Number of features: {len(Train_Var)}\n')
        f.write(str(Train_Var))


def train_model(filename:str, magnitudes:list, configs:dict, test_frac:float, seed:int, output_dir:str, scheme:str):
    
    print(f'# Seed = {seed}')
    print(f'# Scheme: {scheme}')
    
    Dataset, Training_Data_Features, Training_Data_Target, Testing_Data_Features, Train_Var, Save_IDs \
        = config_model(filename, magnitudes, configs, test_frac, seed, output_dir, scheme)
    
    if scheme == 'AllTrain':
        Model, Model_Fit, len_train = all_train(Dataset, Training_Data_Features, Training_Data_Target, Train_Var, seed,
                                                output_dir)
    
    else:
        Model, Model_Fit, len_train = kfold(Dataset, Training_Data_Features, Training_Data_Target, Testing_Data_Features,
                                            Train_Var, Save_IDs, seed, output_dir)

    save_model(scheme, filename, Dataset, Training_Data_Features, Train_Var, Model, len_train, output_dir)
    
    return Model, Model_Fit
        
    
def plot_loss(scheme:str, seed:int, output_dir:str):
    
    model = os.path.join(output_dir, 'SavedModels')
    Folds = os.listdir(model)
    
    if scheme == 'AllTrain':
        loss_file = pd.read_csv(os.path.join(model, Folds[0], f'Seed{seed}_{Folds[0]}.csv'))
        ax = plt.figure(figsize=(4, 4))
        Min_Y = np.min(loss_file['loss'])
        plt.plot(loss_file['loss'], lw=1, alpha=1)
        plt.ylim(Min_Y-5, 50)
        plt.ylabel('Loss (NLL)')
        plt.xlabel('Epochs')
        
    else:
        _, ax = plt.subplots(1, len(Folds)+1, figsize=(3*len(Folds)+1, 4))
        loss = []
        val_loss = []
        for i in range(len(Folds)):
            loss_file = pd.read_csv(os.path.join(model, Folds[i], f'Seed{seed}_{Folds[i]}.csv'))
            loss.append(loss_file['loss'])
            val_loss.append(loss_file['val_loss'])
            Min_Y = np.min(loss_file['loss'])
            ax[i].plot(loss_file['loss'], lw=1, alpha=1, label='Train')
            ax[i].plot(loss_file['val_loss'], lw=1, alpha=1, label='Validation')
            ax[i].set_title(f'Fold {i}')
            if i == 0: ax[i].set_ylabel('Loss (NLL)')
            ax[i].set_xlabel('Epochs')
            ax[i].set_ylim(Min_Y-5, Min_Y+50)
            ax[i].legend()
        MeanTrain = np.median(np.array(loss), axis=0)
        MeanValid = np.median(np.array(val_loss), axis=0)
        Min_Y = np.min(MeanTrain)
        ax[-1].plot(MeanTrain, lw=1, alpha=1, label='Train')
        ax[-1].plot(MeanValid, lw=1, alpha=1, label='Validation')
        ax[-1].set_title('Median')
        ax[-1].set_xlabel('Epochs')
        ax[-1].set_ylim(Min_Y-5, Min_Y+50)
        ax[-1].legend()

    plt.tight_layout()
    plt.savefig(output_dir+'Loss_Comp.pdf', bbox_inches='tight')
