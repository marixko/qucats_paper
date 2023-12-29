import os
from pathlib import Path
import sys
from random import seed
from time import sleep
from threading import Thread

import numpy as np
import tensorflow as tf

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=Warning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

aux = os.path.join(Path.cwd(), "codes")
if str(aux) not in sys.path:
    sys.path.append(aux)

from settings.paths import bnn_path as results_path
from model.loading import load
from model.training import train_model, plot_loss
from model.sampling import sampling
from model.final_prediction import PredictForFileNoTry
from visualization.plots import Benchmarks


train_data = 'iDR3n4_DR16Q_GALEX2_unWISE2-ren.csv'
Seed = 47
scheme = 'KFold'
train = True
predict = True
correct_ext = True

magnitudes = ['broad', 'narrow', 'wise', 'galex']
configs = {'mag': False, 'col': True, 'rat': False}
aperture = 'PStotal'
Conds = [f'r_{aperture} <= 22 | r_{aperture} == 99', 'Z <= 7']
final_predict_path = ''  # ARRUMAR

feat = '_'+''.join([m[0].upper() for m in magnitudes])
obs = '_masktest'
if scheme == 'KFold':
    test_frac = 0.25
    model_path = os.path.join(results_path, f'crossval_model{obs}', '')
else:
    test_frac = 0
    model_path = os.path.join(results_path, f'final_model{obs}', '')

if not os.path.isdir(model_path): os.makedirs(model_path)
print(f'# Model output directory: {model_path}')

zlim = 5
Bench_Dict = {'Do_Histograms': True,
              'Do_PDF_Bins_RA': True,
              'Do_LSS_2D': True,
              'Do_Average_PDFs': True,
              'Do_PDF_Sample': True,
              'Do_Odds_PIT_CRPS_HPDCI_Delta': True,
              'Do_Odds_Hexbin': True,
              'Do_Uncertainties': True,
              'Do_Limited_Metrics': True,
              'Do_Results_Bins': True,
              'Do_Results_Bins_Cumul': True,
              'Do_Zphot_Zspec': True}

seed(0)
np.random.seed(0)
tf.compat.v1.random.set_random_seed(0)
tf.random.set_seed(0)
os.environ['PYTHONHASHSEED'] = str(0)

if train:
    print(f'Training ({scheme})')
    Model, Model_Fit = train_model(Filename=train_data, Aper=aperture, magnitudes=magnitudes, Test_Frac=test_frac,
                                   Bin_Size=0.5, Plot_Preprocessing=False, Configs=configs, Conds=Conds, Seed=Seed,
                                   correct_ext=correct_ext, Scheme=scheme, Plot_Train=False, Output_Dir=model_path)
    plot_loss(scheme, model_path)

if predict:
    print(f'Loading + sampling')

    if scheme == 'KFold':

        Model, Dataset, Testing_Data_Features = load(Filename=train_data, Aper=aperture, magnitudes=magnitudes,
                                                     Test_Frac=test_frac, Bin_Size=0.5, Plot_Preprocessing=False,
                                                     Configs=configs, Conds=Conds, Seed=Seed, correct_ext=correct_ext,
                                                     Dir=model_path, Load_Samples=True)

        Result_DF, Final_PDFs, x = sampling(Aper=aperture, Model=Model, Testing_Dataframe=Dataset,
                                            Testing_Data_Features=Testing_Data_Features, Num_Samples=200,
                                            Output_PDFs=True)

        Result_DF.to_csv(model_path+'Results_DF.csv', index=False)
        np.savetxt(model_path+'x.txt', x)
        
        bench_path = os.path.join(model_path, 'benchmarks', '')
        if not os.path.isdir(bench_path): os.makedirs(bench_path)
        Benchmarks(Result_DF, Final_PDFs, x, aperture, zlim, {'z': (0, 5, 10), 'r_'+aperture: (16, 22, 12)}, Seed,
                   bench_path, Bench_Dict, DarkMode=False, Save=True, Show=False, Close=True)
        print()

    else:
        print(f'PREDICTING {scheme} 3')

        # To run the code with GPU, set Run_on_GPU to True
        # https://stackoverflow.com/questions/37660312/how-to-run-tensorflow-on-cpu
        Run_on_GPU = False
        if Run_on_GPU == False:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            print('# INFO: Not using GPU')
        if Run_on_GPU == True:
            print('# INFO: Using GPU')
        
        folders = {'model': model_path,
                   'input': os.path.join(final_predict_path, 'input'),
                   'output': os.path.join(final_predict_path, 'output')}

        print('# Prediction with model from', folders['model'])
        All_Files = [s.replace('.csv', '') for s in os.listdir(folders['input'])]
        Predicted_Files = [s.replace('.csv', '') for s in os.listdir(folders['output'])]
        Files = np.setdiff1d(All_Files, Predicted_Files) # Fields that will be predicted

        Files_to_predict_parallel = 2
        Threads = np.arange(0, len(Files)+1, 1)
        print('# Number of files: ', len(Files))

        Processes = {}
        for i in range(len(Threads)-1):
            Processes[i] = Thread(target=PredictForFileNoTry,
                                  args=(Files[Threads[i]:Threads[i+1]], aperture, magnitudes, configs, folders))

        for lista in np.array_split(np.arange(0, len(Threads)-1), np.ceil(len(Threads)/Files_to_predict_parallel)):
            print('# Starting threads:', lista)
            for i in lista:
                Processes[i].start()
                sleep(1.5)

            for i in lista:
                Processes[i].join()
            print('# Finished threads:', lista)
            print()
