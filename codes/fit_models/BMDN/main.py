import os
import sys
import warnings
from random import seed

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf

main_dir_name = 'qucats_paper'
mycwd = os.path.abspath(os.getcwd())
main_dir = os.path.join(mycwd.split(main_dir_name)[0], main_dir_name)
sys.path.append(os.path.join(main_dir, 'codes'))

from settings.columns import aper
from settings.paths import bmdn_path
from misc.loading import load
from misc.training import train_model, plot_loss
from misc.sampling import sampling
from misc.plots import Benchmarks


tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=Warning)

data = 'STRIPE82_DR4_DR16Q1a_unWISE2a_GALEXDR672a.csv'
model_seed = 47
scheme = 'KFold'

mags = ['broad', 'wise', 'galex']
configs = {'mag': False, 'col': True, 'rat': False}

obs = '_dr4_BWG_700e'
if scheme == 'KFold':
    test_frac = 0.25
    model_path = os.path.join(bmdn_path, f'crossval_model{obs}', '')
elif scheme == 'AllTrain':
    test_frac = 0
    model_path = os.path.join(bmdn_path, f'final_model{obs}', '')
else:
    print('Choose scheme.')

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

print(f'Training ({scheme})')

Model, Model_Fit = train_model(filename=data, magnitudes=mags, configs=configs, test_frac=test_frac,
                                seed=model_seed, output_dir=model_path, scheme=scheme)

try:
    plot_loss(scheme, model_seed, model_path)
except Exception as erro:
    print(erro)

if scheme == 'KFold':
    
    print(f'Loading + sampling')

    Model, Testing_Data, Testing_Data_Features = load(filename=data, magnitudes=mags, configs=configs,
                                                      test_frac=test_frac, seed=model_seed, input_dir=model_path,
                                                      load_samples=True)
    
    Result_DF, Final_PDFs, x = sampling(Model=Model, Testing_Data=Testing_Data,
                                        Testing_Data_Features=Testing_Data_Features)

    Result_DF.to_csv(model_path+'Results_DF.csv', index=False)
    np.savetxt(model_path+'x.txt', x)
    
    bench_path = os.path.join(model_path, 'benchmarks', '')
    if not os.path.isdir(bench_path): os.makedirs(bench_path)
    Benchmarks(Result_DF, Final_PDFs, x, aper, zlim, {'z': (0, 5, 10), 'r_'+aper: (16, 22, 12)}, model_seed,
                bench_path, Bench_Dict, DarkMode=False, Save=True, Show=False, Close=True)
    print()
