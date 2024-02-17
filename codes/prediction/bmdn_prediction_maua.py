import os
import sys
from pathlib import Path

aux = os.path.join(Path.cwd(), 'codes')
if str(aux) not in sys.path:
    sys.path.append(aux)

import joblib
from tqdm import tqdm
from pandas import read_csv
from tensorflow.keras.models import load_model

from settings.paths import result_path, bmdn_path
from bmdn_functions import Process_Final, FinalPredict


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

## Basically the function PredictForFileNoTry ##

model_path = os.path.join(bmdn_path, 'final_model_dr4_BNWG')
predict_path = os.path.join(result_path, 'prediction', 'bmdn')

folders = {'model': model_path,
           'input': os.path.join(predict_path, 'input'),
           'output': os.path.join(predict_path, 'output')}

files_list = [s.replace('.csv', '') for s in os.listdir(folders['input'])]


# Loading the model
model = os.path.join(folders['model'], 'SavedModels')
PZ_Model = load_model(os.path.join(model, 'Fold0'), compile=False)

# Loading scalers
Scaler_1 = joblib.load(os.path.join(folders['model'], 'Scaler_1_Quantile.joblib'))
Scaler_2 = joblib.load(os.path.join(folders['model'], 'Scaler_2_MinMax.joblib'))

# Loading features
with open(os.path.join(folders['model'], 'Model_Summary.txt'), 'r') as summary_file:
    feature_list = eval(summary_file.readlines()[-1])

Progress_Bar = tqdm(files_list)
for file in Progress_Bar:
    Progress_Bar.set_description(f'# Predicting for file: {file}')
    
    HeaderToSave = ['ID', 'RA', 'DEC', 'z_bmdn_peak', 'n_peaks_bmdn']
    for i in range(7):
        HeaderToSave.append(f'pdf_weight_{i}')
    for i in range(7):
        HeaderToSave.append(f'pdf_mean_{i}')
    for i in range(7):
        HeaderToSave.append(f'pdf_std_{i}')
    Header = HeaderToSave

    chunk = read_csv(os.path.join(folders['input'], f'{file}.csv'))
    chunk = chunk.reset_index(drop=True)

    ### Processing data (calculate colours, ratios, mask) ###
    PredictSample, PredictSample_Features = Process_Final(chunk, feature_list, Scaler_1, Scaler_2)
    Result_DF = FinalPredict(PZ_Model, PredictSample, PredictSample_Features)

    # Saving results DataFrame
    Result_DF[HeaderToSave].to_csv(os.path.join(folders['output'], f'{file}.csv'), mode='a', index=False)