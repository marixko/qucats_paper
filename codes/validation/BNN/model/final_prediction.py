import os

import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import joblib

from model.preprocessing import Process_Final, Correct_Extinction
from model.sampling import FinalPredict

def PredictForFileNoTry(files_list:list, aper:str, magnitudes:list, configs:dict, folders:dict):
    
    # Loading the model
    model = os.path.join(folders['model'], 'SavedModels')
    PZ_Model = tf.keras.models.load_model(os.path.join(model, 'Fold0'), compile=False)

    # Loading scalers
    Scaler_1 = joblib.load(os.path.join(folders['model'], 'Scaler_1_Quantile.joblib'))
    Scaler_2 = joblib.load(os.path.join(folders['model'], 'Scaler_2_MinMax.joblib'))

    Progress_Bar = tqdm(files_list)
    for file in Progress_Bar:
        Progress_Bar.set_description(f'# Predicting for file: {file}')
        
        HeaderToSave = ['ID', 'RA', 'DEC', 'zphot', 'zphot_2.5p', 'zphot_16p', 'zphot_84p', 'zphot_97.5p',
                        'pdf_peaks', 'zphot_second_peak', 'pdf_width', 'odds']
        for i in range(7):
            HeaderToSave.append(f'pdf_weight_{i}')
        for i in range(7):
            HeaderToSave.append(f'pdf_mean_{i}')
        for i in range(7):
            HeaderToSave.append(f'pdf_std_{i}')
        Header = HeaderToSave

        Pred_in_chunks = False
        chunk = pd.read_csv(os.path.join(folders['input'], f'{file}.csv'))
        ChunkSize = 50000
        if len(chunk) >= ChunkSize:
            Pred_in_chunks = True
            print(f'# File {file} is too large, predicting in chunks...')

        ### Processing data (calculate colours, ratios, mask) ###
        if Pred_in_chunks:
            for chunk in pd.read_csv(os.path.join(folders['input'], f'{file}.csv'), chunksize=ChunkSize):

                try: 
                    chunk.rename(columns={'B_in': 'B'}, inplace=True)
                except:
                    pass

                chunk = chunk.reset_index(drop=True)
                chunk = Correct_Extinction(chunk, aper)

                PredictSample, PredictFeatures, PredictMask = Process_Final(chunk, aper, magnitudes, configs)
                PredictSample_Features = Scaler_2.transform(Scaler_1.transform(PredictSample[PredictFeatures]))
                PredictSample_Features[PredictMask] = 0

                Result_DF = FinalPredict(PZ_Model, chunk, PredictSample_Features)

                # Saving results DataFrame
                Result_DF[HeaderToSave].to_csv(os.path.join(folders['output'], f'{file}.csv'),
                                               mode='a', index=False, header=Header)
                Header = None
        
        else:
            try: 
                chunk.rename(columns={'B_in': 'B'}, inplace=True)
            except:
                pass

            chunk = chunk.reset_index(drop=True)
            chunk = Correct_Extinction(chunk, aper)

            PredictSample, PredictFeatures, PredictMask = Process_Final(chunk, aper, magnitudes, configs)
            PredictSample_Features = Scaler_2.transform(Scaler_1.transform(PredictSample[PredictFeatures]))
            PredictSample_Features[PredictMask] = 0

            Result_DF = FinalPredict(PZ_Model, chunk, PredictSample_Features)

            # Saving results DataFrame
            Result_DF[HeaderToSave].to_csv(os.path.join(folders['output'], f'{file}.csv'), mode='a', index=False)
