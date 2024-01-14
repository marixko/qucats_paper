import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
from pathlib import Path
from time import sleep
from threading import Thread

import numpy as np

aux = os.path.join(Path.cwd(), 'codes')
if str(aux) not in sys.path:
    sys.path.append(aux)

from settings.paths import result_path, bmdn_path
from bmdn_functions import PredictForFileNoTry

model_path = os.path.join(bmdn_path, 'final_model_dr4_BNWG')
predict_path = os.path.join(result_path, 'prediction', 'bmdn')

folders = {'model': model_path,
           'input': os.path.join(predict_path, 'input'),
           'output': os.path.join(predict_path, 'output')}

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
                            args=(Files[Threads[i]:Threads[i+1]], folders))

for lista in np.array_split(np.arange(0, len(Threads)-1), np.ceil(len(Threads)/Files_to_predict_parallel)):
    print('# Starting threads:', lista)
    for i in lista:
        Processes[i].start()
        sleep(1.5)

    for i in lista:
        Processes[i].join()
    print('# Finished threads:', lista)
    print()
