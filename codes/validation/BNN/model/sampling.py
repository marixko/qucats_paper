import gc

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import integrate
import tensorflow_probability as tfp; tfd = tfp.distributions

from utils.metrics import Odds, PIT, CRPS, HPDCI, Q


def Calc_PDF(x, Weights, Means, STDs):
    '''To calculate PDFs.'''
    if isinstance(Weights, str):
        Weights = np.fromstring(Weights, sep=',')
        Means = np.fromstring(Means, sep=',')
        STDs = np.fromstring(STDs, sep=',')
    PDF = np.sum( Weights * (1/(STDs*np.sqrt(2*np.pi))) * np.exp((-1/2)*((x[:, None]-Means)**2)/(STDs)**2), axis=1 )
    return PDF/np.trapz(PDF, x)


def sampling(Model:dict, Testing_Dataframe:dict, Testing_Data_Features, Num_Samples=200, Output_PDFs=True, aper='PStotal'):

    Folds = Model.keys()
    print(f"# Predicting for {len(Folds)} folds")
    x = np.linspace(0, 7, 7000, endpoint=True)
    
    Testing_Data_Target = Testing_Dataframe['Test']['Z'].values
    Result_DF = pd.DataFrame()
    Result_DF['ID_SPLUS'] = Testing_Dataframe['Test']['ID'].values
    Result_DF['RA'] = Testing_Dataframe['Test']['RA_1'].values
    Result_DF['DEC'] = Testing_Dataframe['Test']['DEC_1'].values
    Result_DF['r_'+aper] = Testing_Dataframe['Test']['r_'+aper].values
    Result_DF['z'] = Testing_Data_Target
    
    Fold_Weights = {}
    Fold_Means = {}
    Fold_STDs = {}
    for fold in Folds:
        
        Samples_Weights = []
        Samples_Means = []
        Samples_STDs = []
        for i in tqdm(range(Num_Samples)):
            
            Pred = Model[fold](Testing_Data_Features)
            Weight = Pred.submodules[1].probs_parameter().numpy()
            Mean = Pred.submodules[0].mean().numpy().reshape(len(Testing_Data_Features), np.shape(Weight)[1])
            Std = Pred.submodules[0].stddev().numpy().reshape(len(Testing_Data_Features), np.shape(Weight)[1])
            
            Sorted_Weight_Index = np.flip(np.argsort(Weight, axis=1), axis=1)
            Samples_Weights.append(Weight[np.arange(len(Weight))[:,None], Sorted_Weight_Index])
            Samples_Means.append(Mean[np.arange(len(Mean))[:,None], Sorted_Weight_Index])
            Samples_STDs.append(Std[np.arange(len(Std))[:,None], Sorted_Weight_Index])

        Fold_Weights[fold] = np.median(Samples_Weights, axis=0)
        Fold_Means[fold] = np.median(Samples_Means, axis=0)
        Fold_STDs[fold] = np.median(Samples_STDs, axis=0)

        Fold_PhotoZs = []
        Fold_Odds = []
        Fold_PITs = []
        Fold_CRPS = []
        Fold_HPDCI = []

        for i in range(len(Testing_Data_Features)):
            
            Obj_PDF = Calc_PDF(x, Fold_Weights[fold][i], Fold_Means[fold][i], Fold_STDs[fold][i])
            Approx_Zphot = x[np.argmax(Obj_PDF)]
            x_detailed = np.linspace(Approx_Zphot-0.025, Approx_Zphot+0.025, 400)
            Fold_PhotoZs.append(x_detailed[np.argmax(
                Calc_PDF(x_detailed, Fold_Weights[fold][i], Fold_Means[fold][i], Fold_STDs[fold][i])
                )])
            Fold_HPDCI.append(HPDCI(x, Obj_PDF, Testing_Data_Target[i]))
            
            Obj_CDF = integrate.cumtrapz(Obj_PDF, x, initial=0)
            Fold_Odds.append(Odds(x, Obj_CDF, Fold_PhotoZs[i]))
            Fold_PITs.append(PIT(x, Obj_CDF, Testing_Data_Target[i]))
            Fold_CRPS.append(CRPS(x, Obj_CDF, Testing_Data_Target[i]))

        Result_DF[f'zphot_{fold}'] = Fold_PhotoZs
        Result_DF[f'Odds_{fold}'] = Fold_Odds
        Result_DF[f'PIT_{fold}'] = Fold_PITs
        Result_DF[f'CRPS_{fold}'] = Fold_CRPS
        Result_DF[f'HPDCI_{fold}'] = Fold_HPDCI

    Final_Weights = np.median([Fold_Weights[fold] for fold in Folds], axis=0)
    Final_Means = np.median([Fold_Means[fold] for fold in Folds], axis=0)
    Final_STDs = np.median([Fold_STDs[fold] for fold in Folds], axis=0)

    PDF = tfd.MixtureSameFamily(
          mixture_distribution=tfd.Categorical(probs=Final_Weights),
          components_distribution=tfd.Normal(loc=Final_Means, scale=Final_STDs))
    PDF_STDs = PDF.stddev().numpy()

    Fine_ZPhot = []
    Final_PDFs = []
    Oddss = []
    PITs = []
    CRPSs = []
    HPDCIs = []
    Q68_Lower = []
    Q68_Higher = []
    Q95_Lower = []
    Q95_Higher = []

    for i in range(len(Testing_Data_Features)):
        
        Obj_PDF = Calc_PDF(x, Final_Weights[i], Final_Means[i], Final_STDs[i])
        Final_PDFs.append(Obj_PDF)
        Approx_Zphot = x[np.argmax(Obj_PDF)]
        x_detailed = np.linspace(Approx_Zphot-0.025, Approx_Zphot+0.025, 400)
        Fine_ZPhot.append(x_detailed[np.argmax(
            Calc_PDF(x_detailed, Final_Weights[i], Final_Means[i], Final_STDs[i])
            )])
        HPDCIs.append(HPDCI(x, Obj_PDF, Testing_Data_Target[i]))

        Obj_CDF = integrate.cumtrapz(Obj_PDF, x, initial=0)
        Oddss.append(Odds(x, Obj_CDF, Fine_ZPhot[i]))
        PITs.append(PIT(x, Obj_CDF, Testing_Data_Target[i]))
        CRPSs.append(CRPS(x, Obj_CDF, Testing_Data_Target[i]))

        Q68_Lower.append(Q(68, x, Obj_CDF))
        Q68_Higher.append(Q(68, x, Obj_CDF, lower=False))
        Q95_Lower.append(Q(95, x, Obj_CDF))
        Q95_Higher.append(Q(95, x, Obj_CDF, lower=False))
        
    Sec_ZPhot = []
    Num_Peaks = []

    for i in range(len(Final_PDFs)):
        Diff = np.diff(Final_PDFs[i])/np.diff(x)
        Peak_Idx = np.where(np.diff(np.sign(Diff)) == -2)[0]

        if np.sum(Final_PDFs[i][Peak_Idx] >= np.max(Final_PDFs[i])/3) >= 2:
            Num_Peaks.append(np.sum(Final_PDFs[i][Peak_Idx] >= np.max(Final_PDFs[i])/3))
            Sorted_Idxs = Peak_Idx[np.argsort(Final_PDFs[i][Peak_Idx])[::-1]]

            if np.count_nonzero(Final_PDFs[i][Peak_Idx] >= np.max(Final_PDFs[i])/3) >= 2:
                Sorted_Idxs = Sorted_Idxs[:2] + 1
                Sec_ZPhot.append(x[Sorted_Idxs[-1]])

        else:
            Num_Peaks.append(1)
            Sec_ZPhot.append(np.nan)

    Result_DF['zphot'] = Fine_ZPhot
    Result_DF['Odds'] = Oddss
    Result_DF['PIT'] = PITs
    Result_DF['CRPS'] = CRPSs
    Result_DF['HPDCI'] = HPDCIs
    Result_DF['zphot_err'] = PDF_STDs
    Result_DF['zphot_2.5q'] = Q95_Lower
    Result_DF['zphot_16q'] = Q68_Lower
    Result_DF['zphot_84q'] = Q68_Higher
    Result_DF['zphot_97.5q'] = Q95_Higher
    Result_DF['zphot_sec'] = Sec_ZPhot
    Result_DF['num_peaks'] = Num_Peaks

    if Output_PDFs:
        
        for i in range(len(Final_Weights[0])):
            Result_DF[f'PDF_Weight_{i}'] = Final_Weights[:,i]
        for i in range(len(Final_Means[0])):
            Result_DF[f'PDF_Mean_{i}'] = Final_Means[:,i]
        for i in range(len(Final_STDs[0])):
            Result_DF[f'PDF_STD_{i}'] = Final_STDs[:,i]

    return Result_DF, Final_PDFs, x


def FinalPredict(Model:dict, Testing_Dataframe, Testing_Data_Features, Num_Samples=200):
    
    x = np.linspace(0, 7, 7000, endpoint=True)
    
    Result_DF = pd.DataFrame()
    Result_DF['ID'] = Testing_Dataframe['ID'].values
    Result_DF['RA'] = Testing_Dataframe['RA_1'].values
    Result_DF['DEC'] = Testing_Dataframe['DEC_1'].values
    
    Samples_Weights = []
    Samples_Means = []
    Samples_STDs = []
    for i in tqdm(range(Num_Samples)):
        
        Pred = Model(Testing_Data_Features)
        Weight = Pred.submodules[1].probs_parameter().numpy()
        Mean = Pred.submodules[0].mean().numpy().reshape(len(Testing_Data_Features), np.shape(Weight)[1])
        Std = Pred.submodules[0].stddev().numpy().reshape(len(Testing_Data_Features), np.shape(Weight)[1])
        
        Sorted_Weight_Index = np.flip(np.argsort(Weight, axis=1), axis=1)
        Samples_Weights.append(Weight[np.arange(len(Weight))[:,None], Sorted_Weight_Index])
        Samples_Means.append(Mean[np.arange(len(Mean))[:,None], Sorted_Weight_Index])
        Samples_STDs.append(Std[np.arange(len(Std))[:,None], Sorted_Weight_Index])

    Final_Weights = np.median(Samples_Weights, axis=0)
    Final_Means = np.median(Samples_Means, axis=0)
    Final_STDs = np.median(Samples_STDs, axis=0)
    
    del Samples_Weights
    del Samples_Means
    del Samples_STDs
    gc.collect()

    PDF = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(probs=Final_Weights),
          components_distribution=tfd.Normal(loc=Final_Means, scale=Final_STDs))
    PDF_STDs = PDF.stddev().numpy()

    Fine_ZPhot = []
    Final_PDFs = []
    Oddss = []
    Q68_Lower = []
    Q68_Higher = []
    Q95_Lower = []
    Q95_Higher = []
    for i in range(len(Testing_Data_Features)):
        
        Obj_PDF = Calc_PDF(x, Final_Weights[i], Final_Means[i], Final_STDs[i])
        Final_PDFs.append(Obj_PDF)
        Approx_Zphot = x[np.argmax(Obj_PDF)]
        x_detailed = np.linspace(Approx_Zphot-0.025, Approx_Zphot+0.025, 400)
        Fine_ZPhot.append(x_detailed[np.argmax(
            Calc_PDF(x_detailed, Final_Weights[i], Final_Means[i], Final_STDs[i])
            )])

        Obj_CDF = integrate.cumtrapz(Obj_PDF, x, initial=0)
        Oddss.append(Odds(x, Obj_CDF, Fine_ZPhot[i]))
        Q68_Lower.append(Q(68, x, Obj_CDF))
        Q68_Higher.append(Q(68, x, Obj_CDF, lower=False))
        Q95_Lower.append(Q(95, x, Obj_CDF))
        Q95_Higher.append(Q(95, x, Obj_CDF, lower=False))
        
    Sec_ZPhot = []
    Num_Peaks = []
    for i in range(len(Final_PDFs)):
        Diff = np.diff(Final_PDFs[i])/np.diff(x)
        Peak_Idx = np.where(np.diff(np.sign(Diff)) == -2)[0]

        if np.sum(Final_PDFs[i][Peak_Idx] >= np.max(Final_PDFs[i])/3) >= 2:

            Num_Peaks.append(np.sum(Final_PDFs[i][Peak_Idx] >= np.max(Final_PDFs[i])/3))
            Sorted_Idxs = Peak_Idx[np.argsort(Final_PDFs[i][Peak_Idx])[::-1]]

            if np.count_nonzero(Final_PDFs[i][Peak_Idx] >= np.max(Final_PDFs[i])/3) >= 2:
                Sorted_Idxs = Sorted_Idxs[:2]
                Sec_ZPhot.append(x[Sorted_Idxs[-1]])

        else:
            Num_Peaks.append(1)
            Sec_ZPhot.append(np.nan)

    Result_DF['zphot'] = Fine_ZPhot
    Result_DF['zphot_2.5p'] = Q95_Lower
    Result_DF['zphot_16p'] = Q68_Lower
    Result_DF['zphot_84p'] = Q68_Higher
    Result_DF['zphot_97.5p'] = Q95_Higher
    Result_DF['pdf_peaks'] = Num_Peaks
    Result_DF['zphot_second_peak'] = Sec_ZPhot
    Result_DF['pdf_width'] = PDF_STDs
    Result_DF['odds'] = Oddss
        
    for i in range(len(Final_Weights[0])):
        Result_DF[f'pdf_weight_{i}'] = np.round(Final_Weights[:,i], 8)
    for i in range(len(Final_Means[0])):
        Result_DF[f'pdf_mean_{i}'] = np.round(Final_Means[:,i], 8)
    for i in range(len(Final_STDs[0])):
        Result_DF[f'pdf_std_{i}'] = np.round(Final_STDs[:,i], 8)
        
    Result_DF.fillna(-99, inplace=True)
    Result_DF = Result_DF.astype({'ID': str, 'RA': np.float64, 'DEC': np.float64, 'zphot': np.float32,
                                  'zphot_2.5p': np.float32, 'zphot_16p': np.float32,
                                  'zphot_84p': np.float32, 'zphot_97.5p': np.float32,
                                  'pdf_peaks': np.float32, 'zphot_second_peak': np.float32,
                                  'pdf_width': np.float32, 'odds': np.float32})

    return Result_DF
