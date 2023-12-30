import pandas as pd
import numpy as np
import warnings

def nmad(y:pd.core.frame.Series, y_pred:pd.core.frame.Series) -> float:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')    
            nmad = 1.48 * np.median(np.abs(y_pred-y - np.median(y_pred-y)) / (1+y))
        return nmad
    except:
        return np.nan

def bias(y:pd.core.frame.Series, y_pred:pd.core.frame.Series) -> float:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  
            bias = np.mean(y-y_pred)
        return bias
    except:
        return np.nan

def out_frac(y:pd.core.frame.Series, y_pred:pd.core.frame.Series, cut = 0.30) -> float:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  
            eta = np.abs(y_pred-y)/(1+y)
            frac = np.count_nonzero( eta > cut ) / len(y)
        return frac
    except:
        return np.nan 


def mse(y:pd.core.frame.Series, y_pred:pd.core.frame.Series) -> float:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  
            mse = np.sum((y_pred-y)**2) / len(y)
        return mse
    except:
        return np.nan

def rmse(y:pd.core.frame.Series, y_pred:pd.core.frame.Series) -> float:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  
            rmse = np.sqrt(mse(y,y_pred))
        return rmse
    except:
        return np.nan 
