import numpy as np
import warnings

## Single-point estimate metrics ##

def mse(y, y_pred) -> float:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            dy = y - y_pred
            mse = np.sum(dy**2) / len(y)
        return mse
    except:
        return np.nan


def rmse(y, y_pred) -> float:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  
            rmse = np.sqrt(mse(y, y_pred))
        return rmse
    except:
        return np.nan


def nmad(y, y_pred) -> float:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            dy = y - y_pred   
            sigma = 1.48 * np.median(np.abs(dy-np.median(dy)) / (1+y))
        return sigma
    except:
        return np.nan


def bias(y, y_pred) -> float:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  
            dy = y - y_pred
            bias = np.mean(dy)
        return bias
    except:
        return np.nan


def out_frac(y, y_pred, cut=0.30) -> float:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  
            dy = y - y_pred
            dy_norm = dy / (1 + y)
            eta = np.count_nonzero(np.abs(dy_norm) > cut) / len(y)
        return eta
    except:
        return np.nan


## BMDN ##

def find_nearest_idx(array, value):
    '''General function to find the nearest idx of an item in a list
    https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array'''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def Odds(x, cdf_object, zphot):
    '''arXiv 9811189, eq. 17
    Also calculated as the integral of the PDF between z_peak +/- 0.02'''
    plus = cdf_object[find_nearest_idx(x, zphot+0.1)]
    minus = cdf_object[find_nearest_idx(x, zphot-0.1)]
    return plus - minus


def PIT(x, cdf_object, zspec):
    '''arXiv 1608.08016, eq. 2'''
    return cdf_object[find_nearest_idx(x, zspec)]


def step(x, y):
    '''Step function'''
    return 1 * (x > y)


def CRPS(x, cdf_object, zspec):
    '''arXiv 1608.08016, eq. 4'''
    return np.trapz( (cdf_object - step(x, zspec))**2, x )


def Check_Intervals(x):
    '''Calculate HPDCI per object
    https://stackoverflow.com/questions/33345780/empirical-cdf-in-python-similiar-to-matlabs-one'''
    List, last = [[]], None
    for elem in x:
        if last is None or abs(last - elem) <= 1:
            List[-1].append(elem)
        else:
            List.append([elem])
        last = elem
    return List


def HPDCI(x, pdf_object, zspec):
    '''arXiv 1601.07857'''
    HPDCI_Indexes = list(np.where(pdf_object >= pdf_object[find_nearest_idx(x, zspec)])[0])
    HPDCI_Indexes = Check_Intervals(HPDCI_Indexes)

    Object_HPDCI = 0
    for k in range(len(HPDCI_Indexes)):
        Object_HPDCI += np.trapz(pdf_object[HPDCI_Indexes[k]], x[HPDCI_Indexes[k]])

    return Object_HPDCI


def Q(alpha:int, x, cdf_object, lower=True):
    '''Calculate the alpha% credible interval.'''
    q_val = (1 - (alpha/100)) / 2
    if lower: return x[find_nearest_idx(cdf_object, q_val)]
    else: return x[find_nearest_idx(cdf_object, 1-q_val)]


## Printing functions ##

def print_metrics_xval(z, idx_per_fold=None, r=4):
    
    rmse_list = []
    sigma_list = []
    bias_list = []
    n30_list = []
    n15_list = []

    for fold in z.fold.unique():
        if idx_per_fold:
            z_copy = z.loc[idx_per_fold["valf"+str(fold)]]
        else:
            z_copy = z       

        rmse_list.append(rmse(z_copy[z_copy["fold"]==fold].Z, z_copy[z_copy["fold"]==fold].z_pred))
        sigma_list.append(nmad(z_copy[z_copy["fold"]==fold].Z, z_copy[z_copy["fold"]==fold].z_pred))
        bias_list.append(bias(z_copy[z_copy["fold"]==fold].Z, z_copy[z_copy["fold"]==fold].z_pred))
        n30_list.append(out_frac(z_copy[z_copy["fold"]==fold].Z, z_copy[z_copy["fold"]==fold].z_pred, 0.3))
        n15_list.append(out_frac(z_copy[z_copy["fold"]==fold].Z, z_copy[z_copy["fold"]==fold].z_pred, 0.15))

    print('RMSE', np.round(np.mean(rmse_list), r), np.round(np.std(rmse_list), r))
    print('NMAD', np.round(np.mean(sigma_list), r),  np.round(np.std(sigma_list), r))
    print('bias', np.round(np.mean(bias_list), r),  np.round(np.std(bias_list), r))
    print('n15', np.round(np.mean(n15_list), r),  np.round(np.std(n15_list), r))
    print('n30', np.round(np.mean(n30_list), r),  np.round(np.std(n30_list), r))
    return


def print_metrics_test(z_true, z_pred, r=4):
    print('RMSE', np.round(rmse(z_true,z_pred), r))
    print('NMAD', np.round(nmad(z_true,z_pred), r))
    print('bias', np.round(bias(z_true,z_pred), r))
    print('n15', np.round(out_frac(z_true,z_pred, 0.15), r))
    print('n30', np.round(out_frac(z_true,z_pred, 0.30), r))
    return
