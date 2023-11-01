import numpy as np
import pandas as pd
from settings.columns import wise, galex, splus

pd.set_option('mode.chained_assignment',None)

def prep_data(dataframe:pd.core.frame.DataFrame, aper, rmax = "None", rmin = "None", zmax = "None", zmin = "None", val_mb="None"):
    df = dataframe.copy()
    
    if rmax !=  "None":
        df = df.loc[(df['r'+'_'+aper]<=rmax)|(df['r'+'_'+aper]==99)]

    if rmin !=  "None":
        df = df.loc[(df['r'+'_'+aper]>rmin)|(df['r'+'_'+aper]==99)]

    if zmax !=  "None":
        df = df.loc[(df['Z']<=zmax)]
    
    if zmin !=  "None":
        df = df.loc[(df['Z']>zmin)]

    if val_mb != "None":
        df[wise] = df[wise].replace(-1, val_mb)
        df[wise+galex] = df[wise+galex].fillna(value=val_mb)
        if val_mb != 99:
            df[splus] = df[splus].replace(99, val_mb)
        
    else:
        df = df.loc[(df['nDet'+'_'+aper]==12)&(df['b_x']==3)&(df['W1_MAG']!=-1)&(df['W2_MAG']!=-1)].dropna(subset=wise+galex)
        
    return df


def create_bins(data:pd.core.frame.DataFrame, return_data = False, var = "Z", bin_size=0.5, bins="None"):

    if bins == "None":
        xmin = np.floor(np.min(data[var]))
        xmax = np.ceil(np.max(data[data[var]!=99][var]))
        if xmax - np.max(data[data[var]!=99][var]) > bin_size:
            xmax = xmax - bin_size
        bins = np.arange(xmin, xmax+bin_size, bin_size)

    if isinstance(data, pd.core.frame.DataFrame):
        # itv = pd.cut(data[var], bins=bins, labels=range(int(2*xmax)))
        itv = pd.cut(data[var], bins=bins, labels=range(len(bins)-1))
    elif isinstance(data, pd.core.frame.Series):
        itv = pd.cut(data, bins=bins, labels=range(len(bins)-1))
    else:
        print('data must be DataFrame or Series.')
        itv = pd.Series()
    
    if return_data:
        # data.insert(loc=310, column=var+'class', value=itv)
        data[var+'class'] = itv
        return data, bins, itv
    else:
        return bins, itv



def rename_aper(data):
    list_aper = ['iso','aper_3', 'aper_6', 'PStotal', 'auto', 'petro']
    for aper in list_aper:
        data.rename(columns={"PhotoFlag_R": "PhotoFlag_r", 
                        "U_"+aper: "u_"+aper,
                        "G_"+aper: "g_"+aper,
                        "R_"+aper: "r_"+aper,
                        "I_"+aper: "i_"+aper,
                        "Z_"+aper: "z_"+aper,
                        "F378_"+aper: "J0378_"+aper,
                        "F395_"+aper: "J0395_"+aper,
                        "F410_"+aper: "J0410_"+aper,
                        "F430_"+aper: "J0430_"+aper,
                        "F515_"+aper: "J0515_"+aper,
                        "F660_"+aper: "J0660_"+aper,
                        "F861_"+aper: "J0861_"+aper,
                        "e_U_"+aper: "e_u_"+aper,
                        "e_G_"+aper: "e_g_"+aper,
                        "e_R_"+aper: "e_r_"+aper,
                        "e_I_"+aper: "e_i_"+aper,
                        "e_Z_"+aper: "e_z_"+aper,
                        "e_F378_"+aper: "e_J0378_"+aper,
                        "e_F395_"+aper: "e_J0395_"+aper,
                        "e_F410_"+aper: "e_J0410_"+aper,
                        "e_F430_"+aper: "e_J0430_"+aper,
                        "e_F515_"+aper: "e_J0515_"+aper,
                        "e_F660_"+aper: "e_J0660_"+aper,
                        "e_F861_"+aper: "e_J0861_"+aper,
                        "s2n_U_"+aper: "s2n_u_"+aper,
                        "s2n_G_"+aper: "s2n_g_"+aper,
                        "s2n_R_"+aper: "s2n_r_"+aper,
                        "s2n_I_"+aper: "s2n_i_"+aper,
                        "s2n_Z_"+aper: "s2n_z_"+aper,
                        "s2n_F378_"+aper: "s2n_J0378_"+aper,
                        "s2n_F395_"+aper: "s2n_J0395_"+aper,
                        "s2n_F410_"+aper: "s2n_J0410_"+aper,
                        "s2n_F430_"+aper: "s2n_J0430_"+aper,
                        "s2n_F515_"+aper: "s2n_J0515_"+aper,
                        "s2n_F660_"+aper: "s2n_J0660_"+aper,
                        "s2n_F861_"+aper: "s2n_J0861_"+aper
                        }, inplace=True)
    return data