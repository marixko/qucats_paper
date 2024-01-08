from os import listdir

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from matplotlib.gridspec import GridSpec
from matplotlib import cm
from IPython.display import display

from utils.metrics import nmad, bias, rmse, out_frac as outf


def ecdf(sample) -> tuple:
    '''Used in HPDCI_Plot
    https://stackoverflow.com/questions/33345780/empirical-cdf-in-python-similiar-to-matlabs-one'''
    
    # convert sample to a numpy array, if it isn't already
    sample = np.atleast_1d(sample)
    # find the unique values and their corresponding counts
    quantiles, counts = np.unique(sample, return_counts=True)
    # take the cumulative sum of the counts and divide by the sample size to
    # get the cumulative probabilities between 0 and 1
    cumprob = np.cumsum(counts).astype(np.double) / sample.size
    return quantiles, cumprob


def calc_ratios(Results_DF, odds_bins, delta_value) -> list:
    '''Used in Odds_PIT_CRPS_HPDCI_Delta'''
    ratios = []
    for value in odds_bins:
        odds_cond = Results_DF['Odds'] >= value
        ratios.append(100*(np.count_nonzero(
            Results_DF[odds_cond]['Delta'] <= delta_value
            ) / len(Results_DF[odds_cond]['Delta'])
                           ))
    return ratios


def do_bins_grid(vars_dict:dict) -> tuple:
    '''Used in Result_Bins'''
    
    bins_dict = {}
    for var in vars_dict.keys():
        bins_dict[var] = np.linspace(vars_dict[var][0], vars_dict[var][1], vars_dict[var][2], endpoint=False)
    
    grid_dict = {}
    for var in bins_dict.keys():
        grid_dict[var] = np.diff(bins_dict[var])[0]
    
    return bins_dict, grid_dict


def do_col_orders(metrics:list, cols:list) -> dict:
    '''Used in Limited_Metrics'''
    orders = {}
    other_metrics = [item for item in metrics if item not in cols]
    for i, col in enumerate(cols):
        temp_cols = cols.copy()
        del temp_cols[i]
        new_cols = [col] + temp_cols
        orders[col] = new_cols + other_metrics    
    
    return orders


def random_index(Results_DF, Aper, r_conds, z_conds, seed) -> list:
    '''Used in PDF_Sample'''
    idxs = np.zeros((len(r_conds), len(z_conds)))

    np.random.seed(seed)
    for i in range(len(r_conds)):
        df_r = Results_DF[[f'r_{Aper}', 'z']].query(r_conds[i])
        
        for j in range(len(z_conds)):
            df_conds = df_r.query(z_conds[j])
            
            try:
                idxs[i][j] = np.random.choice(df_conds.index)
            except ValueError:
                print(f'No objects found for query ({r_conds[i]}) & ({z_conds[j]})')
                idxs[i][j] = np.nan
    
    idxs = idxs[:, ~np.all(np.isnan(idxs), axis=0)]
    idxs = idxs[~np.all(np.isnan(idxs), axis=1), :]
    idxs = idxs.astype(int)
    
    return idxs


def Calculate_Metric(Spec_Z, Photo_Z, R_Mag, Aper:str, Bins_Dict:dict, Grid_Dict:dict,
                     Metric=None, Bins=None, Cumulative=False) -> list:
    
    Spec_Z = pd.Series(Spec_Z).reset_index(drop=True)
    Photo_Z  = pd.Series(Photo_Z).reset_index(drop=True)
    R_Mag = pd.Series(R_Mag).reset_index(drop=True)
    Results = []
    
    if Bins == 'z':
        obj = Spec_Z
    elif Bins == 'r_'+Aper:
        obj = R_Mag

    if Cumulative == False:
        for bin_val in Bins_Dict[Bins]:
            
            mask = obj.between(bin_val, bin_val+Grid_Dict[Bins])
            Spec_Z_Bin = Spec_Z[mask]
            Phot_Z_Bin = Photo_Z[mask]
            
            if Metric == 'nmad':
                Results.append(nmad(Spec_Z_Bin, Phot_Z_Bin))
            elif Metric == 'bias':
                Results.append(bias(Spec_Z_Bin, Phot_Z_Bin))
            elif Metric == 'rmse':
                Results.append(rmse(Spec_Z_Bin, Phot_Z_Bin))
            elif Metric == 'outf15':
                Results.append(outf(Spec_Z_Bin, Phot_Z_Bin, 0.15))
            elif Metric == 'outf30':
                Results.append(outf(Spec_Z_Bin, Phot_Z_Bin, 0.3))
            elif Metric == 'numb':
                Results.append(len(Phot_Z_Bin - Spec_Z_Bin))

    if Cumulative == True:
        for i in range(1, len(Bins_Dict[Bins])+1):
            
            mask = obj.between(Bins_Dict[Bins][0], Bins_Dict[Bins][0]+Grid_Dict[Bins]*i)
            Spec_Z_Bin = Spec_Z[mask]
            Phot_Z_Bin = Photo_Z[mask]
            
            if Metric == 'nmad':
                Results.append(nmad(Spec_Z_Bin, Phot_Z_Bin))
            elif Metric == 'bias':
                Results.append(bias(Spec_Z_Bin, Phot_Z_Bin))
            elif Metric == 'rmse':
                Results.append(rmse(Spec_Z_Bin, Phot_Z_Bin))
            elif Metric == 'outf15':
                Results.append(outf(Spec_Z_Bin, Phot_Z_Bin, 0.15))
            elif Metric == 'outf30':
                Results.append(outf(Spec_Z_Bin, Phot_Z_Bin, 0.3))
            elif Metric == 'numb':
                Results.append(len(Phot_Z_Bin - Spec_Z_Bin))

    return Results


def Histogram_Descr(Train, Test, Aper:str, Output_Dir:str, Save=True, Show=False, Close=True):
    
    Bins = {'Z': np.arange(0, 7.5, 0.5),
            'r_'+Aper: np.arange(16, 22.5, 0.5)}
    n_Bins = len(Bins.keys())
    
    fig, axes = plt.subplots(nrows=1, ncols=n_Bins, figsize=(14, 5), sharey='row')
    plt.subplots_adjust(wspace=0.1)
    colors = [('#D3B7E7', '#FFC171'), ('#D3B7E7', '#FFC171')]
    
    i = 0
    xlabels = ['$z_{spec}$', 'r']
    for col, bins in Bins.items():
        axes[i].hist(Train[col], edgecolor='gray', bins=bins, log=True, color=colors[i][0], label='Train', alpha=1)
        axes[i].hist(Test[col], edgecolor='gray', bins=bins, log=True, color=colors[i][1], label='Test', alpha=1)
        axes[i].set_xlabel(xlabels[i])
        axes[i].yaxis.set_tick_params(labelbottom=True)
        axes[i].grid(False)
        i += 1
    axes[0].set_ylabel('Number of objects')
    plt.legend(loc='center', ncol=2, bbox_to_anchor=[0.5, 0.93], bbox_transform=fig.transFigure)

    if Save:
        for col in Bins.keys():
            plt.savefig(Output_Dir+f'Histogram_{"-".join(Bins.keys())}.png', bbox_inches='tight', dpi=500)
    if Show: plt.show()
    if Close: plt.close()


def Z_Mag(Results_DF, Aper:str, Output_Dir:str, Save=True, Show=False, Close=True):
    
    fig = plt.figure(figsize=(9, 8))
    gs = GridSpec(ncols=2, nrows=1, figure=fig, width_ratios=[1, 0.1])
    ax = fig.add_subplot(gs[0])
    
    im = ax.hexbin(Results_DF['Z'], Results_DF['r_'+Aper], norm=clr.LogNorm(vmax=300),
                            gridsize=140, mincnt=1, cmap='viridis', extent=(0, 7, 15, 22))
    ax.grid(False)
    ax.set_xlim(0, 7)
    ax.set_ylim(15, 22)
    ax.set_xlabel('$z_{spec}$')
    ax.set_ylabel('r')
    
    ax_cb = fig.add_subplot(gs[1])
    plt.colorbar(im, cax=ax_cb)
    ax_cb.set_ylabel('Número de objetos')

    if Save: plt.savefig(Output_Dir+f'Scatter_Zmag.png', bbox_inches='tight')
    if Show: plt.show()
    if Close: plt.close()


def Histograms(Results_DF, PDF_List, x, Z_Lim:float, Output_Dir:str, DarkMode=False, Save=True, Show=False, Close=True):
    
    _, _ = plt.subplots(figsize=(10, 7))

    plt.hist(Results_DF['z'], density=True, histtype='stepfilled',
             lw=2, bins=100, alpha=0.3, range=(0, Z_Lim), label='Spectroscopic', zorder=0)
    plt.hist(Results_DF['zphot'], density=True, histtype='step',
             lw=2, bins=100, alpha=1, range=(0, Z_Lim), label='Photometric', zorder=2)
    plt.plot(x, np.sum(PDF_List, axis=0)/np.trapz(np.sum(PDF_List, axis=0), x), ls='--',
             lw=2, label='PDF sum', zorder=1)

    plt.xticks(np.arange(0, Z_Lim+0.5, 0.5))
    plt.xlim(0, Z_Lim)

    legend = plt.legend()
    if DarkMode: plt.setp(legend.get_texts(), color='w')

    plt.xlabel('Redshift')

    if Save: plt.savefig(Output_Dir+'Histogram_Redshift.png', bbox_inches='tight')
    if Show: plt.show()
    if Close: plt.close()


def PDF_Bins_RA(Results_DF, PDF_List, x, Z_Lim:float, Output_Dir:str, Save=True, Show=False, Close=True):
    
    # Creating a dataframe for PDFs
    PDF_DF = pd.DataFrame(PDF_List)

    # Joining the Results and PDF dataframes
    Teste_DF = Results_DF[['RA', 'DEC']].reset_index(drop=True
                                                            ).join(PDF_DF).join(Results_DF[['z', 'zphot']])

    # Transforming RA from deg to rad
    Teste_DF['RA_rad'] = Teste_DF['RA'] * np.pi/180

    # 'Fixing' the RA coords
    Teste_DF['RA_rad'][Teste_DF['RA_rad'].between(4, 10)] = \
        Teste_DF['RA_rad'][Teste_DF['RA_rad'].between(4, 10)]-np.max(Teste_DF['RA_rad'])
    Teste_DF['RA'][Teste_DF['RA'].between(300, 360)] = \
        Teste_DF['RA'][Teste_DF['RA'].between(300, 360)]-360

    # For Dec < 0
    fig = plt.figure(figsize=(20, 10))
    plt.subplots_adjust(wspace=0)
    ax = fig.add_subplot(121, projection='3d')

    Data = Teste_DF[(Teste_DF['DEC'] < 0 & Teste_DF['RA'].between(-1*180/np.pi, 1*180/np.pi))]

    RA_Range = np.arange(np.min(Data['RA']), np.max(Data['RA']), 0.05*180/np.pi)

    for i in RA_Range:
        Idxs = Data['RA'].between(i, i+0.05*180/np.pi)
        xs = [Data['RA'][Idxs].mean()] * len(x[x <= Z_Lim])
        ys = x[x <= Z_Lim]
        zs = Data[np.arange(0, len(x[x <= Z_Lim]), 1)].values[Idxs].mean(axis=0)#/Max
        ax.plot(xs, ys, zs, lw=2)

    ax.set_ylim(0, Z_Lim)
    ax.set_xlabel('RA (rad)'); ax.set_ylabel('Photometric redshift'); ax.set_zlabel('PDF(z)')
    ax.tick_params(axis='both', width=10, pad=0)
    ax.set_title('DEC < 0')

    # For Dec > 0
    ax = fig.add_subplot(122, projection='3d')
    Data = Teste_DF[(Teste_DF['DEC'] > 0 & Teste_DF['RA'].between(-1*180/np.pi, 1*180/np.pi))]
    RA_Range = np.arange(np.min(Data['RA']), np.max(Data['RA']), 0.05*180/np.pi)

    for i in RA_Range:
        Idxs = Data['RA'].between(i, i+0.05*180/np.pi)
        xs = [Data['RA'][Idxs].mean()] * len(x[x <= Z_Lim])
        ys = x[x <= Z_Lim]
        zs = Data[np.arange(0, len(x[x <= Z_Lim]), 1)].values[Idxs].mean(axis=0)#/Max
        ax.plot(xs, ys, zs, lw=2)

    ax.set_ylim(0, Z_Lim)
    ax.set_xlabel('RA (rad)')
    ax.set_ylabel('Photometric redshift')
    ax.set_zlabel('PDF(z)')
    ax.set_title('DEC > 0')

    if Save: plt.savefig(Output_Dir+'PDF_in_bins_RA.png', bbox_inches='tight')
    if Show: plt.show()
    if Close: plt.close()


def LSS_2D(Results_DF, Z_Lim:float, Output_Dir:str, Save=True, Show=False, Close=True):
    
    _, ax = plt.subplots(1, 2, figsize=(12, 12), subplot_kw=dict(projection='polar'))
    plt.subplots_adjust(wspace=0.3)

    ax[0].set_ylim(0, Z_Lim)
    ax[0].set_yticks(np.arange(0, Z_Lim+0.5, 0.5))
    ax[0].set_yticklabels(['%.1f      ' %s for s in np.arange(0, Z_Lim+0.5, 0.5)], verticalalignment='center')
    ax[0].set_theta_offset(45*np.pi/180)
    ax[0].scatter(Results_DF['RA']*np.pi/180, Results_DF['z'], s=1, alpha=0.2)
    ax[0].set_thetamax(-60)
    ax[0].set_thetamin(60)
    ax[0].set_title('Sample with z')

    ax[1].set_ylim(0, Z_Lim)
    ax[1].set_yticks(np.arange(0, Z_Lim+0.5, 0.5))
    ax[1].set_yticklabels(['%.1f      ' %s for s in np.arange(0, Z_Lim+0.5, 0.5)], verticalalignment='center')
    ax[1].set_theta_offset(45*np.pi/180)
    ax[1].scatter(Results_DF['RA']*np.pi/180, Results_DF['zphot'], s=1, alpha=0.2)
    ax[1].set_thetamax(-60)
    ax[1].set_thetamin(60)
    ax[1].set_title('Sample with zphot')

    ax[0].grid(False)
    ax[1].grid(False)

    if Save: plt.savefig(Output_Dir+'LSS.png', dpi=300, bbox_inches='tight')
    if Show: plt.show()
    if Close: plt.close()


def Average_PDFs(Results_DF, PDF_List, x, Aper:str, Z_Lim:float, Output_Dir:str, DarkMode=False,
                 Save=True, Show=False, Close=True):
    
    Bins_Mag_Av = np.arange(16, 22, 1)
    Bins_Z_Av = np.arange(0, Z_Lim, 0.5)
    nbins = 100

    _, ax = plt.subplots(2, 1, figsize=(14, 10))
    plt.subplots_adjust(hspace=0.3)

    Colours = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11']

    col_idx = 0
    for val in Bins_Mag_Av:
        Cond = Results_DF['r_'+Aper].between(val, val+1)
        PDFs_in_bin = np.mean(np.array(PDF_List)[Cond], axis=0)

        ax[0].hist(Results_DF['z'][Cond], range=(0, Z_Lim),
                   bins=nbins, density=True, lw=2, color=Colours[col_idx], alpha=0.3)
        ax[0].plot(x, PDFs_in_bin, lw=3, color=Colours[col_idx],
                   label=f'{val:.1f} - {val+1:.1f} ({len(np.array(Results_DF["r_"+Aper])[Cond])})')
        col_idx += 1
    
    legend = ax[0].legend(ncol=2)
    if DarkMode: plt.setp(legend.get_texts(), color='w')
    ax[0].set_xlim(0, Z_Lim)
    ax[0].set_xticks(np.arange(0, Z_Lim+0.5, 0.5))
    ax[0].set_title(f'Average PDFs as a function of r_{Aper}')
    ax[0].set_xlabel('Photometric redshift')

    col_idx = 0
    for val in Bins_Z_Av:
        Cond = Results_DF['z'].between(val, val+0.5)
        PDFs_in_bin = np.mean(np.array(PDF_List)[Cond], axis=0)

        ax[1].hist(Results_DF['z'][Cond], range=(0, Z_Lim),
                   bins=nbins, density=True, lw=2, color=Colours[col_idx], alpha=0.3)
        ax[1].plot(x, PDFs_in_bin, lw=3, color=Colours[col_idx],
                   label=f'{val:.1f} - {val+0.5:.1f} ({len(np.array(Results_DF["z"])[Cond])})')
        col_idx += 1
    
    legend = ax[1].legend(ncol=2)
    if DarkMode: plt.setp(legend.get_texts(), color='w')
    ax[1].set_xlim(0, Z_Lim)
    ax[1].set_xticks(np.arange(0, Z_Lim+0.5, 0.5))
    ax[1].set_title('Average PDFs as a function of spectroscopic redshift')
    ax[1].set_xlabel('Photometric redshift')

    if Save: plt.savefig(Output_Dir+'Average_PDFs.png', bbox_inches='tight')
    if Show: plt.show()
    if Close: plt.close()


def Average_Z_PDFs(Results_DF, Train_DF, PDF_List, x, Z_Lim:float, Output_Dir:str, Save=False, Show=True, Close=True):
    
    Bins_Z_Av = np.arange(0, Z_Lim, 0.5)
    nbins = 100

    fig, ax = plt.subplots(figsize=(15, 9))
    plt.subplots_adjust(hspace=0.3)

    Colours = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11']

    col_idx = 0
    for val in Bins_Z_Av:
        Cond_test = Results_DF['z'].between(val, val+0.5)
        Cond_train = Train_DF['Z'].between(val, val+0.5)
        PDFs_in_bin = np.mean(np.array(PDF_List)[Cond_test], axis=0)

        ax.hist(Results_DF['z'][Cond_test], range=(0, Z_Lim),
                   bins=nbins, density=True, lw=2, color=Colours[col_idx], alpha=0.3)
        ax.plot(x, PDFs_in_bin, lw=3, color=Colours[col_idx],
                   label=f'{val:.1f} - {val+0.5:.1f} ({len(np.array(Train_DF["Z"])[Cond_train])})')
        col_idx += 1
    
    ax.legend(loc='center', ncol=5, bbox_to_anchor=[0.512, 0.94], bbox_transform=fig.transFigure,
              title='Redshift espectroscópico')
    ax.set_xlim(0, Z_Lim)
    ax.set_ylim(0, 5.5)
    ax.set_xticks(np.arange(0, Z_Lim+0.5, 0.5))
    ax.set_ylabel('Densidade')
    ax.set_xlabel('$z_{phot}$')

    if Save: plt.savefig(Output_Dir+'Average_Z_PDFs.png', bbox_inches='tight')
    if Show: plt.show()
    if Close: plt.close()


def PDF_Sample(Results_DF, PDFs_Dict:dict, x, Aper:str, Seed:int, Output_Dir:str,
               DarkMode=False, Save=True, Show=False, Close=True, idxs=None):
    
    r_conds = [f'r_{Aper} < 20', f'20 < r_{Aper} < 21.3', f'r_{Aper} > 21.3']
    z_conds = ['z < 0.5', '0.5 < z < 3.5', '3.5 < z < 5']
    if idxs is None:
        idxs = random_index(Results_DF, Aper, r_conds, z_conds, Seed)
    n_r = idxs.shape[0]
    n_z = idxs.shape[1]
    
    colors = ['#e69d00', '#009e74', '#56b3e9', '#cc79a7', '#0071b2', '#d54b00']
    fig, axes = plt.subplots(nrows=n_r, ncols=n_z, figsize=(3.6*n_z, 2.4*n_r))
    for i in range(n_r):
        for j in range(n_z):
            ax = axes[i][j]
            idx = idxs[i][j]
            
            try:
                name = Results_DF.loc[idx, 'SDSS_NAME']
                ax.plot([], label=f'SDSS J{name}')
                leg = ax.legend(loc='upper right', fontsize=10,
                                handlelength=0, handletextpad=0, borderpad=0.1, framealpha=0.7)
                leg.get_frame().set_linewidth(0.5)
            except:
                ra = round(Results_DF.loc[idx, 'RA'], 2)
                dec = round(Results_DF.loc[idx, 'DEC'], 3)
                ax.plot([], label=f'RA: {ra}\nDec: {dec}')
                ax.legend(handlelength=0, handletextpad=0)
            
            for k, name in enumerate(PDFs_Dict):
                ax.plot(x, PDFs_Dict[name][idx], label=name, c=colors[k])
            ax.axvline(Results_DF.loc[idx, 'z'], ls='--', c='k', label='$z_{spec}$')
            ax.set_xlim(0, 5)
            ax.set_xticks(range(6))
            
            if i == 0:
                ax.set_title(z_conds[j].replace('z', '$z_{spec}$'))
            if n_r-i == 1:
                ax.set_xlabel('$z_{phot}$')
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel('$p(z_{phot})$')
            elif n_z-j == 1:
                ax2 = ax.twinx()
                ax2.set_yticks([])
                ax2.set_ylabel(r_conds[i].replace(f'r_{Aper}', '$r$'), rotation=270, labelpad=16, size=16)
                
    handles, labels = ax.get_legend_handles_labels()
    legend = fig.legend(handles=handles[1:], labels=labels[1:], loc='lower center',
                        bbox_to_anchor=(0.5, -0.05), ncol=1+len(PDFs_Dict))
    fig.align_ylabels()
    fig.tight_layout()
    if DarkMode: plt.setp(legend.get_texts(), color='w')
    if Save: plt.savefig(Output_Dir+'PDF_Samples.png', bbox_inches='tight', dpi=500)
    if Show: plt.show()
    if Close: plt.close()


def Odds_Hist(Results_DF, Aper:str, Z_Lim:float, Output_Dir:str, ax=None, plot_Folds=False,
              Save=True, Show=False, Close=True):
    
    if not ax: _, ax = plt.subplots(figsize=(8, 8))
    cond = Results_DF['z'] <= Z_Lim
    
    viridis = cm.get_cmap('viridis', 40)
    N, bins, patches = ax.hist(Results_DF['Odds'][cond], range=(0, 1), bins=40, histtype='step', lw=2)
    X_std = (Results_DF[f'r_{Aper}'][cond] - Results_DF[f'r_{Aper}'][cond].min(axis=0)
             ) / (Results_DF[f'r_{Aper}'][cond].max(axis=0) - Results_DF[f'r_{Aper}'][cond].min(axis=0))
    #ax.axvline(Results_DF['Odds'][cond].median(), c='k', ls='--')
    ax.set_xlim(0, 1)
    
    # for i in range(len(bins)-1):
    #     patches[i].set_facecolor(viridis(X_std[Results_DF['Odds'][cond].between(bins[i], bins[i+1])].median()))
    
    # for i in range(len(bins)-1):
    #     if i % 3 == 0:
    #         text = Results_DF[f'r_{Aper}'][cond & (Results_DF['Odds'][cond].between(bins[i], bins[i+1]))].median()
    #         ax.text((bins[i] + bins[i+1])/2, N[i]+0.9, f'{text:.2f}',
    #                 rotation=90, ha='center', va='top', color='black')
    #         ax.plot([(bins[i] + bins[i+1])/2, (bins[i] + bins[i+1])/2], [N[i]+0.05, N[i]+0.4], color='black', lw=0.5)

    if plot_Folds:
        Folds = [int(s[4:]) for s in listdir(Output_Dir+'SavedModels/')]  # [4:] removes 'Fold' from folder name
        for fold in Folds:
            ax.hist(Results_DF[f'Odds_{fold}'], density=True, range=(0, 1), bins=40,
                    histtype='step', lw=1, ls='--')

    #ax.set_ylim(0, np.max(N)+0.1)
    #ax.set_xlim(-0.02, 1.02)
    ax.set_ylabel('Frequência')
    ax.set_xlabel('Odds')
    
    if Save: plt.savefig(Output_Dir+'Odds.png', bbox_inches='tight')
    if Show: plt.show()
    if Close: plt.close()


def PIT_Hist(Results_DF, Z_Lim:float, Output_Dir:str, ax=None, plot_Folds=False, Save=True, Show=False, Close=True):
    
    if not ax: _, ax = plt.subplots(figsize=(8, 8))
    cond = Results_DF['z'] <= Z_Lim
    
    N, _, _ = ax.hist(Results_DF['PIT'][cond], range=(0, 1), bins=40, histtype='step', lw=2)
    #ax.axvline(Results_DF['PIT'][cond].median(), c='k', ls='--')
    ax.set_xlim(0, 1)

    if plot_Folds:
        Folds = [int(s[4:]) for s in listdir(Output_Dir+'SavedModels/')]  # [4:] removes 'Fold' from folder name
        for fold in Folds:
            ax.hist(Results_DF[f'PIT_{fold}'][cond], density=True, range=(0, 1), bins=40, histtype='step', ls='--')

    #ax.set_ylim(0, np.max(N)+0.1)
    #ax.set_xlim(-0.02, 1.02)
    #ax.set_ylabel('Relative Frequency')
    ax.set_xlabel('PIT')
    # ax.text(0.5, 0.95, f'Median PIT = {np.median(Results_DF["PIT"][cond]):.3f}',
    #          va='center', ha='center', transform=ax.transAxes, color='black')

    # To show a comparison with an uniform distribution:
    ax.axhline(204.17499999999998, c='k')
    # N_uniform, _ = np.histogram(np.random.uniform(0, 1, int(1e8)), density=True, bins=40)
    # ax.hist(np.random.uniform(0, 1, int(1e8)), density=True, range=(0, 1), bins=40, alpha=0.8, lw=1,
    #          histtype='step', color='k', label='Uniform distribution')
    # N_mean = np.mean(N**2)
    # N_std = np.std(N**2)
    # N_uniform_mean = np.mean(N_uniform**2)
    # N_uniform_std = np.std(N_uniform**2)
    # ax.text(0.5, 0.1, f'PIT$^2$:  {N_mean:.3f} ({N_std:.3f})\nUniform$^2$: {N_uniform_mean:.3f} ({N_uniform_std:.3f})', 
    #             fontsize=10, ha='center', va='bottom', bbox=dict(boxstyle='round', ec=(0.8, 0.8, 0.8), fc=(1, 1, 1)))
    
    if Save: plt.savefig(Output_Dir+'PIT.png', bbox_inches='tight', dpi=500)
    if Show: plt.show()
    if Close: plt.close()


def CRPS_Hist(Results_DF, Z_Lim:float, Output_Dir:str, ax=None, plot_Folds=False, DarkMode=False,
              Save=True, Show=False, Close=True):
    
    if not ax: _, ax = plt.subplots(figsize=(8, 8))
    cond = Results_DF['z'] <= Z_Lim
    
    N, _, _ = ax.hist(Results_DF['CRPS'][cond], range=(0, 1), bins=40, histtype='step', lw=2, label='Final')
    #ax.axvline(Results_DF['CRPS'][cond].median(), c='k', ls='--')
    ax.set_xlim(0, 1)

    if plot_Folds:
        Folds = [int(s[4:]) for s in listdir(Output_Dir+'SavedModels/')]  # [4:] removes 'Fold' from folder name
        if len(Folds) > 1:
            for fold in Folds:
                ax.hist(Results_DF[f'CRPS_{fold}'][cond], density=True, range=(0, 1), bins=40, ls='--',
                        histtype='step', label=f'Fold {fold}')

    #ax.set_yscale('log')
    #ax.set_xscale('log')
    #ax.set_ylim(0, np.max(N)+0.1)
    #ax.set_xlim(-0.02, 1.02)
    #ax.set_ylabel('Relative Frequency')
    ax.set_xlabel('CRPS')        
    if plot_Folds:
        legend = ax.legend(loc='center right', handlelength=0.6)
        if DarkMode: plt.setp(legend.get_texts(), color='w')

    text = np.median(Results_DF['CRPS'][cond])
    #ax.text(0.5, 0.95, f'Median CRPS = {text:.3f}', ha='center', va='center', transform=ax.transAxes, color='black')
    
    if Save: plt.savefig(Output_Dir+'CRPS.png', bbox_inches='tight')
    if Show: plt.show()
    if Close: plt.close()


def HPDCI_Plot(Results_DF, Aper:str, Output_Dir:str, ax=None, Save=True, Show=False, Close=True):
    
    if not ax: _, ax = plt.subplots(figsize=(12, 7))
    
    r_mags = [16, 17, 18, 19, 20, 21]
    Dashes = [[1, 0], [3, 1, 3, 1], [4, 1, 1, 1], [4, 1, 1, 1, 1, 1], [4, 1, 1, 1, 1, 1, 1, 1],
              [4, 1, 1, 1, 1, 1, 1, 1, 1, 1], [4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1]]

    c, f = ecdf(Results_DF['HPDCI'].values)
    ax.plot(c, f, label='All', dashes=Dashes[0], lw=1.5)
    dash_idx = 1
    for mag in r_mags:
        Cond = (Results_DF[f'r_{Aper}'] > mag) & (Results_DF[f'r_{Aper}'] <= mag+1)
        c, f = ecdf(Results_DF['HPDCI'][Cond].values)
        ax.plot(c, f, label=f'${mag}<r\leq{mag+1}$', dashes=Dashes[dash_idx], lw=1)
        dash_idx += 1

    ax.plot([0, 1], [0, 1], ls='--', color='black', lw=0.75, zorder=-1, label='Ideal')
    ax.set_ylabel('$\hat{F} \ (c)$')
    ax.set_xlabel('c')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right', fontsize=10)

    # Inset ax
    axins = ax.inset_axes([0.01, 0.5, 0.3, 0.47])
    axins.plot(c, f, label='All', dashes=Dashes[0], lw=2)
    dash_idx = 1

    for mag in r_mags:
        Cond = (Results_DF[f'r_{Aper}'] > mag) & (Results_DF[f'r_{Aper}'] <= mag+1)
        c, f = ecdf(Results_DF['HPDCI'][Cond].values)
        axins.plot(c, f, dashes=Dashes[dash_idx], lw=2)
        dash_idx += 1

    axins.plot([0, 1], [0, 1], ls='--', color='black', lw=1, zorder=-1)

    # sub region of the original image
    x1, x2, y1, y2 = 0.4, 0.6, 0.4, 0.6
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticklabels('')
    axins.set_yticklabels('')

    ax.indicate_inset_zoom(axins, edgecolor='black', lw=1)
    
    if Save: plt.savefig(Output_Dir+'HPDCI.png', bbox_inches='tight')
    if Show: plt.show()
    if Close: plt.close()


def Odds_PIT_CRPS_HPDCI_Delta(Results_DF, Aper:str, Z_Lim:float, Output_Dir:str, DarkMode=False, Save=True, Show=False, Close=True):
    
    fig = plt.figure(figsize=(15, 10))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    gs = GridSpec(2, 3, figure=fig)
    plot_Folds = False

    # Odds #
    ax1 = fig.add_subplot(gs[0, 0])
    Odds_Hist(Results_DF, Aper, Z_Lim, Output_Dir, ax1, plot_Folds, Save=False, Show=False, Close=False)

    # PIT #
    ax2 = fig.add_subplot(gs[0, 1])
    PIT_Hist(Results_DF, Z_Lim, Output_Dir, ax2, plot_Folds, Save=False, Show=False, Close=False)

    # CRPS #
    ax3 = fig.add_subplot(gs[0, 2])
    CRPS_Hist(Results_DF, Z_Lim, Output_Dir, ax3, plot_Folds, DarkMode, Save=False, Show=False, Close=False)

    # HPDCI (only for the final HPDCI) #
    ax4 = fig.add_subplot(gs[1, 0:2])
    HPDCI_Plot(Results_DF, Aper, Output_Dir, ax=ax4, Save=False, Show=False, Close=False)

    # Delta vs Odds (only for the final photo-z) #
    ax5 = fig.add_subplot(gs[1, 2])

    try:
        Results_DF['Delta'] = Results_DF['zphot'] - Results_DF['z']

        odds_bins = np.arange(0, Results_DF['Odds'].max()-0.05, 0.05)
        for delta_z_val in np.arange(0, 1.2, 0.2):
            ax5.plot(odds_bins, calc_ratios(Results_DF, odds_bins, delta_z_val), lw=2,
                        label=f'$\delta z$ $\leqslant$ {delta_z_val:.1f}', zorder=1-delta_z_val)    
        ax5.set_xlim(0, Results_DF['Odds'].max())
        ax5.set_ylabel('Fraction of objects (%)')
        ax5.set_xlabel('Odds greater than')
        legend = ax5.legend(fontsize=10)
        if DarkMode: plt.setp(legend.get_texts(), color='w')
    
    except:
        print("# Error plotting PDF Delta. Check the value len(Results_DF[odds_cond]['Delta']), ", end='')
        print("if it's zero it will return an error.")      

    if Save: plt.savefig(Output_Dir+'ODDS_PIT_CRPS_HPDCI_Delta.png', bbox_inches='tight')
    if Show: plt.show()
    if Close: plt.close()


def Odds_Hexbin(Results_DF, Z_Lim:float, Output_Dir:str, DarkMode=False, Save=True, Show=False, Close=True):
    
    odds = Results_DF['Odds'].quantile([0.25, 0.5, 0.75]).values.round(2)
    n_odds = len(odds)
    x = np.arange(0, Z_Lim+0.1, 0.1)
    fig = plt.figure(figsize=(6*n_odds+3, 6))
    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    gs = GridSpec(ncols=n_odds+1, nrows=1, figure=fig, width_ratios=n_odds*[1]+[0.1])

    for i in range(len(odds)):
        
        ax = fig.add_subplot(gs[i])
        Cond = Results_DF['Odds'] >= odds[i]
        
        if sum(Cond) != 0:
            im = ax.hexbin(Results_DF['z'][Cond], Results_DF['zphot'][Cond], norm=clr.LogNorm(vmax=300),
                            gridsize=100, mincnt=1, cmap='viridis', extent=(0, Z_Lim, 0, Z_Lim))
            ax.set_xlabel('Spectroscopic redshift')
            ax.plot([0, Z_Lim], [0, Z_Lim], ls='--', color='red', lw=1, label='x=y')
            ax.fill_between(x=x, y1=x+0.15*(1+x), y2=Z_Lim, color='red', alpha=0.10, label='Outlier region')
            ax.fill_between(x=x, y1=x-0.15*(1+x), y2=0, color='red', alpha=0.10)
        else:
            print(f'# Error plotting odds >= {odds[i]}.')

        if i == 0:
            ax.set_ylabel('Photometric redshift')
        elif i == n_odds//2:
            legend = ax.legend(loc='center', bbox_to_anchor=(0.5, -0.18), ncol=2)
            if DarkMode: plt.setp(legend.get_texts(), color='w')
        
        ax.set_xlim(0, Z_Lim)
        ax.set_ylim(0, Z_Lim)
        N = len(Results_DF['z'][Cond])
        perc = 100*len(Results_DF['z'][Cond])/len(Results_DF['z'])
        ax.set_title(f'Odds $\geqslant$ {odds[i]}, N={N} ({perc:.0f}%)')    

    ax_cb = fig.add_subplot(gs[n_odds])
    plt.colorbar(im, cax=ax_cb)
    ax_cb.set_ylabel('Number of objects')

    if Save: plt.savefig(Output_Dir+'Hexbin_Odds.png', bbox_inches='tight')
    if Show: plt.show()
    if Close: plt.close()


def Uncertainties(Results_DF, Aper:str, Z_Lim:float, Output_Dir:str, Save=True, Show=False, Close=True):
    
    fig = plt.figure(figsize=(13, 11))
    gs = GridSpec(ncols=3, nrows=2, figure=fig, width_ratios=[1, 1, 0.1])

    ax1 = fig.add_subplot(gs[0, 0])
    im = ax1.hexbin(Results_DF['z'], Results_DF['Odds'], bins='log', mincnt=1, cmap='viridis',
                    extent=(0, Z_Lim, 0, 1), gridsize=75)
    ax1.set_xlim(0, Z_Lim)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel('Spectroscopic redshift')
    ax1.set_ylabel('Odds')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hexbin(Results_DF['r_'+Aper], Results_DF['Odds'], bins='log', mincnt=1, cmap='viridis',
                   extent=(16, 22, 0, 1), gridsize=75)
    ax2.set_xlim(16, 22)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel(f'r_{Aper} magnitude')
    
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hexbin(Results_DF['zphot'], Results_DF['Odds'], bins='log', mincnt=1, cmap='viridis',
                   extent=(0, Z_Lim, 0, 1), gridsize=75)
    ax3.set_xlim(0, Z_Lim)
    ax3.set_ylim(0, 1)
    ax3.set_xlabel('Photometric redshift')
    ax3.set_ylabel('Odds')
    
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hexbin(Results_DF['zphot_err'], Results_DF['Odds'], bins='log', mincnt=1, cmap='viridis',
                   extent=(0, 1.6, 0, 1), gridsize=75)
    ax4.set_xlim(0, 1.6)
    ax4.set_ylim(0, 1)
    ax4.set_xlabel('Photometric redshift error (zphot_err)')
    
    ax_cb = fig.add_subplot(gs[0:2, 2])
    plt.colorbar(im, cax=ax_cb)
    ax_cb.set_ylabel('Number of objects')
    
    if Save: plt.savefig(Output_Dir+'Uncertainties.png', bbox_inches='tight')
    if Show: plt.show()
    if Close: plt.close()


def Limited_Metrics(Results_DF, Aper:str, Output_Dir:str, PDFs:bool, Save=True, Show=False):
    
    Metrics = ['z', 'r_'+Aper, 'NMAD', 'RMSE', 'Bias']
    Col_Lims = {'z': ((0, 3.5), (0, 5), (0, 7)),
                'r_'+Aper: ((16, 20), (18, 21.3), (21.3, 22), (0, 23))}
    
    if PDFs:
        Metrics.extend(['Odds', 'PIT', 'CRPS'])
        Col_Lims['Odds'] = (0, 0.2, 0.4)
    
    metrics = do_col_orders(Metrics, list(Col_Lims.keys()))
    All_Limited_Metrics = {}
    for Col, Lims in Col_Lims.items():
        
        Limited_Metrics = {}
        for metric in metrics[Col]:
            Limited_Metrics[metric] = []
            
            for lim in Lims:
                
                if Col != 'Odds': Condition = ((Results_DF[Col] > lim[0]) & (Results_DF[Col] <= lim[1]))
                else: Condition = (Results_DF[Col] >= lim)
                Lim_ZSpec = Results_DF['z'][Condition]
                Lim_ZPhot = Results_DF['zphot'][Condition]
                
                if metric == Col:
                    calc_metric = lim
                else:
                    if metric == 'z': calc_metric = np.median(Results_DF['z'][Condition])
                    elif metric == 'r_'+Aper: calc_metric = np.median(Results_DF['r_'+Aper][Condition])
                    elif metric == 'NMAD': calc_metric = nmad(Lim_ZSpec, Lim_ZPhot)
                    elif metric == 'RMSE': calc_metric = rmse(Lim_ZSpec, Lim_ZPhot)
                    elif metric == 'Bias': calc_metric = bias(Lim_ZSpec, Lim_ZPhot)
                    elif metric == 'Odds': calc_metric = np.median(Results_DF['Odds'][Condition])
                    elif metric == 'PIT':  calc_metric = np.median(Results_DF['PIT'][Condition])
                    elif metric == 'CRPS': np.median(Results_DF['CRPS'][Condition])
                    
                Limited_Metrics[metric].append(calc_metric)
        
        All_Limited_Metrics[Col] = pd.DataFrame(Limited_Metrics).set_index(Col)
    
    if Save:
        for Col, DataFrame in All_Limited_Metrics.items():
            DataFrame.to_csv(Output_Dir+f'/Results_{Col}Lim.csv')
    if Show:
        for Col, DataFrame in All_Limited_Metrics.items():
            display(All_Limited_Metrics[Col])


def Result_Bins_Fold(Results_DF, Aper:str, Output_Dir:str, bins_dict:dict, Cumulative:bool,
                Save=True, Show=False, Close=True):
    
    Bins_Dict, Grid_Dict = do_bins_grid(bins_dict)
    
    Metrics = ['nmad', 'rmse', 'bias']
    n_metrics = len(Metrics)
    Bins = ['z', 'r_'+Aper]
    n_bins = len(Bins_Dict.keys())
    Folds = [int(s[4:]) for s in listdir(Output_Dir+'SavedModels/')] # [4:] removes 'Fold' from folder name
    
    dat = {0: pd.read_csv('D:/Documentos/QSOs-bNN/_data/external/0.csv'),
           1: pd.read_csv('D:/Documentos/QSOs-bNN/_data/external/1.csv'),
           2: pd.read_csv('D:/Documentos/QSOs-bNN/_data/external/2.csv'),
           3: pd.read_csv('D:/Documentos/QSOs-bNN/_data/external/3.csv'),
           4: pd.read_csv('D:/Documentos/QSOs-bNN/_data/external/4.csv')}

    rat_x = 6
    rat_y = 4
    fig, axes = plt.subplots(n_metrics, n_bins, figsize=(rat_x*n_bins, rat_y*n_metrics))
    fig.subplots_adjust(hspace=0.3, wspace=0.25)
    Colors = ['#0071b2', '#d54b00', '#56b3e9', '#cc79a7', '#e69d00', '#009e74']
    
    Y_Labels = ['$\sigma_{NMAD}$', '$\sigma_{RMSE}$', '$\mu$']
    X_Labels = ['$z_{spec}$', 'r']
    
    Result_List = {}
    Result_List2 = {}
    for j in range(n_bins):
        bin_col = Bins[j]
        
        Bins_Col = Bins_Dict[bin_col]
        Grid_Col = Grid_Dict[bin_col]
        
        for i in range(n_metrics):
            metric = Metrics[i]
            
            Result_List[metric, bin_col] = []
            Result_List2[metric, bin_col] = []
            for fold in Folds:
                Result_List[metric, bin_col].append(Calculate_Metric(Results_DF['z'],
                                                                     Results_DF[f'zphot_{fold}'],
                                                                     Results_DF['r_'+Aper], Aper,
                                                                     Bins_Dict, Grid_Dict,
                                                                     Metric=metric, Bins=bin_col,
                                                                     Cumulative=Cumulative))
                results_df = dat[fold]
                Result_List2[metric, bin_col].append(Calculate_Metric(results_df['Z'],
                                                                      results_df['z_pred'],
                                                                      results_df['r_'+Aper], Aper,
                                                                      Bins_Dict, Grid_Dict,
                                                                      Metric=metric, Bins=bin_col,
                                                                      Cumulative=Cumulative))
            
            #print(metric, bin_col)
            #print(Result_List[metric, bin_col])
            if metric != 'numb':
                axes[i, j].scatter(Bins_Col+(Grid_Col/2), np.nanmedian(Result_List[metric, bin_col], axis=0),
                                   color=Colors[0], s=10, label='Bayesian Mixture Density Network')
                axes[i, j].plot(Bins_Col+(Grid_Col/2), np.nanmedian(Result_List[metric, bin_col], axis=0),
                                color=Colors[0], ls='--', lw=1)
                
                axes[i, j].scatter(Bins_Col+(Grid_Col/2), np.nanmedian(Result_List2[metric, bin_col], axis=0),
                                   color=Colors[1], s=10, label='Random Forest')
                axes[i, j].plot(Bins_Col+(Grid_Col/2), np.nanmedian(Result_List2[metric, bin_col], axis=0),
                                color=Colors[1], ls='--', lw=1)
                
                #axes2 = axes[i, j].twinx()
                #axes2.hist(Results_DF[bin_col], density=True, histtype='step', color='k', lw=1, zorder=0)
                #axes2.grid(False)
                
                axes[i, j].set_xlim(Bins_Col[0], Bins_Col[-1])
                axes[i, j].set_xticks(np.arange(Bins_Col[0], Bins_Col[-1]+1, 1))
                
                if metric != 'bias':
                    for k in [1]: # If you want to also plot the 1 and 2 sigma intervals, just change this to [1, 2, 3]
                        axes[i, j].fill_between(
                            Bins_Col+(Grid_Col/2),
                            y1=np.nanmedian(Result_List[metric, bin_col], axis=0)+k*np.nanstd(Result_List[metric, bin_col], axis=0),
                            y2=np.clip(
                                np.nanmedian(Result_List[metric, bin_col], axis=0)-k*np.nanstd(Result_List[metric, bin_col], axis=0),
                                0, 10), color=Colors[0], alpha=0.35)
                        
                        axes[i, j].fill_between(
                            Bins_Col+(Grid_Col/2),
                            y1=np.nanmedian(Result_List2[metric, bin_col], axis=0)+k*np.nanstd(Result_List2[metric, bin_col], axis=0),
                            y2=np.clip(
                                np.nanmedian(Result_List2[metric, bin_col], axis=0)-k*np.nanstd(Result_List2[metric, bin_col], axis=0),
                                0, 10), color=Colors[1], alpha=0.35)
                else:
                    axes[i, j].axhline(0, color='gray', lw=2, zorder=-1)
                    for k in [1]:
                        axes[i, j].fill_between(
                            Bins_Col+(Grid_Col/2),
                            y1=np.nanmedian(Result_List[metric, bin_col], axis=0)+k*np.nanstd(Result_List[metric, bin_col], axis=0),
                            y2=np.nanmedian(Result_List[metric, bin_col], axis=0)-k*np.nanstd(Result_List[metric, bin_col], axis=0),
                            color=Colors[0], alpha=0.35)
                        
                        axes[i, j].fill_between(
                            Bins_Col+(Grid_Col/2),
                            y1=np.nanmedian(Result_List2[metric, bin_col], axis=0)+k*np.nanstd(Result_List2[metric, bin_col], axis=0),
                            y2=np.nanmedian(Result_List2[metric, bin_col], axis=0)-k*np.nanstd(Result_List2[metric, bin_col], axis=0),
                            color=Colors[1], alpha=0.35)
            
            else:
                axes[i, j].bar(Bins_Col+(Grid_Col/2), np.nanmedian(Result_List[metric, bin_col], axis=0), Grid_Col/2)

            axes[i, j].set_ylabel(Y_Labels[i])
            axes[i, j].set_xlabel(X_Labels[j])
            #axes[i, j].ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)

    plt.legend(loc='center', ncol=2, bbox_to_anchor=[0.5, 0.9], bbox_transform=fig.transFigure)
    fig.align_ylabels()
    #plt.figtext(0.5, 0.91, f'Performance as a function of r_{Aper} and spec-z', ha='center', va='center', size=16)
    
    # Save results as csvs (easier to use later)
    dfs = []
    dfs2 = []
    for j in range(n_bins):
        bin_col = Bins[j]
        Bins_Col = Bins_Dict[bin_col]
        Grid_Col = Grid_Dict[bin_col]
    
        Results_Plot = pd.DataFrame(Bins_Col+(Grid_Col/2), columns=[f'{Bins_Col}']).join(
            pd.DataFrame(np.nanmedian(Result_List['nmad', bin_col], axis=0), columns=['NMAD_Median'])).join(
                pd.DataFrame(np.nanstd(Result_List['nmad', bin_col], axis=0), columns=['NMAD_STD'])).join(
                    pd.DataFrame(np.nanmedian(Result_List['rmse', bin_col], axis=0), columns=['RMSE_Median'])).join(
                        pd.DataFrame(np.nanstd(Result_List['rmse', bin_col], axis=0), columns=['RMSE_STD'])).join(
                            pd.DataFrame(np.nanmedian(Result_List['bias', bin_col], axis=0), columns=['BIAS_MEDIAN'])).join(
                                pd.DataFrame(np.nanstd(Result_List['bias', bin_col], axis=0), columns=['BIAS_STD']))
        Results_Plot2 = pd.DataFrame(Bins_Col+(Grid_Col/2), columns=[f'{Bins_Col}']).join(
            pd.DataFrame(np.nanmedian(Result_List2['nmad', bin_col], axis=0), columns=['NMAD_Median'])).join(
                pd.DataFrame(np.nanstd(Result_List2['nmad', bin_col], axis=0), columns=['NMAD_STD'])).join(
                    pd.DataFrame(np.nanmedian(Result_List2['rmse', bin_col], axis=0), columns=['RMSE_Median'])).join(
                        pd.DataFrame(np.nanstd(Result_List2['rmse', bin_col], axis=0), columns=['RMSE_STD'])).join(
                            pd.DataFrame(np.nanmedian(Result_List2['bias', bin_col], axis=0), columns=['BIAS_MEDIAN'])).join(
                                pd.DataFrame(np.nanstd(Result_List2['bias', bin_col], axis=0), columns=['BIAS_STD']))
        
        dfs.append(Results_Plot)
        dfs2.append(Results_Plot2)

    if Save:
        if Cumulative: s = '_Cumul'
        else: s = ''
        for j in range(n_bins):
            bin_col = Bins[j]
            dfs[j].to_csv(Output_Dir+f'Results_{bin_col}{s}.csv', index=False)
        plt.savefig(Output_Dir+f'Metrics_Bin_Fold{s}.png', bbox_inches='tight')
    if Show:
        plt.show()
        for j in range(n_bins):
            display(dfs[j])
            display(dfs2[j])
    if Close: plt.close()


def Result_Bins(models:dict, Aper:str, Output_Dir:str, bins_dict:dict, Cumulative:bool, Save=True, Show=False, Close=True):
    
    Bins_Dict, Grid_Dict = do_bins_grid(bins_dict)
    n_bins = len(Bins_Dict.keys())
    Metrics = {
        'nmad': '$\sigma_{NMAD}$',
        'rmse': '$\sigma_{RMSE}$',
        'bias': '$\mu$'
    }
    n_metrics = len(Metrics)

    rat_x = 6
    rat_y = 4
    fig, ax = plt.subplots(n_metrics, n_bins, figsize=(rat_x*n_bins, rat_y*n_metrics))
    fig.subplots_adjust(hspace=0.3, wspace=0.25)
    Colors = ['#db1d1dff', '#4821d6ff', 'orange', 'green']
    
    X_Labels = ['$z_{spec}$', 'r']
    
    for j in range(n_bins):
        bin_col = list(Bins_Dict.keys())[j]
        
        Bins_Col = Bins_Dict[bin_col]
        Grid_Col = Grid_Dict[bin_col]
        
        for i, metric in enumerate(Metrics):

            tup = (i, j) if n_metrics > 1 else j
            
            for k, name in enumerate(models):
                
                calc = Calculate_Metric(models[name]['z'], models[name][f'zphot'], models[name]['r_'+Aper], Aper,
                                        Bins_Dict, Grid_Dict, Metric=metric, Bins=bin_col, Cumulative=Cumulative)
                
                if n_metrics > 1: s = ''
                else:
                    cond = models[name]['z'] <= 5
                    total_calc = round(eval(metric)(models[name]['z'][cond], models[name][f'zphot'][cond]), 3)
                    s = f' ({total_calc})'
                ax[tup].scatter(Bins_Col+(Grid_Col/2), calc, color=Colors[k], s=10, label=name)#+s)
                ax[tup].plot(Bins_Col+(Grid_Col/2), calc, color=Colors[k], ls='--', lw=1)
            
            ax[tup].set_xlim(Bins_Col[0], Bins_Col[-1])
            ax[tup].set_xticks(np.arange(Bins_Col[0], Bins_Col[-1]+1, 1))
            
            if metric == 'bias':
                ax[tup].axhline(0, color='gray', lw=2, zorder=-1)

            ax[tup].set_ylabel(Metrics[metric])
            ax[tup].set_xlabel(X_Labels[j])
            #axes[i, j].ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)

    if len(models) > 1:
        plt.legend(loc='center', ncol=len(models), bbox_to_anchor=[0.5, 0.9], bbox_transform=fig.transFigure)
    fig.align_ylabels()

    if Save:
        if Cumulative: s = '_Cumul'
        else: s = ''
        plt.savefig(Output_Dir+f'Metrics_Bin{s}_{"-".join(Metrics.keys())}.png', bbox_inches='tight', dpi=300)
    if Show:
        plt.show()
    if Close: plt.close()


def Zphot_Zspec(Result_DF, Aper:str, Z_Lim:float, Output_Dir:str, Save=True, Show=False, Close=True):
    
    fig, [ax, ax2] = plt.subplots(nrows=2, ncols=1, figsize=(5.2, 7.8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    cmap = plt.cm.jet
    bounds = np.arange(16, 22.5, 0.5)
    norm = clr.BoundaryNorm(bounds, cmap.N)
    Cond = (Result_DF['r_'+Aper] >= 0) & (Result_DF['r_'+Aper] <= 22)

    points = ax.hexbin(Result_DF['z'][Cond], Result_DF['zphot'][Cond], C=Result_DF['r_'+Aper][Cond],
                       norm=norm, cmap=cmap, alpha=0.8, extent=(0, Z_Lim, 0, Z_Lim), gridsize=100, mincnt=0)
    ax.plot([0, 1], [0, 1], color='k', ls='--', transform=ax.transAxes, linewidth=2)
    ax.label_outer()

    ax2.hexbin(Result_DF['z'][Cond], Result_DF['zphot'][Cond]-Result_DF['z'][Cond], C=Result_DF['r_'+Aper][Cond],
               norm=norm, cmap=cmap, alpha=0.8, extent=(0, Z_Lim, -Z_Lim, Z_Lim), gridsize=(100, 30), mincnt=0)
    ax2.axhline(0, color='black', linestyle='dashed')
    ax.set_ylim(0, Z_Lim)
    ax.set_xlim(0, Z_Lim)
    ax2.set_ylim(-Z_Lim, Z_Lim)
    ax2.set_xlim(0, Z_Lim)
    ax2.set_yticks(np.arange(-4, 6, 2))
    ax2.set_xlabel('$z_{spec}$')
    ax.set_ylabel('$z_{phot}$')
    ax2.set_ylabel('$z_{phot} - z_{spec}$')
    
    plt.subplots_adjust(hspace=0.02)
    cax = fig.add_axes([1.0, 0.12, 0.03, 0.76])
    cax.grid(False)
    fig.colorbar(points, cax=cax, label='r')
    
    if Save: plt.savefig(Output_Dir+'Zphot_Zspec.png', bbox_inches='tight', dpi=500)
    if Show: plt.show()
    if Close: plt.close()


def Benchmarks(Results_DF, PDF_List, x, Aper:str, Z_Lim:float, bins_dict:dict, Seed:int, Output_Dir:str, Bench_Dict:dict,
               DarkMode=False, Save=True, Show=False, Close=True):
    
    print('# Benchmarking #')
    
    if Save:
        print(f'# Saving plots to {Output_Dir}')
    else:
        print('# Not saving plots')
    
    plt.rcParams.update({'axes.titlesize': 16, 'axes.labelsize': 14,
                         'axes.grid': True, 'axes.formatter.use_mathtext': True,
                         'xtick.labelsize': 14, 'xtick.minor.visible': True,
                         'xtick.minor.size': 2, 'xtick.minor.width': 0.6,
                         'xtick.major.size': 3.5, 'xtick.major.width': 0.8,
                         'ytick.labelsize': 14, 'ytick.minor.visible': True,
                         'ytick.minor.size': 2, 'ytick.minor.width': 0.6,
                         'ytick.major.size': 3.5, 'ytick.major.width': 0.8,
                         'legend.fontsize': 14})
    if DarkMode:
        plt.rcParams.update({'lines.color': 'white', 'patch.edgecolor': 'white', 'text.color': 'white',
                             'axes.facecolor': '#1a1a1a', 'axes.edgecolor': 'lightgray', 'axes.labelcolor': 'white',
                             'grid.linewidth': 0.5, 'grid.alpha': 0.5, 'grid.color': 'lightgray',
                             'xtick.color': 'white', 'ytick.color': 'white',
                             'figure.facecolor': 'black', 'figure.edgecolor': 'black',
                             'savefig.facecolor': '(0.0, 0.0, 1.0, 0.0)', 'savefig.edgecolor': 'black'})
    else:
        plt.rcParams.update({'lines.color': 'C0', 'patch.edgecolor': 'black', 'text.color': 'black',
                             'axes.facecolor': 'white', 'axes.edgecolor': 'black', 'axes.labelcolor': 'black',
                             'grid.linewidth': 0.8, 'grid.alpha': 1.0, 'grid.color': '#b0b0b0',
                             'xtick.color': 'k', 'ytick.color': 'k',
                             'figure.facecolor': 'white', 'figure.edgecolor': 'white',
                             'savefig.facecolor': 'white', 'savefig.edgecolor': 'white'})
    
    if Bench_Dict['Do_Histograms']:
        try: Histograms(Results_DF, PDF_List, x, Z_Lim, Output_Dir, DarkMode, Save, Show, Close)
        except: print('# Error plotting redshift histograms')
    
    if Bench_Dict['Do_PDF_Bins_RA']:
        try: PDF_Bins_RA(Results_DF, PDF_List, x, Z_Lim, Output_Dir, Save, Show, Close)
        except: print('# Error plotting PDFs in bins of RA')
    
    if Bench_Dict['Do_LSS_2D']:
        try: LSS_2D(Results_DF, Z_Lim, Output_Dir, Save, Show, Close)
        except: print('# Error plotting LSS 2D')
    
    if Bench_Dict['Do_Average_PDFs']:
        try: Average_PDFs(Results_DF, PDF_List, x, Aper, Z_Lim, Output_Dir, DarkMode, Save, Show, Close)
        except: print('# Error plotting average PDFs')
    
    if Bench_Dict['Do_PDF_Sample']:
        try: PDF_Sample(Results_DF, {'': PDF_List}, x, Aper, Seed, Output_Dir, DarkMode, Save, Show, Close)
        except: print('# Error plotting PDF samples')
    
    if Bench_Dict['Do_Odds_PIT_CRPS_HPDCI_Delta']:
        try: Odds_PIT_CRPS_HPDCI_Delta(Results_DF, Aper, Z_Lim, Output_Dir, DarkMode, Save, Show, Close)
        except: print('# Error plotting Odds, PIT, CRPS, HPDCI, and Delta')
    
    if Bench_Dict['Do_Odds_Hexbin']:
        try: Odds_Hexbin(Results_DF, Z_Lim, Output_Dir, DarkMode, Save, Show, Close)
        except: print('# Error plotting hexbin with odds')
    
    if Bench_Dict['Do_Uncertainties']:
        try: Uncertainties(Results_DF, Aper, Z_Lim, Output_Dir, Save, Show, Close)
        except: print('# Error plotting uncertainties')
    
    if Bench_Dict['Do_Limited_Metrics']:
        try: Limited_Metrics(Results_DF, Aper, Output_Dir, True, Save, Show)
        except: print('# Error plotting limited metrics')
    
    if Bench_Dict['Do_Results_Bins']:
        try: Result_Bins({'': Results_DF}, Aper, Output_Dir, bins_dict, False, Save, Show, Close)
        except: print('# Error plotting binned metrics')
    
    if Bench_Dict['Do_Results_Bins_Cumul']:
        try: Result_Bins({'': Results_DF}, Aper, Output_Dir, bins_dict, True, Save, Show, Close)
        except: print('# Error plotting binned metrics cumulatively')
    
    if Bench_Dict['Do_Zphot_Zspec']:
        try: Zphot_Zspec(Results_DF, Aper, Z_Lim, Output_Dir, Save, Show, Close)
        except: print('# Error plotting zphot x zspec')
