#!/usr/bin/env python
import os

import numpy as np
import pandas as pd
from scipy.stats import binom
import matplotlib.pyplot as plt
import matplotlib.colors as MplColors

from settings.paths import validation_path, img_path
from utils.preprocessing import create_bins
from utils.crossvalidation import metric_per_bin, count_bins


plt.rcParams["font.size"] = 22
plt.rcParams["ytick.minor.visible"] = True
plt.rcParams["xtick.minor.visible"] = True

CB_color_cycle = ["#377eb8", "#ff7f00", "#4daf4a", "#f781bf", "#a65628", "#984ea3", "#999999", "#e41a1c", "#dede00"]


def cols2labels(colnames):
    """cols2labels modifies column names for plots

    Parameters
    ----------
    colnames : dict

    Returns
    -------
    dict
        Updated names
    """    

    for key, value in colnames.items():
        try:
            if colnames[key][1:] == "mag": #rmag, gmag,...
                colnames[key] = value[0]

            if "_" in colnames[key]: # W1_MAG, J0395_auto...
                colnames[key] = value.split('_')[0]
        except:
            pass
    return colnames


def plot_sample(train, test, var:str, save=False):
    
    if var == 'r_PStotal':
        bins = np.arange(16, 22.25, 0.25)
        xlabel = 'r'
    elif var == 'Z':
        bins = np.arange(0, 5.25, 0.25)
        xlabel = '$z_{spec}$'
    
    plt.figure(figsize=(10, 6))
    plt.hist(train[var], label='Train', bins=bins, log=True, color='#dede00', histtype='step', lw=3)
    plt.hist(test[var], label='Test', bins=bins, log=True, color='#984ea3')
    plt.ylabel('Counts')
    plt.xlabel(xlabel)
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(img_path, f'train_test_{var.split("_")[0]}.png'),
                    bbox_inches='tight', facecolor='white', dpi=300)
        plt.savefig(os.path.join(img_path, f'train_test_{var.split("_")[0]}.eps'),
                    bbox_inches='tight', facecolor='white', format='eps')
    plt.show()
    plt.close()


def plot_scatter_z(model, per="r", save=False):
    plt.rcParams["font.size"] = 22
    plt.rcParams["ytick.minor.visible"] = True
    plt.rcParams["xtick.minor.visible"] = True

    df = pd.read_table(os.path.join(validation_path, "original", "z_"+str(model)), sep=",", index_col="index")
    
    fig, [ax,ax2] = plt.subplots(2, 1, figsize=(8,10), sharex=True, gridspec_kw={'height_ratios':[2,1]})
    df_99 = df.query("r_PStotal==99")
    df = df.drop(df_99.index)
    
    lim_mag = 20
    df = df.query("r_PStotal <"+str(lim_mag))
    
    # df = df.query("r_iso > 20 and r_iso<22")
    cmap = plt.cm.jet
    # bounds = np.arange(15,20.5,0.5)
    bounds = np.arange(15,22.5,0.5)
    norm = MplColors.BoundaryNorm(bounds,cmap.N)

    ax.grid()
    points = ax.hexbin(df.Z, df.z_pred, C=df.r_PStotal, gridsize=100, mincnt=0,
                       extent=(0,7,0,7), norm=norm, cmap=cmap, alpha=0.8)
    ax.plot([0,1], [0,1], "k--", transform=ax.transAxes, linewidth=2)
    ax.scatter(df_99.Z, df_99.z_pred, c="black", marker="o", edgecolor="white", s=60)

    ax2.grid()
    ax2.hexbin(df.Z, df.z_pred - df.Z, C=df.r_PStotal, gridsize=(100,30), mincnt=0,
               extent=(0,7,-6.8,3.4), norm=norm, cmap=cmap, alpha=0.8)
    ax2.scatter(df_99.Z, df_99.z_pred - df_99.Z, c="black", marker="o", edgecolor="white", s=60)
    
    ax2.axhline(0, color="black", linestyle="dashed")

    ax.set_ylim(0, 7)
    ax.set_xlim(0, 7)
    ax2.set_ylim(-6.8, 3.4)
    ax2.set_xlim(0,7)
    ax2.set_xlabel("Spectroscopic Redshift")
    ax.set_ylabel(r"$\bar{z}$")
    ax2.set_ylabel(r"$\bar{z}- z$")

    ax.label_outer()
    
    plt.subplots_adjust(hspace=0.01)
    cax = fig.add_axes([1., 0.1, 0.03, 0.86])
    cbar = fig.colorbar(points, cax=cax, label="r magnitude")

    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(img_path, "scatter_z_per_"+per+"_lim"+str(lim_mag)+"_"+str(model)+".png"))
    # ax.set_ylabel(dict_df[name][column+"_median"].name)


def plot_metric_per_bin(list_models, data, metric, bins, color_feat, per="r_PStotal", cutoff=5, std=False, save=False,
                        idx=None, legend=True):
    string = str(list_models[0])
    result = {}
    for model in list_models:
        if model != list_models[0]:
            string = string+"x"+str(model)
        result[model] = metrics_bin(data=data[model], metric=metric, bins=bins, var=per)
    
    fig, ax = plt.subplots(1,1, figsize=(10,7))

    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    for name in data:            
        ax.scatter(result[name].bins, result[name][metric.__name__+"_median"], color=color_feat[name])
        ax.plot(result[name].bins, result[name][metric.__name__+"_median"], "--", label=name, color=color_feat[name])
        
        if std == True:
            ax.fill_between(
                result[name].bins,
                result[name][metric.__name__+"_median"]-result[name][metric.__name__+"_median"[0:-6]+"std"],
                result[name][metric.__name__+"_median"]+result[name][metric.__name__+"_median"[0:-6]+"std"],
                alpha=0.5, color=color_feat[name])

    if per == "r_PStotal":
        ax.set_xlabel("r")
    elif per == "Z":
        ax.set_xlabel("Spectroscopic Redshift")
    else:
        ax.set_xlabel("g-r")           
    
    if metric.__name__ == "rmse":
        ax.set_ylabel(r"$\sigma_{RMSE}$")
    elif metric.__name__ == "out_frac":
        ax.set_ylabel(r"$\eta_{0.30}$")
        plt.gca().set_ylim(bottom=0)
    elif metric.__name__ == "bias":
        ax.set_ylabel("Bias")
    elif metric.__name__ == "mse":
        ax.set_ylabel("MSE")
    else:
        ax.set_ylabel(r"$\sigma_{NMAD}$")   
        
    ax.grid()

    if legend:
        if len(list_models) > 1:
            ax.legend(loc="upper right", prop={"size":15})
        leg = plt.legend()
        leg_lines = leg.get_lines()
        plt.setp(leg_lines, linewidth=3)
    
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(img_path, string+"_"+metric.__name__+"_"+per+".png"),
                     facecolor="white", transparent=False)
        plt.savefig(os.path.join(img_path, string+"_"+metric.__name__+"_"+per+".eps"),
                     facecolor="white", transparent=False, format="eps")
    return fig


def metrics_bin(data, metric, bins="None", var="g-r"):
    bins_r, itv_r = create_bins(data, return_data=False, var=var,  bins=bins)
    bin_size = bins_r[1] - bins_r[0]
    half_bins_r = np.arange(bins_r[0]+(bin_size/2), bins_r[-1]+bin_size/2, bin_size)
    
    list_folds = data.fold.unique()

    original = pd.DataFrame()
    for i in list_folds:
        aux = data.query("fold == "+str(i))
        output_bin = metric_per_bin(metric=metric, z=aux, itvs=itv_r)
        output_bin = pd.DataFrame(output_bin, columns=[metric.__name__+"_fold"+str(i)])
        original = pd.concat([original,output_bin], axis=1)
        original.insert(0, "n_fold"+str(i), count_bins(aux, itv_r))
        

    result = pd.DataFrame()
    aux = np.repeat(metric.__name__, 5)
    aux = [t+'_fold'+str(i) for i,t in enumerate(aux)] 
    result.insert(0,metric.__name__+"_std", original[aux].T.std())
    result.insert(0, metric.__name__+"_median", original[aux].T.median())
    original.insert(0, "bins", half_bins_r)

    aux = np.repeat("n_fold", 5)
    aux = [t+str(i) for i,t in enumerate(aux)]
    result.insert(0, "n", original[aux].T.sum())
    result.insert(0, "bins", half_bins_r)
    return result


def plot_PDFs(alg:str, models_dict:dict, sdss, z, x, idxs, colors_dict:dict, title, save=False):
    
    r_conds = [f'r_PStotal < 20', f'20 < r_PStotal < 21.3', f'r_PStotal > 21.3']
    z_conds = ['z < 0.5', '0.5 < z < 3.5', '3.5 < z < 5']
    n_r = idxs.shape[0]
    n_z = idxs.shape[1]

    if n_r > 1 and n_z > 1: figsize = (3.8*n_z, 2.4*n_r)
    else: figsize = (6, 4)
    fig, axes = plt.subplots(nrows=n_r, ncols=n_z, figsize=figsize)
    for i in range(n_r):
        for j in range(n_z):
            
            if n_r > 1 and n_z > 1:
                ax = axes[i][j]
                idx = idxs[i][j]
            else:
                ax = axes
                idx = idxs[0][0]
            
            sdss_name = sdss.loc[idx, 'SDSS_NAME']
            if n_r > 1 and n_z > 1:
                ax.plot([], label=f'SDSS J{sdss_name}')
                leg = ax.legend(loc='best', fontsize=10, handlelength=0, handletextpad=0, borderpad=0.1,
                                framealpha=0.7)
                leg.get_frame().set_linewidth(0.5)
            
            for model_name, df in models_dict.items():
                
                if alg == 'BMDN':
                    weights = df.loc[idx, [f'PDF_Weight_{i}' for i in range(6)]].values.astype(float)
                    means = df.loc[idx, [f'PDF_Mean_{i}' for i in range(6)]].values.astype(float)
                    stds = df.loc[idx, [f'PDF_STD_{i}' for i in range(6)]].values.astype(float)
                    pdf = np.sum(
                        weights * (1/(stds*np.sqrt(2*np.pi))) * np.exp((-1/2)*((x[:, None]-means)**2)/(stds)**2),
                        axis=1)
                    pdf = pdf / np.trapz(pdf, x)
                    ax.axvline(df.loc[idx, 'zphot'], ls='--', c=colors_dict[model_name])
                
                elif alg == 'FlexCoDE':
                    pdf = df.loc[idx, [f'z_flex_pdf_{i}' for i in range(1, 201)]].values
                    ax.axvline(df.loc[idx, 'z_flex_peak'], ls='--', c=colors_dict[model_name])
                
                else:
                    print('"alg" param must be "BMDN" or "FlexCoDE"')
                    break
                
                ax.plot(x, pdf, c=colors_dict[model_name], label=model_name)
            
            # ax.axvline(z.loc[idx, 'z'], ls=':', c='k', label='$z_{spec}$')
            ax.scatter(z.loc[idx, 'z'], -0, c='#333333', s=120, marker="^", label='$z_{spec}$', zorder=12233)

            ax.set_xlim(0, 5)
            ax.set_ylim(0, 1.3* np.max(pdf))
            
            ax.set_xticks(range(6))
            ax.tick_params(axis='both', which='major', labelsize=14)
            # ax.grid()
            if i == 0:
                if n_r > 1 and n_z > 1:
                    ax.set_title(z_conds[j].replace('z', '$z_{spec}$'), size=16)
                else:
                    ax.set_title(f'J{sdss_name}', size=16)
            if n_r-i == 1:
                ax.set_xlabel('$z_{phot}$', size=16)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel('$p(z_{phot})$', size=16)
            elif n_z-j == 1:
                ax2 = ax.twinx()
                ax2.set_yticks([])
                ax2.set_ylabel(r_conds[i].replace(f'r_PStotal', '$r$'), rotation=270, labelpad=16, size=16)
    if n_z > 1:        
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles=handles[1:], labels=labels[1:], loc='lower center', bbox_to_anchor=(0.5, -0.05),
                ncol=1+len(models_dict), fontsize=14)
    else: ax.legend(fontsize=12)
    fig.align_ylabels()
    plt.suptitle(title, fontsize=18)
    fig.tight_layout(pad=0.5)
    
    if save:
        name = f'PDFs_{alg}' if n_r > 1 and n_z > 1 else f'J{sdss_name}_{alg}'
        plt.savefig(os.path.join(img_path, f'{name}.png'),  bbox_inches='tight', facecolor='white', dpi=300)
        plt.savefig(os.path.join(img_path, f'{name}.eps'),  bbox_inches='tight', facecolor='white', format='eps')
    plt.show()
    plt.close()


def plot_with_uniform_band(alg:str, models_dict:dict, colors_dict:dict, ci_level=0.95, n_bins=50, save=False):    
    '''
    Plots the PIT/HPD histogram and calculates the confidence interval for the bin values,
    were the PIT/HPD values follow an uniform distribution
    
    (Aesthetically) Modified from: https://github.com/lee-group-cmu/cdetools/blob/master/python/src/cdetools/plot_utils.py

    @param alg: WRITE
    @param models_dict: WRITE
    @param colors_dict: WRITE
    @param ci_level: a float between 0 and 1 indicating the size of the confidence level
    @param n_bins: an integer, the number of bins in the histogram

    @returns The matplotlib figure object with the histogram of the PIT/HPD values and the CI for the uniform distribution
    '''

    # Extract the number of CDEs
    n = len(list(models_dict.values())[0])

    # Creating upper and lower limit for selected uniform band
    ci_quantity = (1-ci_level) / 2
    low_lim = binom.ppf(q=ci_quantity, n=n, p=1/n_bins)
    upp_lim = binom.ppf(q=ci_level + ci_quantity, n=n, p=1/n_bins)

    # Creating figure
    plt.figure(figsize=(12,7))
    
    for model_name, values in models_dict.items():
        plt.hist(values, bins=n_bins, label=model_name, color=colors_dict[model_name], histtype='step', lw=4)
    
    plt.axhline(y=low_lim, color='grey')
    plt.axhline(y=upp_lim, color='grey')
    plt.axhline(y=n/n_bins, label='Uniform Average (95% CI)', color='#555555', linewidth=4)
    plt.fill_between(x=np.linspace(0, 1, 100), y1=np.repeat(low_lim, 100), y2=np.repeat(upp_lim, 100),
                     color='grey', alpha=0.2)
    plt.xticks()
    plt.yticks()
    plt.xlim(-0.01, 1.01)
    plt.xlabel('PIT values')
    plt.ylabel('Counts')
    plt.title(alg)
    plt.legend(loc='upper center', fontsize=18)
    if save:
        plt.savefig(os.path.join(img_path, f'PIT_{alg}.png'),  bbox_inches='tight', facecolor='white', dpi=300)
        plt.savefig(os.path.join(img_path, f'PIT_{alg}.eps'),  bbox_inches='tight', facecolor='white', format='eps')
    plt.show()
    plt.close()
