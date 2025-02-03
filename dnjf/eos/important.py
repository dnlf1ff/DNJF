import os
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from util import *
from log import *
from plot import eos_plot
import sys
import numpy as np

def birch_murnaghan_energy(V, E0, V0, B0, B0_prime):
    eta = (V0 / V)**(2/3)  # Strain factor
    term1 = (eta - 1)
    term2 = 6 - 4 * eta
    return E0 + (9 * V0 * B0 / 16) * (term1**3 * B0_prime + term1**2 * term2)
  
def fit_bm(system, vol, pe, conf, verbose=False, maxfev=50000,ftol=1e-8, xtol=1e-8, gtol=1e-8,logger=logger):
    initial_guess =  conf[system]['initial_guess']
    params, covariance, infodict, mesg, ier = curve_fit(f=birch_murnaghan_energy, xdata=vol, ydata=pe,  p0=initial_guess, full_output=True, maxfev=maxfev,ftol=ftol, xtol=xtol, gtol=gtol)
    if verbose:
        logger.debug(f"Fit Information:")
        logger.debug(f"  Number of function evaluations: {infodict['nfev']}")
        logger.debug(f"  Infodict: {infodict}\n")
        logger.debug(f"  Fit message: {mesg}\n")
        logger.debug(f"  Fit success: {ier}\n")
    E0_fit, V0_fit, B0_fit, B0_prime_fit = params
    B0_fit = B0_fit * 160.2176620
    
    logger.info(f"Fitted Parameters:")
    logger.info(f"  E0 = {E0_fit:.4f} eV/atom")
    logger.info(f"  V0 = {V0_fit:.4f} A³/atom")
    logger.info(f"  B0 = {B0_fit:.6f} eV/A³, {B0_fit} GPa")
    logger.info(f"  B0' = {B0_prime_fit:.4f}\n")
    vol_fit = np.linspace(min(vol), max(vol), 360) #200
    pe_fit = birch_murnaghan_energy(vol_fit, *params)
    
    return vol_fit, pe_fit, B0_fit #mp.arrays

def comrade(system, df, mlps, return_df=True, logger=logger):
    # df = df.iloc[::-1]
    df.reset_index(inplace=True)
    df.drop('index',axis=1, inplace=True)

    bravais_s = df['bravais'].to_list()
    tag=f'{bravais_s[0]}{bravais_s[1]}'
    for mlp in mlps:
        df[f'pe-dft_{mlp}-rel'] = None
        df[f'pe-{mlp}-rel'] = None
        df[f'del_E-{mlp}'] = None
        df['del_E-dft'] = None
        dft_min=min(df['pe-dft'][0].min(), df['pe-dft'][1].min())
        mlp_min=min(df[f'pe-{mlp}'][0].min(), df[f'pe-{mlp}'][1].min())
        for i, row in df.iterrows():
            df.at[i, f'pe-dft_{mlp}-rel'] = np.asarray(row['pe-dft'] - dft_min)
            df.at[i, f'pe-{mlp}-rel'] = np.asarray(row[f'pe-{mlp}'] - mlp_min)
            df.at[i, f'del_E-dft'] = df['pe-dft'][0].min() - df['pe-dft'][1].min()
            df.at[i, f'del_E-{mlp}'] = df[f'pe-{mlp}'][0].min() - df[f'pe-{mlp}'][1].min()
            save_dict(df, os.path.join(os.environ['JAR'], f'{system}_{tag}_fit.pkl'))
    if return_df:
        return df
    return

def fit_system(system, mlps, conf=None, df= None, return_df = False, save_df=True,logger=logger):
    bravais_s = df['bravais'].to_list() 
    tag=f'{bravais_s[0]}{bravais_s[1]}'
    if conf is None:
        conf = load_conf(os.path.join(os.environ['PRESET'], 'eos','inp.yaml'))
    if df is None:
        df = load_dict(os.path.join(os.environ['JAR'],f'{system}_mlp.pkl'))
    for mlp in mlps:
        df[f'vol-{mlp}-fit'] = None
        df[f'pe-{mlp}-fit'] =  None
        df[f'vol-dft_{mlp}-fit'] =  None
        df[f'pe-dft_{mlp}-fit'] =  None
        df[f'b0-{mlp}'] = None
        df[f'b0-dft']=None
        for i, row in df.iterrows():
            try:
                fit_vol_mlp, fit_pe_mlp, b0_mlp = fit_bm(system, vol=row[f'vol-{mlp}'], pe=row[f'pe-{mlp}-rel'], conf=conf, verbose=True)
                fit_vol_dft, fit_pe_dft, b0_dft = fit_bm(system, vol=row[f'vol-dft'], pe=row[f'pe-dft_{mlp}-rel'], conf=conf, verbose=True)
            except:
                fit_vol_mlp, fit_pe_mlp, b0_mlp = fit_bm(system, vol=row[f'vol-{mlp}'], pe=row[f'pe-{mlp}-rel'], conf=conf, verbose=True,ftol=1e-4,xtol=1e-4, gtol=1e-4)
                fit_vol_dft, fit_pe_dft, b0_dft = fit_bm(system, vol=row[f'vol-dft'], pe=row[f'pe-dft_{mlp}-rel'], conf=conf, verbose=True, ftol=1e-4, xtol=1e-4,gtol=1e-4)
                
            pe_min = min(row[f'pe-{mlp}'])
            
            df.at[i,f'vol-{mlp}-fit'] = fit_vol_mlp #list of np.array
            df.at[i, f'pe-{mlp}-fit'] = fit_pe_mlp # ''
            df.at[i,f'vol-dft_{mlp}-fit'] = fit_vol_dft #list of np.array
            df.at[i, f'pe-dft_{mlp}-fit'] = fit_pe_dft # ''
            df.at[i,f'b0-{mlp}'] = b0_mlp
            df.at[i, 'b0-dft'] = b0_dft

        eos_plot(system, df, mlp) #TODO: may not be functional coding but qnd ...
        if save_df:
            save_dict(df, os.path.join(os.environ['JAR'],f'{system}_{tag}_fit.pkl'))
    if return_df:
        return df
    return

def fit_main(system, mlps, df=None):
    set_env('eos')
    logger = get_logger(system, f'{system}.fit.log',job='fit')
    if df is None:
        df = load_dict(os.path.join(os.environ['JAR'],f'{system}_mlp.pkl'))
        df_s = [df.drop(i) for i in range(3)]
        for df_ in df_s:
            df_ = comrade(system, df_, mlps)
            fit_system(system, mlps, df=df_)

if __name__ == '__main__':
    mlps = ['chgTot','chgTot_l3i3','chgTot_l4i3','chgTot_l3i5','omat_epoch1','omat_epoch2','omat_epoch3','omat_epoch4','omat_ft_r5','m3g_r55','m3g_r6','m3g_n','r5pp','omat_i3pp','omat_i3pp','omat_i5_epoch1','omat_i5_epoch2','omat_i5_epoch3','omat_i5_epoch4','omat_i5pp_epoch1','omat_i5pp_epoch2','omat_i5pp_epoch3','omat_i5pp_epoch4','mace','matsim']
    fit_main(system=sys.argv[1], mlps=mlps)

import matplotlib.pyplot as plt
from matplotlib import rcParams
import pickle
import numpy as np
import pandas as pd
import os

from util import make_dir 

def eos_plot(system, df, mlp):
    fig = plt.figure(figsize=(6.5,6.5))
    plt.title(f'{system}-{mlp}', fontsize=17, fontweight='bold', pad=10, loc='right')
    rcParams['font.family'] = 'Arial' 
    for i, row in enumerate(df.iterrows()):
        row = row[1]
        if row['bravais'] == 'fcc':
            plt.plot(row[f'vol-dft_{mlp}-fit'],  row[f'pe-dft_{mlp}-fit'], color = '#080C85', linestyle='-', linewidth=3,zorder=3)
            plt.scatter(row[f'vol-dft'], row[f'pe-dft_{mlp}-rel'], label=f'DFT {row["bravais"]} {row["mp_id"]}', color='#080C85', marker='X', edgecolors='k', s=290, alpha=1,zorder=3)
            
            plt.plot(row[f'vol-{mlp}-fit'], row[f'pe-{mlp}-fit'], color='#002EFF', linestyle=':', linewidth=3,zorder=3)
            plt.scatter(row[f'vol-{mlp}'],row[f'pe-{mlp}-rel'], label=f'MLP {row["bravais"]} {row["mp_id"]}', edgecolors='k', color='#002EFF', marker='X',s=290, alpha=1,zorder=3)

        elif row['bravais'] == 'hcp':
            plt.plot(row[f'vol-dft_{mlp}-fit'],  row[f'pe-dft_{mlp}-fit'],color = '#028200', linestyle='-', linewidth=3,zorder=2)
            plt.scatter(row[f'vol-dft'], row[f'pe-dft_{mlp}-rel'], label=f'DFT {row["bravais"]} {row["mp_id"]}', color='#117810', marker='h', edgecolors='k', s=330, alpha=1,zorder=2)

            plt.plot(row[f'vol-{mlp}-fit'], row[f'pe-{mlp}-fit'], color='#89b661', linestyle=':', linewidth=3,zorder=2)
            plt.scatter(row[f'vol-{mlp}'],row[f'pe-{mlp}-rel'], label=f'MLP {row["bravais"]} {row["mp_id"]}', edgecolors='k', color='#6aa360', marker="h", s=330, alpha=1,zorder=2)

        elif row['bravais'] == 'bcc':
            plt.plot(row[f'vol-dft_{mlp}-fit'],  row[f'pe-dft_{mlp}-fit'], color = '#1c3191', linestyle='-', linewidth=3,zorder=1)
            plt.scatter(row[f'vol-dft'], row[f'pe-dft_{mlp}-rel'], label=f'DFT {row["bravais"]} {row["mp_id"]}', color='#1c3191', marker='D', edgecolors='k', s=200, alpha=1,zorder=1)

            plt.plot(row[f'vol-{mlp}-fit'], row[f'pe-{mlp}-fit'], color='#6f87e8', linestyle=':', linewidth=3,zorder=1)
            plt.scatter(row[f'vol-{mlp}'],row[f'pe-{mlp}-rel'], label=f'MLP {row["bravais"]} {row["mp_id"]}', edgecolors='k', color='#6f87e8', marker='D', s=200, alpha=1,zorder=1)
    
    plt.ylabel("Potential energy (eV/atom)", fontsize=19, labelpad=0, fontweight='bold')
    plt.xlabel('Volume ' +r'($\mathbf{\AA^3/atom}$)', fontsize=17, labelpad=-1, fontweight='bold') 
    plt.tick_params( axis="x", direction="in", width=2.5, length=9, labelsize=17, pad=4) 
    plt.tick_params( axis="y", direction="in", width=2.5, length=9, labelsize=19, pad=4) 
     

    plt.legend(fontsize=11)
    axs = plt.gca()
    axs.spines['top'].set_linewidth(4)
    axs.spines['bottom'].set_linewidth(4) 
    axs.spines['left'].set_linewidth(4)
    axs.spines['right'].set_linewidth(4)    
    axs.set_box_aspect(1)

    tag=f"{df['bravais'].to_list()[0]}{df['bravais'].to_list()[1]}"

    plt.tight_layout() 
    path=make_dir(os.path.join(os.environ['PLOT'],'eos',system.lower()), return_path=True)
    plt.savefig(os.path.join(path,f'{system}_{tag}-{mlp}.png'))
    plt.close(fig)
    

def scatter_b0(df, mlp, return_scores=False): #TODO: group with elements
    colors = ['#4265ff','#202c7c','#9db0c2','#0032fa','#fa325a','#135718','#78a5bf','#424f8a','#d1200d','#6fa8e8','#638545',
              '#d4686b','#006394','#01248c','#801f18','#417027','#1dccb8','#2f0d94','#4265ff']
    
    
    fcc_mask = df['bravais'] == 'fcc'
    hcp_mask = df['bravais'] == 'hcp'
    bcc_mask = df['bravais'] == 'bcc'
    fcc = df[fcc_mask]
    hcp = df[hcp_mask]
    bcc = df[bcc_mask]
     
    fcc_mae = mean_absolute_error(fcc['b0-dft'],fcc[f'b0-{mlp}'])
    fcc_r2 = r2_score(fcc['b0-dft'],fcc[f'b0-{mlp}'])
    hcp_mae = mean_absolute_error(hcp['b0-dft'],hcp[f'b0-{mlp}'])
    hcp_r2 = r2_score(hcp['b0-dft'],hcp[f'b0-{mlp}'])
    bcc_mae = mean_absolute_error(bcc['b0-dft'],bcc[f'b0-{mlp}'])
    bcc_r2 = r2_score(bcc['b0-dft'],bcc[f'b0-{mlp}'])
    mae = mean_absolute_error(df['b0-dft'],df[f'b0-{mlp}'])
    r2 = r2_score(df['b0-dft'],df[f'b0-{mlp}'])
     

    fig, axs = plt.subplots(figsize=(6.5,6.5))
    axs.set_title(mlp, loc='right',fontsize=17, fontweight='bold', pad=10)
    x=np.arange(-100,500,0.1)
    y=np.arange(-100,500,0.1)
    
    axs.plot(x,y,color = colors[2], linestyle='--', linewidth=2, zorder=1)
    
    axs.scatter(fcc['b0-dft'],fcc[f'b0-{mlp}'],label='FCC', color='#4a6cf2', marker='X', edgecolors='k', s=240, alpha=0.8, zorder=3)
    axs.scatter(hcp['b0-dft'],hcp[f'b0-{mlp}'],label=f'HCP', color='#2dc9b7', marker='h', edgecolors='k', s=360, alpha=1, zorder=2)
    axs.scatter(bcc['b0-dft'],bcc[f'b0-{mlp}'],label=f'BCC', color='#319bd8', marker='D', edgecolors='k', s=200, alpha=0.8, zorder=1)
    
    axs.set_ylabel("MLP bulk modulus (GPa)", fontsize=19, labelpad=0, fontweight='bold')
    axs.set_xlabel('DFT bulk modulus (GPa)', fontsize=17, labelpad=-1, fontweight='bold') 
    axs.tick_params( axis="x", direction="in", width=2.5, length=9, labelsize=17, pad=4) 
    axs.tick_params( axis="y", direction="in", width=2.5, length=9, labelsize=19, pad=4)  

    axs.legend(fontsize=12, ncol=3, frameon=True, loc='upper left', bbox_to_anchor=(0.0, 1.0))
    axs.spines['top'].set_linewidth(4)
    axs.spines['bottom'].set_linewidth(4) 
    axs.spines['left'].set_linewidth(4)
    axs.spines['right'].set_linewidth(4)    
    axs.set_box_aspect(1)

    # axs.text(-45,41,f'MAE: {l2i5_E_mae:.2f}',verticalalignment='top', horizontalalignment='left', fontsize=10, color='black')
    # axs.text(-45, 48, fr'$R^2$: {l2i5_E_r2:.2f}', verticalalignment='top', horizontalalignment='left', fontsize=10, color='black')
    axs.annotate(fr'FCC  MAE: {fcc_mae:.2f},   $R^2$: {fcc_r2:.2f}',(0.97,0.22),xycoords='axes fraction',va='top', horizontalalignment='right', fontsize=13, color='black')
    axs.annotate(fr'HCP  MAE: {hcp_mae:.2f},   $R^2$: {hcp_r2:.2f}',(0.97,0.17),xycoords='axes fraction',va='top', horizontalalignment='right', fontsize=13, color='black')
    axs.annotate(fr'BCC  MAE: {bcc_mae:.2f},   $R^2$: {bcc_r2:.2f}',(0.97, 0.12),xycoords='axes fraction',va='top', horizontalalignment='right', fontsize=13, color='black')    
    axs.annotate(fr'TOT  MAE: {mae:.2f},   $R^2$: {r2:.2f}',(0.97,0.07),xycoords='axes fraction',va='top', horizontalalignment='right', fontsize=13, color='black')

    axs.set_xlim(-50,500)
    axs.set_ylim(-50,500)
    fig.set_layout_engine('tight')
    fig.savefig(os.path.join(plot, 'b0',f'{mlp}_b0.png'))
    if return_scores:
        return mae, r2
    return


def scatter_del_E(df, mlp, return_scores=False): #TODO: group with elements
    colors = ['#4265ff','#202c7c','#9ddel_Ec2','#0032fa','#fa325a','#135718','#78a5bf','#424f8a','#d1200d','#859cb7','#638545',
              '#d4686b','#006394','#01248c','#801f18','#417027','#1dccb8','#2f0d94','#4265ff']
    
    
    fb_mask = df['bravais'] == 'FCC-BCC'
    fh_mask = df['bravais'] == 'FCC-HCP'
    hb_mask = df['bravais'] == 'HCP-BCC'
    fb = df[fb_mask]
    fh = df[fh_mask]
    hb = df[hb_mask]
     
    fb_mae = mean_absolute_error(fb['del_E-dft'],fb[f'del_E-{mlp}'])
    fb_r2 = r2_score(fb['del_E-dft'],fb[f'del_E-{mlp}'])
    fh_mae = mean_absolute_error(fh['del_E-dft'],fh[f'del_E-{mlp}'])
    fh_r2 = r2_score(fh['del_E-dft'],fh[f'del_E-{mlp}'])
    hb_mae = mean_absolute_error(hb['del_E-dft'],hb[f'del_E-{mlp}'])
    hb_r2 = r2_score(hb['del_E-dft'],hb[f'del_E-{mlp}'])
    mae = mean_absolute_error(df['del_E-dft'],df[f'del_E-{mlp}'])
    r2 = r2_score(df['del_E-dft'],df[f'del_E-{mlp}'])
     

    fig, axs = plt.subplots(figsize=(6.5,6.5))
    axs.set_title(mlp, loc='right',fontsize=17, fontweight='bold', pad=10)
    x=np.arange(-3000,3000,0.1)
    y=np.arange(-3000,3000,0.1)
    
    axs.plot(x,y,color = '#859cb7', linestyle='--', linewidth=2, zorder=1)
    
    axs.scatter(fb['del_E-dft'],fb[f'del_E-{mlp}'],label='FCC-BCC', color='#526fff', marker='^', edgecolors='k', s=240, alpha=1, zorder=1)
    axs.scatter(fh['del_E-dft'],fh[f'del_E-{mlp}'],label=f'FCC-HCP', color='#ff5e00', marker='^', edgecolors='k', s=240, alpha=.8, zorder=3)
    axs.scatter(hb['del_E-dft'],hb[f'del_E-{mlp}'],label=f'HCP-BCC', color='#00aeff', marker='^', edgecolors='k', s=240, alpha=0.8, zorder=2)
    axs.set_ylabel(fr'MLP $\Delta$E (meV)', fontsize=19, labelpad=0, fontweight='bold')
    axs.set_xlabel(fr'DFT $\Delta$E (meV)', fontsize=17, labelpad=-1, fontweight='bold') 
    axs.tick_params( axis="x", direction="in", width=3, length=9, labelsize=17, pad=4) 
    axs.tick_params( axis="y", direction="in", width=3, length=9, labelsize=19, pad=4)  

    axs.legend(fontsize=11, ncol=1, frameon=True, loc='upper left', bbox_to_anchor=(0.0, 1.0))
    axs.spines['top'].set_linewidth(4)
    axs.spines['bottom'].set_linewidth(4) 
    axs.spines['left'].set_linewidth(4)
    axs.spines['right'].set_linewidth(4)    
    axs.set_box_aspect(1)

    # axs.text(-45,41,f'MAE: {l2i5_E_mae:.2f}',verticalalignment='top', horizontalalignment='left', fontsize=10, color='black')
    # axs.text(-45, 48, fr'$R^2$: {l2i5_E_r2:.2f}', verticalalignment='top', horizontalalignment='left', fontsize=10, color='black')
    axs.annotate(fr'FCC-BCC  MAE: {fb_mae:.2f}, $R^2$: {fb_r2:.2f}',(0.98,0.23), bbox=dict(facecolor="white",edgecolor="none",alpha=0.7),xycoords='axes fraction',va='top', horizontalalignment='right', fontsize=13, color='black')
    axs.annotate(fr'FCC-HCP  MAE: {fh_mae:.2f}, $R^2$: {fh_r2:.2f}',(0.98,0.18),bbox=dict(facecolor="white",edgecolor="none",alpha=0.7),xycoords='axes fraction',va='top', horizontalalignment='right', fontsize=13, color='black')
    axs.annotate(fr'HCP-BCC  MAE: {hb_mae:.2f}, $R^2$: {hb_r2:.2f}',(0.98, 0.13),bbox=dict(facecolor="white",edgecolor="none",alpha=0.7),xycoords='axes fraction',va='top', horizontalalignment='right', fontsize=13, color='black')    
    axs.annotate(fr'TOTAL  MAE: {mae:.2f},   $R^2$: {r2:.2f}',(0.98,0.08),bbox=dict(facecolor="white",edgecolor="none",alpha=0.7),xycoords='axes fraction',va='top', horizontalalignment='right', fontsize=13, color='black')

    axs.set_xlim(-1000,650)
    axs.set_ylim(-1000,650)
    axs.set_xticks([-750, -500, -250, 0, 250, 500])
    axs.set_yticks([-750, -500, -250, 0, 250, 500])
    fig.set_layout_engine('tight')
    fig.savefig(os.path.join(plot, 'del_E',f'{mlp}_E.png'))
    if return_scores:
        return mae, r2
    return
