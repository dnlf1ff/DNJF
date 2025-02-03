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
        conf = load_conf(os.path.join(os.environ['CONF'], 'eos','inp.yaml'))
    if df is None:
        df = load_dict(os.path.join(os.environ['JAR'],f'{system}_mlp.pkl'))
    for mlp in mlps:
        df[f'vol-{mlp}-fit'] = None
        df[f'pe-{mlp}-fit'] =  None
        df[f'vol-dft_{mlp}-fit'] =  None
        df[f'pe-dft_{mlp}-fit'] =  None
        df[f'b0-{mlp}'] = None
        df[f'del_b0-dft']=None
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
            df.at[i, 'del_b0-dft'] = b0_dft

        eos_plot(system, df, mlp) #TODO: may not be functional coding but qnd ...
        if save_df:
            save_dict(df, os.path.join(os.environ['JAR'],f'{system}_{tag}_fit.pkl'))
    if return_df:
        return df
    return

def fit_main(system, mlps, df=None):
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

