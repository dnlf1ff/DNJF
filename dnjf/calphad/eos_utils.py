import os
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from mob_utils import *
from logging_utils import *
from loguru import logger
import sys
import numpy as np

def birch_murnaghan_energy(V, E0, V0, B0, B0_prime):
    eta = (V0 / V)**(2/3)  # Strain factor
    term1 = (eta - 1)
    term2 = 6 - 4 * eta
    return E0 + (9 * V0 * B0 / 16) * (term1**3 * B0_prime + term1**2 * term2)
  
def fit_bm(logger, system, vol, pe, conf, verbose=False, maxfev=20000):
    initial_guess =  conf[system]['initial_guess']
    params, covariance, infodict, mesg, ier = curve_fit(f=birch_murnaghan_energy, xdata=vol, ydata=pe,  p0=initial_guess, full_output=True, maxfev=maxfev)
    if verbose:
        logger.debug(f"Fit Information:")
        logger.debug(f"  Number of function evaluations: {infodict['nfev']}")
        logger.debug(f"  Infodict: {infodict}\n")
        logger.debug(f"  Fit message: {mesg}\n")
        logger.debug(f"  Fit success: {ier}\n")
    
    E0_fit, V0_fit, B0_fit, B0_prime_fit = params
    
    logger.info(f"Fitted Parameters:")
    logger.info(f"  E0 = {E0_fit:.4f} eV/atom")
    logger.info(f"  V0 = {V0_fit:.4f} A³/atom")
    logger.info(f"  B0 = {B0_fit:.6f} eV/A³, {B0_fit * 160.21766208:.4f} GPa")
    logger.info(f"  B0' = {B0_prime_fit:.4f}\n")

    vol_fit = np.linspace(min(vol), max(vol), 300) #200
    pe_fit = birch_murnaghan_energy(vol_fit, *params)
    
    try:
        pe_fit = np.asarray(pe_fit)
    except Exception as e:
        logger.debug(f"EXCEPTION {e}")

    return vol_fit, pe_fit, B0_fit


def sanitize_df(mlp, df):
    logger.info("sanitize_df FUNC")
    try:
        df[f'pe-{mlp}'] = df[f'pe-{mlp}'].apply(lambda x: np.asarray(eval(x)))
    except Exception as e:
        logger.debug(f"while sanitizing df .. {e} occured")
    
    try:
        df[f'vol-{mlp}'] = df[f'vol-{mlp}'].apply(lambda x: np.asarray(eval(x)))
    except Exception as e:
        logger.debug(f"while sanitizing df .. {e} occured")
    return df

def fit_single(system, mlps, conf, return_df = False):
    
    df = pd.read_csv(os.path.join(os.environ['OUT'],f'{system}_mrg.csv' ), index_col=0)
     
    for mlp in mlps:
        df = sanitize_df(mlp, df)

        fited_vols = []
        fited_pes = []
        fited_b0s = [] 
        pe_mins = []
        for row in df.iterrows():
            row = row[1]
            try:
                fit_vol, fit_pe, b0 = fit_bm(logger, system, vol=eval(row[f'vol-{mlp}']), pe=eval(row[f'pe-{mlp}']), conf=conf, verbose=True, maxfev=20000)
            except:
                fit_vol, fit_pe, b0 = fit_bm(logger, system, vol=row[f'vol-{mlp}'], pe=row[f'pe-{mlp}'], conf=conf, verbose=True, maxfev=20000)
            fited_vols.append(fit_vol)
            fited_pes.append(fit_pe)
            fited_b0s.append(b0*160.2176620)
            pe_mins.append(min(eval(row[f'pe-{mlp}'])))
            
        df[f'vol-fit-{mlp}'] = fited_vols
        df[f'fit-pe-{mlp}'] = fited_pes
        df[f'b0-fit-{mlp}'] = fited_b0s
        df[f'pe-min-{mlp}'] = pe_mins
            
        plot_eos(system, df, mlp) #TODO: may not be functional coding but stpol ...
        df.to_csv(os.path.join(os.environ['OUT'],f'{system}_fit.csv' ))


    if return_df:
        return df
    return

def plot_eos(system, df,mlp): #TODO: comb
    fig, axs = plt.subplots(figsize=(8,8))
    axs.set_title({mlp}, fontsize=26, fontweight='bold', pad=10)
    colors = ['#202c7c','#9db0c2','#0032fa','#fa325a','#135718','#78a5bf','#424f8a','#d1200d','#6fa8e8','#638545', '#d4686b','#006394','#01248c','#801f18','#417027','#1dccb8','#2f0d94','#4265ff']
    
    for i, row in enumerate(df.iterrows()): #TODO: choose right structures ...
        colors = ['#202c7c','#9db0c2','#0032fa','#fa325a','#135718','#78a5bf','#424f8a','#d1200d','#6fa8e8','#638545', '#d4686b','#006394','#01248c','#801f18','#417027','#1dccb8','#2f0d94','#4265ff']

        row = row[1]
        axs.plot(row[f'fit-vol-fp'],  row[f'fit-pe-fp'],label=f'DFT {row["bravais_lattice"]} {row["mp_id"]} {row["task_id"]}', color = colors[i], linestyle='-', linewidth=3, zorder=1)
        axs.scatter(row[f'vol-fp'], row[f'pe-fp'], label=f'DFT {row["bravais_lattice"]} {row["mp_id"]} {row["task_id"]}', color=colors[i], marker='o', edgecolors='k', s=200, alpha=1, zorder=2)
        axs.plot(row[f'fit-vol-{mlp}'], row[f'fit-pe-{mlp}'], color=colors[i+1], linestyle=':', linewidth=3, zorder =1)
        axs.scatter(row[f'vol-{mlp}'],row[f'pe-{mlp}'], label=f'{mlp} {row["bravais_lattice"]} {row["mp_id"]} {row["task_id"]}', edgecolors='k', color=colors[i+1], marker='^',s=240, alpha=1, zorder=2)
    
    axs.set_ylabel("Potential energy (eV/atom)", fontsize=23, labelpad=0, fontweight='bold')
    axs.set_xlabel('Volume ' +r'($\mathbf{\AA^3/atom}$)', fontsize=23, labelpad=-1, fontweight='bold') 
    axs.tick_params( axis="x", direction="in", width=3.5, length=13, labelsize=25, pad=4) 
    axs.tick_params( axis="y", direction="in", width=3.5, length=13, labelsize=25, pad=4)  

    axs.legeqd(fontsize=16)

    axs.spines['top'].set_linewidth(5)
    axs.spines['bottom'].set_linewidth(5) 
    axs.spines['left'].set_linewidth(5)
    axs.spines['right'].set_linewidth(5)    
    axs.set_box_aspect(1)

    fig.set_layout_engine('tight')
    path = make_dir(os.path.join(os.environ['PLOT'], 'eos'),return_path=True) 
    fig.savefig(os.path.join(path, f'{system}-all.png'))

def run_fit():
    set_env('eos')
    conf = load_conf(os.path.join(os.environ['PRESET'],'params.yaml'))
    logger = get_logger('eos_fit')
    mlps = ['fp','chgTot','chgTot_l3i3','chgTot_l4i3','chgTot_l3i5','omat_epoch1','omat_epoch2','omat_epoch3','omat_epoch4','omat_ft_r5','m3g_r55','m3g_r6','m3g_n','mace','matsim']
    systems = ['Ag','Au','Co','Fe','K','Li','Mg','Ni','Pb','Pd','Pt','Rb','V','Zn','K','W','Mo']
    systems = systems[int(sys.argv[1]):int(sys.argv[2])] 
    
    for system in systems:
        df = fit_single(system, mlps, conf, return_df=True)
        df =  df.sort_values(by='bravais_lattice',ascending=True).reset_index(drop=True)
        df.to_csv(os.path.join(os.environ['OUT'],f'{system}_sort.csv'))

if __name__ == '__main__':
    run_fit()
