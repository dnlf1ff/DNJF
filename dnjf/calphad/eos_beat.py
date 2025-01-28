import os
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from mob_utils import *
from logging_utils impoer *
from loguru import logger
import sys

import pandas as pd

from sklearn.metrics import mean_absolute_error, r2_score

from itertools import combinations

def sanitize(mlp, df):
    df[f'pe-{mlp}'] = df[f'pe-{mlp}'].apply(lambda x: eval(x))
    df[f'vol-{mlp}'] = df[f'vol-{mlp}'].apply(lambda x: eval(x))
    return df

def get_mlp_data(mlp, return_df=False):
    systems_ = ['Ag','Au','Co','Cs','Cu','Fe','K','Li','Mg','Na','Ni','Pb','Pd','Pt','Rb','V','Zn','K','W','Mo']
    mlp_df = pd.DataFrame() 
    
    systems = []
    mp_ids = []
    task_ids = []
    bravais_lattices = []
    b0s = []

    for system in systems_:
        df = pd.read_csv(os.path.join(os.environ['OUT'],f'{system}_sort.csv'))
        # df = df.drop_duplicates(subset=['bravais_lattice'],keep='last')
        systems.append(df['formula_pretty'])
        mp_ids.append(df['mp_id'])
        task_ids.append(df['task_id'])
        bravais_lattices.append(df['bravais_lattice'])
        b0s.append(df[f'b0-fit-{mlp}'])
        mlp_df['b0-fp'] = df['b0-fit-fp']
        
        mlp_df['system'] = systems
        mlp_df['mp_id'] = mp_ids
        mlp_df['task_id'] = task_ids
        mlp_df['bravais_lattice'] = bravais_lattices
        mlp_df['b0'] = b0s

        mlp_df.to_csv(os.path.join(os.environ['OUT'],'calc',f'{mlp}-b0.csv'))
    
    if return_df:
        return mlp_df
    return


def del_phases(mlp, return_df = False):
    systems = ['Ag','Au','Co','Cs','Cu','Fe','K','Li','Mg','Na','Ni','Pb','Pd','Pt','Rb','V','Zn','K','W','Mo']
    for system in systems:
        df = pd.read_csv(os.path.join(os.environ['OUT'],f'{system}_sort.csv'))
        new_df = pd.DataFrame()

        combs = combinations(range(len(df)), 2)
        
        for mlp in mlps: #TODO: functional coding, save data in 3d array/tensor
            phases=[]
            mp_ids = []
            del_es = []
            for c1, c2 in combs:
                row1 = df.iloc[c1]
                row2 = df.iloc[c2]
                phases.append(f{'row1["bravais_lattice"][0]}->{row2["bravais_lattice"]}')
                mp_ids.append(f{'mp-{row1["mp_id"].split('-')[1]}->{row2["mp_id"].split('-')[1]}')
                del_e = row1[f'pe-min-{mlp}']-row2[f'pe-min-{mlp}']
                del_es.append(del_e)
            
            new_df['system'] = [f'{system}'] * len(combs)
            new_df['mp_id'] = mp_ids
            new_df['phase'] = phases
            new_df[f'del_e-{mlp}'] = del_es
        
        new_df.to_csv(os.path.join(os.environ['OUT'],f'{system}_del.csv'))
    if return_df:
        return new_df
    return

def get_del_df(return_df=False)
    systems = ['Ag','Au','Co','Cs','Cu','Fe','K','Li','Mg','Na','Ni','Pb','Pd','Pt','Rb','V','Zn','K','W','Mo']
    df = pd.read_csv(os.path.join(os.environ['OUT'],f'{system[0]}_del.csv'))
    systems.pop(0)

    for system in systems:
        df = pd.merge(df, pd.read_csv(os.path.join(os.environ['OUT'],f'{system}_del.csv')))
    
    if return_df:
        return df
    return 



def scatter_b0(df, mlp, return_scores=False): #TODO: group with elements
    colors = ['#4265ff','#202c7c','#9db0c2','#0032fa','#fa325a','#135718','#78a5bf','#424f8a','#d1200d','#6fa8e8','#638545', '#d4686b','#006394','#01248c','#801f18','#417027','#1dccb8','#2f0d94','#4265ff']
    
    mae = mean_absolute_error(mlp_df['b0-fit-fp'],mlp_df['b0'])
    r2 = r2_score(mlp_df['b0-fit-fp'], mlp_df['b0'])

    fig, axs = plt.subplots(figsize=(8,8))
    axs.set_title(mlp, fontsize=26, fontweight='bold', pad=10)
    x=np.arange(0,500,0.1)
    y=np.arange(0,500,0.1)
    
    axs.plot(x,y,color = colors[2], linestyle='--', linewidth=2, zorder=1)
    axs.scatter(df['b0-fp'], df['b0'],label=f'{mlp}', color=colors[i], marker='o', edgecolors='k', s=200, alpha=1, zorder=2)
    
    axs.set_ylabel("MLP bulk modulus (GPa)", fontsize=23, labelpad=0, fontweight='bold')
    axs.set_xlabel('DFT bulk modulus (GPa)', fontsize=23, labelpad=-1, fontweight='bold') 
    axs.tick_params( axis="x", direction="in", width=3.5, length=13, labelsize=25, pad=4) 
    axs.tick_params( axis="y", direction="in", width=3.5, length=13, labelsize=25, pad=4)  

    axs.legend(fontsize=16)

    axs.spines['top'].set_linewidth(5)
    axs.spines['bottom'].set_linewidth(5) 
    axs.spines['left'].set_linewidth(5)
    axs.spines['right'].set_linewidth(5)    
    axs.set_box_aspect(1)

    # axs.text(-45,41,f'MAE: {l2i5_E_mae:.2f}',verticalalignment='top', horizontalalignment='left', fontsize=10, color='black')
    # axs.text(-45, 48, fr'$R^2$: {l2i5_E_r2:.2f}', verticalalignment='top', horizontalalignment='left', fontsize=10, color='black')
    
    axs.annotate(fr'MAE: {l2i5_E_mae:.2f}\n$R^2$: {l2i5_E_r2:.2f}',(0.1,0.9),xycoords='axes fraction',va='top', horizontalalignment='left', fontsize=10, color='black')
    
    fig.set_layout_engine('tight')
    path= make_dir(os.path.join(os.environ['PLOT'],'parity'), return_path=True)
    fig.savefig(os.path.join((path,f'{mlp}-b0.png')))
    
    if return_scroes:
        return mae, r2
    return

def scatter_delE(df, mlp, return_scores=False): #TODO: group with elements
    colors = ['#4265ff','#202c7c','#9db0c2','#0032fa','#fa325a','#135718','#78a5bf','#424f8a','#d1200d','#6fa8e8','#638545', '#d4686b','#006394','#01248c','#801f18','#417027','#1dccb8','#2f0d94','#4265ff']
    
    mae = mean_absolute_error(df['del_e-fp']*1000,df[f'del_e-{mlp}']*1000)
    r2 = r2_score(df['del_e-fp'],df[f'del_e-{mlp}'])

    fig, axs = plt.subplots(figsize=(8,8))
    axs.set_title(mlp, fontsize=26, fontweight='bold', pad=10)
    x=np.arange(0,100,0.1)
    y=np.arange(0,100,0.1)
    
    axs.plot(x,y,color = colors[2], linestyle='--', linewidth=2, zorder=1)
    axs.scatter(df['del_e-fp'], df[f'del_e-{mlp}'],label=f'{mlp}', color='#2387a3', marker='^', edgecolors='k', s=200, alpha=1, zorder=2)
    
    axs.set_ylabel("MLP del E (meV)", fontsize=23, labelpad=0, fontweight='bold')
    axs.set_xlabel('DFT del E (meV)', fontsize=23, labelpad=-1, fontweight='bold') 
    axs.tick_params( axis="x", direction="in", width=3.5, length=13, labelsize=25, pad=4) 
    axs.tick_params( axis="y", direction="in", width=3.5, length=13, labelsize=25, pad=4)  

    axs.legend(fontsize=16)

    axs.spines['top'].set_linewidth(5)
    axs.spines['bottom'].set_linewidth(5) 
    axs.spines['left'].set_linewidth(5)
    axs.spines['right'].set_linewidth(5)    
    axs.set_box_aspect(1)

    # axs.text(-45,41,f'MAE: {l2i5_E_mae:.2f}',verticalalignment='top', horizontalalignment='left', fontsize=10, color='black')
    # axs.text(-45, 48, fr'$R^2$: {l2i5_E_r2:.2f}', verticalalignment='top', horizontalalignment='left', fontsize=10, color='black')
    
    axs.annotate(fr'MAE: {l2i5_E_mae:.2f}\n$R^2$: {l2i5_E_r2:.2f}',(0.1,0.9),xycoords='axes fraction',va='top', horizontalalignment='left', fontsize=10, color='black')
    
    fig.set_layout_engine('tight')
    path= make_dir(os.path.join(os.environ['PLOT'],'parity'), return_path=True)
    fig.savefig(os.path.join((path,f'{mlp}-delE.png')))
    
    if return_scroes:
        return mae, r2
    return


def beat_mlps(mlps, return_df=False):
    error_df = pd.DataFrame()
    error_df['mlp'] = mlps
    maes = []
    r2s = []
    for mlp in mlps:
        df = get_mlp_data(mlp, return_df=True)
        mae, r2 = scatter_b0(df, mlp, return_scores=True) 
        maes.append(mae)
        r2s.append(r2)
    error_df['mae-b0'] = maes
    error_df['r2-b0'] = r2s
    error_df.to_csv(os.path.join(os.environ(['OUT'], 'b0_error.csv')))
    if return_df:
        return error_df
    return

def dust_mlps(mlps, error_df, return_df=False):
    error_df = pd.DataFrame()
    error_df['mlp'] = mlps
    maes = []
    r2s = []
    for mlp in mlps:
        del_phases(mlp) 
    df = get_del_df(return_df=True)
    for mlp in mlps:
        mae, r2 = scatter_del(df, mlp, return_scores=True)
        maes.append(mae)
        r2s.append(r2)
    error_df['mae-delE'] = maes
    error_df['r2-delE'] r2s
    error_df.to_csv(os.path.join(os.environ['OUT'], 'delE_error.csv'))

    if return_df;
        return error_df
    return

if __name__ == '__main__':
    set_env('eos')
    logger = get_logger('eosanddogdays')
    mlps = ['fp','chgTot','chgTot_l3i3','chgTot_l4i3','chgTot_l3i5','omat_epoch1','omat_epoch2','omat_epoch3','omat_epoch4','omat_ft_r5','m3g_r55','m3g_r6','m3g_n','mace','matsim']
    beat_mlps(mlps) 
    dust_mlps(mlps)
