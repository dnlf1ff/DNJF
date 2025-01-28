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
