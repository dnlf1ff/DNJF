import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from util import make_dir 
import numpy as np
import gc

def set_lim(dft_out, mlp_out):
    min_val = min(dft_out.min(), mlp_out.min())
    max_val = max(dft_out.max(), mlp_out.max())
    val_range = abs(max_val-min_val)
    max_lim = max_val + val_range*0.3
    min_lim = min_val - val_range*0.3
    return np.linspace(min_lim, max_lim, 300)

def scatter(out, prop, name, mlp): #TODO: group with elements
    rcParams['font.family'] = 'Arial'
    fig, axs = plt.subplots(figsize=(6.5,6.5))
    if prop == 'force':
        dft_out, mlp_out = force_be(out, mlp)    
        axs.set_ylabel("MLP force (ev/A)", fontsize=19, labelpad=0, fontweight='bold')
        axs.set_xlabel('DFT force (ev/A)', fontsize=17, labelpad=-1, fontweight='bold') 
 
    elif prop == 'stress':
        dft_out, mlp_out = may_the(out, mlp)    
        axs.set_ylabel("MLP stress (kBar)", fontsize=19, labelpad=0, fontweight='bold')
        axs.set_xlabel('DFT stress (kBar)', fontsize=17, labelpad=-1, fontweight='bold') 
 
    else:
        mlp_out = out[f'{prop}-{mlp}']
        dft_out = out[f'{prop}-dft']     
        axs.set_ylabel("MLP energy (ev/atom)", fontsize=19, labelpad=0, fontweight='bold')
        axs.set_xlabel('DFT energy (ev/atom)', fontsize=17, labelpad=-1, fontweight='bold') 
 
    mae = mean_absolute_error(dft_out, mlp_out)
    r2 = r2_score(dft_out, mlp_out) 

    axs.set_title(mlp, loc='right',fontsize=24, fontweight='bold', pad=10)
    lim = set_lim(dft_out, mlp_out)
    axs.plot(lim, lim, color = "#78a5bf", linestyle='--', linewidth=2, zorder=1)
    axs.scatter(dft_out, mlp_out,color='#202c7c', marker='o', edgecolors='k', s=10, alpha=1, zorder=2)
    axs.set_xlim(lim[0], lim[-1]) 
    axs.set_ylim(lim[0], lim[-1]) 
    axs.tick_params( axis="x", direction="in", width=2.5, length=9, labelsize=17, pad=4) 
    axs.tick_params( axis="y", direction="in", width=2.5, length=9, labelsize=19, pad=4)  

    axs.legend(fontsize=12, ncol=3, frameon=True, loc='upper left', bbox_to_anchor=(0.0, 1.0))
    axs.spines['top'].set_linewidth(4)
    axs.spines['bottom'].set_linewidth(4) 
    axs.spines['left'].set_linewidth(4)
    axs.spines['right'].set_linewidth(4)    
    axs.set_box_aspect(1)

    axs.annotate(fr'MAE: {mae:.2f},   $R^2$: {r2:.2f}',(0.97,0.07),xycoords='axes fraction',va='top', horizontalalignment='right', fontsize=13, color='black')

    fig.set_layout_engine('tight')
    
    fig.savefig(os.path.join(os.environ['PLOT'],f'{name}.{mlp}.{prop}.png'))
    plt.close(fig)
    del fig
    gc.collect()

def may_the(out, mlp):
    dft_stress, mlp_stress = out['stress-dft'], out[f'stress-{mlp}'] 
    dft_stress, mlp_stress = np.concatenate([s for s in dft_stress]), np.concatenate([s for s in mlp_stress])
    return dft_stress, mlp_stress

def force_be(out, mlp):
    dft_forces, mlp_forces = out['force-dft'], out[f'force-{mlp}']
    dft_forces, mlp_forces = np.concatenate([f for f in dft_forces]), np.concatenate([f for f in mlp_forces])
   #  x_dft, y_dft, z_dft = np.concatenate([f[0] for f in dft_forces]),np.concatenate([f[1] for f in dft_forces]),np.concatenate([f[2] for f in dft_forces])
    return dft_forces, mlp_forces

