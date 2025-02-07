import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from util import make_dir 

def scatter(out, prop, name, mlp): #TODO: group with elements
    rcParams['font.family'] = 'Arial'
    if prop == 'force':
        dft_out, mlp_out = force_be(out, mlp)
    elif prop == 'stress':
        dft_out, mlp_out = may_the(out, mlp)
    else:
        mlp_out = out[f'{prop}-{mlp}']
        dft_out = out[f'{prop}-dft'] 
    mae = mean_absolute_error(dft_out, mlp_out)
    r2 = r2_score(dft_out, mlp_out) 

    fig, axs = plt.subplots(figsize=(6.5,6.5))
    axs.set_title(mlp, loc='right',fontsize=20, fontweight='bold', pad=10)
    
    axs.plot(x,y,color = "#78a5bf", linestyle='--', linewidth=2, zorder=1)
    
    axs.scatter(dft_out, mlp_out,color='#202c7c', marker='o', edgecolors='k', s=10, alpha=1, zorder=2)
    
    axs.set_ylabel("MLP energy (ev/atom)", fontsize=19, labelpad=0, fontweight='bold')
    axs.set_xlabel('DFT energy (ev/atom)', fontsize=17, labelpad=-1, fontweight='bold') 
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
    
    fig.savefig(os.path.join(os.environ['PLOT'],f'{name}.{prop}_{mlp}.png'))
    plt.close(fig)


def may_the(out, mlp):
    dft_stress, mlp_stress = out['stress-dft'], out[f'stress-{mlp}'] 
    dft_stress, mlp_stress = np.concatenate([s for s in dft_stress]), np.concatenate([s for s in mlp_stress])
    return dft_stress, mlp_stress

def force_be(out, mlp):
    dft_forces, mlp_forces = out['force-dft'], out[f'force-{mlp}']
    dft_forces, mlp_forces = np.stack([f for f in dft_forces]), np.stack([f for f in mlp_forces])
    x_mlp, y_mlp, z_mlp = np.concatenate([f[0] for f in mlp_forces]),np.concatenate([f[1] for f in mlp_forces]),np.concatenate([f[2] for f in mlp_forces])
    x_dft, y_dft, z_dft = np.concatenate([f[0] for f in dft_forces]),np.concatenate([f[1] for f in dft_forces]),np.concatenate([f[2] for f in dft_forces])
    dft_force = np.concatenate(x_dft, y_dft, z_dft)
    mlp_force = np.concatenate(x_dft, y_dft, z_dft)
    return dft_force, mlp_force

