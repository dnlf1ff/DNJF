import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import os
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
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
    
    system = df['system'].to_list()[0] 
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


    tag=f"{df['bravais'].to_list()[0]}{df['bravais'].to_list()[1]}"
    
    axs.set_xlim(-50,500)
    axs.set_ylim(-50,500)
    fig.set_layout_engine('tight')
    
    path=make_dir(os.path.join(os.environ['PLOT'],'b0',system.lower()), return_path=True)
    fig.savefig(os.path.join(path,f'{system.lower()}+{tag}_{mlp}_b0.png'))
    if return_scores:
        return mae, r2
    return


def scatter_del_E(df, mlp, return_scores=False): #TODO: group with elements
    colors = ['#4265ff','#202c7c','#9ddel_Ec2','#0032fa','#fa325a','#135718','#78a5bf','#424f8a','#d1200d','#859cb7','#638545',
              '#d4686b','#006394','#01248c','#801f18','#417027','#1dccb8','#2f0d94','#4265ff']
    
    system = df['system'].to_list()[0] 
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

    tag=f"{df['bravais'].to_list()[0]}{df['bravais'].to_list()[1]}"
    
    path=make_dir(os.path.join(os.environ['PLOT'],'delE',system.lower()), return_path=True)
    fig.savefig(os.path.join(path,f'{system.lower()}+{tag}_{mlp}_delE.png'))
    if return_scores:
        return mae, r2
    return
