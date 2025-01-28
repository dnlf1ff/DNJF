import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd

def mpr_parity(df, mlp, system=None, return_errors = True):
    if system is not None:
        df = get_system_df(df, system)

    fig, axs = plt.subplots(figsize=(8,8))
    axs.set_title(mlp, fontsize=26, fontweight='bold', pad=10)
    colors = ['#202c7c' ]

    dft_pe = np.asarray(df['peps-dft'])
    mlp_pe = np.asarray(df[f'peps-{mlp}'])
    
    data = pd.DataFrame({"dft": dft_pe.tolist(), "mlp": mlp_pe.tolist(), "task-id": list(df['mpid']), "element": list(df['system'])})

    mae, r2 = get_errors(dft_pe,mlp_pe)
    xy = np.vstack([dft_pe.tolist(), mlp_pe.tolist()])
    z = gaussian_kde(xy)(xy)
    # sns.scatterplot(data=data, x="dft",y="mlp",)

    axs.plot([-15,15], [-15,15], linestyle='--', color='grey', alpha=0.8, linewidth=2)
    axs.scatter(dft_pe, mlp_pe, label=f'{mlp}', marker='o', edgecolors='k', s=150, alpha=0.8, zorder=2, c=z, cmap='viridis')
    axs.text(2,4,f'MAE: {mae:.2f}',verticalalignment='top', horizontalalignment='left', fontsize=15, color='black')
    axs.text(2,3, fr'$R^2$: {r2:.2f}', verticalalignment='top', horizontalalignment='left', fontsize=15, color='black')
    
#    if system is None:
#        axs.set_xlim(-16,6)
#        axs.set_ylim(-16,6)
#        axs.set_xticks([-15, -10, -5, 0, 5])
#        axs.set_yticks([-15, -10, -5, 0, 5])
#
    axs.set_ylabel("MLP potential energy (eV/atom)", fontsize=23, labelpad=0, fontweight='bold')
    axs.set_xlabel("DFT potential energy (eV/atom)", fontsize=23, labelpad=0, fontweight='bold')
    axs.tick_params( axis="x", direction="in", width=3.5, length=13, labelsize=25, pad=4) 
    axs.tick_params( axis="y", direction="in", width=3.5, length=13, labelsize=25, pad=4)  

    # axs.legend(fontsize=16)

    axs.spines['top'].set_linewidth(5)
    axs.spines['bottom'].set_linewidth(5) 
    axs.spines['left'].set_linewidth(5)
    axs.spines['right'].set_linewidth(5)    
    axs.set_box_aspect(1)

    fig.set_layout_engine('tight')
    
    if system is not None:
        fig_path = make_dir(task='mpr',path=f'plots/mlp/{mlp}', return_path=True)
        fig.savefig(os.path.join(fig_path, f'{system}-{mlp}.png'))
    
    else:
        fig_path = make_dir(task='mpr', path='plots/mlp', return_path=True)
        fig.savefig(os.path.join(fig_path, f'{mlp}.png'))
    
    if return_errors:
        return mae, r2
    return

def scatter_b0(df, mlp, return_scores=False): #TODO: group with elements
    colors = ['#4265ff','#202c7c','#9db0c2','#0032fa','#fa325a','#135718','#78a5bf','#424f8a','#d1200d','#6fa8e8','#638545', '#d4686b','#006394','#01248c','#801f18','#417027','#1dccb8','#2f0d94','#4265ff']
    
    mae = mean_absolute_error(df['b0-fp'],df[f'b0-{mlp}'])
    r2 = r2_score(df['b0-fp'],df[f'b0-{mlp}'])

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
    axs.annotate(fr'MAE: {mae:.2f}\n$R^2$: {r2:.2f}',(0.1,0.9),xycoords='axes fraction',va='top', horizontalalignment='left', fontsize=10, color='black')
    
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

def eos_plot(system, df,mlp):
    fig, axs = plt.subplots(figsize=(8,8))
    axs.set_title(f'{system}-{mlp}', fontsize=20, fontweight='bold', pad=10)
    colors = ['#202c7c','#4d4892','#0032fa','#fa325a','#004027','#78a5bf','#424f8a','#d1200d','#6f87e8','#89b661', '#ffa7a7','#489ec9','#01248c','#801f18','#417027','#1dccb8','#2f0d94','#4265ff']
    
    for i, row in enumerate(df.iterrows()):
        row = row[1]
        if i == 0:
            axs.plot(row[f'vol-fp-fit'],  row[f'pe-fp-fit'], color = '#1b1b1be1', linestyle='-', linewidth=3, zorder=1)
            axs.scatter(row[f'vol-fp'], row[f'pe-fp'], label=f'DFT {row["bravais_lattice"]} {row["mp_id"]}', color='#1b1b1be1', marker='h', edgecolors='k', s=400, alpha=1, zorder=2)
            axs.plot(row[f'vol-{mlp}-fit'], row[f'pe-{mlp}-fit'], color='#828282e6', linestyle=':', linewidth=3, zorder =1)
            axs.scatter(row[f'vol-{mlp}'],row[f'pe-{mlp}'], label=f'MLP {row["bravais_lattice"]} {row["mp_id"]}', edgecolors='k', color='#828282e6', marker='h',s=400, alpha=1, zorder=2)
        if i == 1:
            axs.plot(row[f'vol-fp-fit'],  row[f'pe-fp-fit'],color = '#028200', linestyle='-', linewidth=3, zorder=1)
            axs.scatter(row[f'vol-fp'], row[f'pe-fp'], label=f'DFT {row["bravais_lattice"]} {row["mp_id"]}', color='#117810', marker='X', edgecolors='k', s=350, alpha=1, zorder=2)
            axs.plot(row[f'vol-{mlp}-fit'], row[f'pe-{mlp}-fit'], color='#89b661', linestyle=':', linewidth=3, zorder =1)
            axs.scatter(row[f'vol-{mlp}'],row[f'pe-{mlp}'], label=f'MLP {row["bravais_lattice"]} {row["mp_id"]}', edgecolors='k', color='#6aa360', marker="X", s=350, alpha=1, zorder=2)
        if i == 2:
            axs.plot(row[f'vol-fp-fit'],  row[f'pe-fp-fit'], color = '#0023c1', linestyle='-', linewidth=3, zorder=1)
            axs.scatter(row[f'vol-fp'], row[f'pe-fp'], label=f'DFT {row["bravais_lattice"]} {row["mp_id"]}', color='#0023c1', marker='D', edgecolors='k', s=200, alpha=1, zorder=2)
            axs.plot(row[f'vol-{mlp}-fit'], row[f'pe-{mlp}-fit'], color='#6f87e8', linestyle=':', linewidth=3, zorder =1)
            axs.scatter(row[f'vol-{mlp}'],row[f'pe-{mlp}'], label=f'MLP {row["bravais_lattice"]} {row["mp_id"]}', edgecolors='k', color='#6f87e8', marker='D', s=200, alpha=1, zorder=2)
    
    axs.set_ylabel("Potential energy (eV/atom)", fontsize=23, labelpad=0, fontweight='bold')
    axs.set_xlabel('Volume ' +r'($\mathbf{\AA^3/atom}$)', fontsize=23, labelpad=-1, fontweight='bold') 
    axs.tick_params( axis="x", direction="in", width=3.5, length=13, labelsize=25, pad=4) 
    axs.tick_params( axis="y", direction="in", width=3.5, length=13, labelsize=25, pad=4) 
     

    axs.legend(fontsize=12)

    axs.spines['top'].set_linewidth(5)
    axs.spines['bottom'].set_linewidth(5) 
    axs.spines['left'].set_linewidth(5)
    axs.spines['right'].set_linewidth(5)    
    axs.set_box_aspect(1)

    fig.set_layout_engine('tight')
    path=make_dir(os.path.join(os.environ['PLOT'],'eos',system), return_path=True)
    fig.savefig(os.path.join(path,f'{system}-{mlp}.png'))
