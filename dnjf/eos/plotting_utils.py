import matplotlib.pyplot as plt
import os
from loguru import logger
plt.rc('font', family = 'Arial')

from scipy.stats import gaussian_kde

from data_utils import *
from mob_utils import *

# def del_E_parity(model):
    
#     colors = ['#afafaf', '#dcd789', '#8999dc', '#89dc96', '#dc89c0', '#d59292', '#c192d5']
#     colors2 = ['#0a0a0a', '#f59100', '#0800f9', '#006a0e', '#ff43c9', '#d00000', '#75134f']

#     fig, axs = plt.subplots(figsize=(8,8))
#     axs.set_title('l2i5', fontsize=14, fontweight='bold')    
#     axs.scatter(dft_Es, eval(df_E['l2i5'][0]), c=colors[:5], alpha=1, marker='^', s=250, edgecolors='black')
#     axs.text(-45,41,f'MAE: {l2i5_E_mae:.2f}',verticalalignment='top', horizontalalignment='left', fontsize=10, color='black')
#     axs.text(-45, 48, fr'$R^2$: {l2i5_E_r2:.2f}', verticalalignment='top', horizontalalignment='left', fontsize=10, color='black')
    
#     axs.set_xlabel(r'DFT $\Delta E_{\mathrm{fcc-hcp}}$ (meV/atom)', labelpad=-2, fontsize=13)
#     axs.set_ylabel(r'MLP $\Delta E_{\mathrm{fcc-hcp}}$ (meV/atom)', labelpad=-6.5,fontsize=13)
#     axs.tick_params( axis="x", direction="in", width=2, length=7, labelsize=13, pad=3) 
#     axs.tick_params( axis="y", direction="in", width=2, length=6, labelsize=13, pad=2) 
#     # ax.legend(fontsize=10, facecolor=None, edgecolor='black', frameon=False, loc='upper left') 
#     axs.set_xlim(-50,50)
#     axs.set_ylim(-50,50)
#     axs.set_xticks([-40, -20, 0, 20, 40])
#     axs.set_yticks([-40, -20, 0, 20, 40])
#     axs.set_box_aspect(1)
#     axs.spines['top'].set_linewidth(2.5)
#     axs.spines['bottom'].set_linewidth(2.5) 
#     axs.spines['left'].set_linewidth(2.5)
#     axs.spines['right'].set_linewidth(2.5)    
#     axs.set_box_aspect(1)
#     #TODO: ZOOM
    
#     fig.savefig
    
# def B_parity(model)
# colors = ['#afafaf', '#dcd789', '#8999dc', '#89dc96', '#dc89c0', '#d59292', '#c192d5']
# colors2 = ['#0a0a0a', '#f59100', '#0800f9', '#006a0e', '#ff43c9', '#d00000', '#75134f']

# fig, axs = plt.subplots(1,5, figsize=(20,8))
# for i, ax in enumerate(axs.flat):
#         ax.plot([-50,400],[-50,400], color='grey', linestyle='--', alpha=0.5)
#         if i == 0:
#             ax.set_title('l3i3', fontsize=12, fontweight='bold')    
#             ax.scatter(l3i3['bulk_modulus_dft'][:5], l3i3['bulk_modulus_calc'][:5], c=colors2[:5], alpha=1, marker='D', s=175,edgecolors='black')
#             ax.scatter(l3i3['bulk_modulus_dft'][5:10], l3i3['bulk_modulus_calc'][5:10], c=colors[:5], alpha=1, marker='H',s=200,edgecolors='black')
            
#         if i == 1:
#             ax.set_title('m3g_l3i3_n', fontsize=12, fontweight='bold')
#             ax.scatter(m3g_l3i3_n['bulk_modulus_dft'][:5], m3g_l3i3_n['bulk_modulus_calc'][:5], c=colors2[:5], alpha=1, marker='D', s=175,edgecolors='black')
#             ax.scatter(m3g_l3i3_n['bulk_modulus_dft'][5:10], m3g_l3i3_n['bulk_modulus_calc'][5:10], c=colors[:5], alpha=1, marker='H',s=200,edgecolors='black')
            
#         if i == 2:
#             ax.set_title('m3g_l3i3_r6', fontsize=12, fontweight='bold')
#             ax.scatter(m3g_l3i3_r6['bulk_modulus_dft'][:5], m3g_l3i3_r6['bulk_modulus_calc'][:5], c=colors2[:5], alpha=1, marker='D', s=175,edgecolors='black')
#             ax.scatter(m3g_l3i3_r6['bulk_modulus_dft'][5:10], m3g_l3i3_r6['bulk_modulus_calc'][5:10], c=colors[:5], alpha=1, marker='H',s=200,edgecolors='black')
            
#         if i == 3:
#             ax.set_title('m3g_l3i3_vr646', fontsize=12, fontweight='bold')
#             ax.scatter(m3g_l3i3_vr646['bulk_modulus_dft'][:5], m3g_l3i3_vr646['bulk_modulus_calc'][:5], c=colors2[:5], alpha=1, marker='D', s=175,edgecolors='black')
#             ax.scatter(m3g_l3i3_vr646['bulk_modulus_dft'][5:10], m3g_l3i3_vr646['bulk_modulus_calc'][5:10], c=colors[:5], alpha=1, marker='H',s=200,edgecolors='black')
            
#         if i == 4:
#             ax.set_title('l3i3_omat',fontsize=12, fontweight='bold')
#             ax.scatter(omat_l3i3['bulk_modulus_dft'][:5], omat_l3i3['bulk_modulus_calc'][:5], c=colors2[:5], alpha=1, marker='D', s=175,edgecolors='black')
#             ax.scatter(omat_l3i3['bulk_modulus_dft'][5:10], omat_l3i3['bulk_modulus_calc'][5:10], c=colors[:5], alpha=1, marker='H',s=200,edgecolors='black')
        
#         ax.set_ylabel('MLP bulk modulus (GPa)', labelpad=-1, fontsize=12, fontweight='bold')
#         ax.set_xlabel('DFT bulk modulus (GPa)', labelpad=-1,fontsize=12, fontweight='bold')
#         ax.tick_params( axis="x", direction="in", width=2, length=7, labelsize=13, pad=3) 
#         ax.tick_params( axis="y", direction="in", width=2, length=6, labelsize=13, pad=2) 
#         # ax.legend(fontsize=10, facecolor=None, edgecolor='black', frameon=True) 
#         ax.set_xlim(0,400)
#         ax.set_ylim(0,400)
#         ax.set_xticks([0,50, 150, 250,350])
#         ax.set_yticks([0,50, 150, 250, 350])
#         ax.set_box_aspect(1)
#         ax.spines['top'].set_linewidth(2.5)
#         ax.spines['bottom'].set_linewidth(2.5) 
#         ax.spines['left'].set_linewidth(2.5)
#         ax.spines['right'].set_linewidth(2.5)    
#         ax.set_box_aspect(1)

def plot_eos(system, df, output_dict, structures, mlp):
    #TODO: only dft
    fig, axs = plt.subplots(figsize=(8,8))
    axs.set_title({mlp}, fontsize=26, fontweight='bold', pad=10)
    colors = ['#202c7c','#801f18' ]
    
    for i, structure in enumerate(structures):
        axs.plot(output_dict[f'vol-dft-{structure}-{mlp}-fit'],  output_dict[f'pe-dft-{structure}-{mlp}-fit'], color = colors[i], linestyle='-', linewidth=3, zorder=1)
        axs.scatter(df[f'vol-dft-{structure}'], df[f'pe-dft-{structure}-{mlp}'], label=f'DFT {structure}', color=colors[i], marker='o', edgecolors='k', s=200, alpha=1, zorder=2)
        axs.plot(output_dict[f'vol-{mlp}-{structure}-fit'],  output_dict[f'pe-{mlp}-{structure}-fit'], color=colors[i], linestyle=':', linewidth=3, zorder =1)
        axs.scatter(df[f'vol-{mlp}-{structure}'], df[f'pe-{mlp}-{structure}-rel'], label=f'{mlp} {structure}', edgecolors='k', color=colors[i], marker='^',s=240, alpha=1, zorder=2)
    
    axs.set_ylabel("Potential energy (eV/atom)", fontsize=23, labelpad=0, fontweight='bold')
    axs.set_xlabel('Volume ' +r'($\mathbf{\AA^3/atom}$)', fontsize=23, labelpad=-1, fontweight='bold') 
    axs.tick_params( axis="x", direction="in", width=3.5, length=13, labelsize=25, pad=4) 
    axs.tick_params( axis="y", direction="in", width=3.5, length=13, labelsize=25, pad=4)  

    axs.legend(fontsize=16)

    axs.spines['top'].set_linewidth(5)
    axs.spines['bottom'].set_linewidth(5) 
    axs.spines['left'].set_linewidth(5)
    axs.spines['right'].set_linewidth(5)    
    axs.set_box_aspect(1)

    fig.set_layout_engine('tight')
    
    fig.savefig(os.path.join('..','EOS','plots',f'{system}-{mlp}.png'))
   

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

def mpr_main(element_wise=False, df_dir='../mpr/outputs/mlp'):
    
    if element_wise:
        log_name = 'ew'
    else:
        log_name = 'tot'

    log_dir = make_dir(task='mpr',path='outputs/errors', return_path=True)
    log_filename=os.path.join(log_dir, f'{log_name}.log')
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add(log_filename, level="DEBUG", format="[{time}] {level} -- {message}", rotation="10MB", retention="7days")

    df = pd.read_csv(os.path.join(df_dir, 'mlps.csv'))
    df = df.dropna(subset=['peps-dft'])
    mlps_ = list(df.columns[4:])
    mlps = [mlps_[i].split('-')[1] for i in range(len(mlps_))]
    systems = list(df['system'].drop_duplicates())
    if element_wise:
        logger.info(f'/...setting element-wise to True.../n')
        logger.info(f'will draw parity plots and calculate mae/r2 scores element-wise (ew ..)')
        logger.info(f'element(system) list: {systems}\n')
        
        result_df = pd.DataFrame()
        result_df['system'] = systems
        
        for mlp in mlps:
            maes = []
            r2s = []
            for system in systems:
                logger.info(f'\nprocess on going for {mlp} - {system}\n')
                mae, r2 = mpr_parity(df, mlp, system) 
                logger.info(f'mlp: {mlp} system: {system}, mae: {mae}, r2: {r2}')
                maes.append(mae)
                r2s.append(r2)
            
            result_df[f'mae-{mlp}'] = maes
            result_df[f'r2-{mlp}'] = r2s
        error_dir = make_dir(task='mpr', path='outputs/errors', return_path=True)
        result_df.to_csv(os.path.join(error_dir, 'errors-ew.csv'))
        logger.info(f'\n process done ... saving results to {os.path.join(error_dir,"errors-ew.csv")}\n')
        return

    logger.info('\n ... element-wise set to False ...\n')
    logger.info(f'will draw parity plots and calculate mae/r2 scores for 7net models')
    logger.info(f'...incorporating multiple gga relaxation task ids assigned to a single material-project id')
    logger.info(f'element(system) list: {systems}')

    result_df = pd.DataFrame()
    result_df['mlp'] = mlps
    maes = []
    r2s = []
    for mlp in mlps:
        logger.info(f'\nprocess on going for {mlp}\n')

        mae, r2 = mpr_parity(df, mlp)
        logger.info(f'mlp: {mlp} mae: {mae} r2: {r2}')
        maes.append(mae)
        r2s.append(r2)

    result_df['mae'] = maes
    result_df['r2'] = r2s

    error_dir = make_dir(task='mpr', path='outputs/errors', return_path= True)
    result_df.to_csv(os.path.join(error_dir, 'errors-tot.csv'))
    logger.info('f\nprocess done ... saved results df to {os.path.join(error_dir, "errors-tot.csv")}\n')
    return

if __name__ == '__main__':
    mpr_main(element_wise=True)
