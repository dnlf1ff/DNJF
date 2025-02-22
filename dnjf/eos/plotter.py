import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import os
from sklearn.metrics import r2_score, mean_absolute_error
from util import make_dir
import gc

colors = ['#4265ff','#202c7c','#9ddel_Ec2','#0032fa','#fa325a','#135718','#78a5bf','#424f8a','#d1200d','#859cb7','#638545','#d4686b','#006394','#01248c','#801f18','#417027','#1dccb8','#2f0d94','#4265ff']
rcParams['font.family'] = 'Arial'

def eos_plot(system, out, mlp, tot=False):
    fig = plt.figure(figsize=(6.5,6.5))
    plt.title(f'{system}-{mlp}', fontsize=17, fontweight='bold', pad=10, loc='right')
    for i, row in enumerate(out.iterrows()):
        row = row[1]
        if row['bravais'] == 'fcc':
            plt.plot(row[f'vol-dft-fit'],  row[f'pe-dft-fit'], color = '#080C85', linestyle='-', linewidth=3,zorder=3)
            plt.scatter(row[f'vol-dft'], row[f'pe-dft'], label=f'DFT {row["bravais"]} {row["mp_id"]}', color='#080C85', marker='X', edgecolors='k', s=290, alpha=1,zorder=3)
            plt.plot(row[f'vol-{mlp}-fit'], row[f'pe-{mlp}-fit'], color='#002EFF', linestyle=':', linewidth=3,zorder=3)
            plt.scatter(row[f'vol-{mlp}'],row[f'pe-{mlp}'], label=f'MLP {row["bravais"]} {row["mp_id"]}', edgecolors='k', color='#002EFF', marker='X',s=290, alpha=1,zorder=3)

        elif row['bravais'] == 'hcp':
            plt.plot(row[f'vol-dft-fit'],  row[f'pe-dft-fit'],color = '#028200', linestyle='-', linewidth=3,zorder=2)
            plt.scatter(row[f'vol-dft'], row[f'pe-dft'], label=f'DFT {row["bravais"]} {row["mp_id"]}', color='#117810', marker='h', edgecolors='k', s=330, alpha=1,zorder=2)

            plt.plot(row[f'vol-{mlp}-fit'], row[f'pe-{mlp}-fit'], color='#89b661', linestyle=':', linewidth=3,zorder=2)
            plt.scatter(row[f'vol-{mlp}'],row[f'pe-{mlp}'], label=f'MLP {row["bravais"]} {row["mp_id"]}', edgecolors='k', color='#6aa360', marker="h", s=330, alpha=1,zorder=2)

        elif row['bravais'] == 'bcc':
            plt.plot(row[f'vol-dft-fit'],  row[f'pe-dft-fit'], color = '#1c3191', linestyle='-', linewidth=3,zorder=1)
            plt.scatter(row[f'vol-dft'], row[f'pe-dft'], label=f'DFT {row["bravais"]} {row["mp_id"]}', color='#1c3191', marker='D', edgecolors='k', s=200, alpha=1,zorder=1)

            plt.plot(row[f'vol-{mlp}-fit'], row[f'pe-{mlp}-fit'], color='#6f87e8', linestyle=':', linewidth=3,zorder=1)
            plt.scatter(row[f'vol-{mlp}'],row[f'pe-{mlp}'], label=f'MLP {row["bravais"]} {row["mp_id"]}', edgecolors='k', color='#6f87e8', marker='D', s=200, alpha=1,zorder=1)
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
    if tot:
        tag='tot'
    else:
        tag=f"{out['bravais'].to_list()[0]}{out['bravais'].to_list()[1]}"

    plt.tight_layout()
    el_path=make_dir(os.path.join(os.environ['PLOT'],'eos','elwise',system.lower()), return_path=True)
    plt.savefig(os.path.join(el_path,f'{system}_{tag}-{mlp}.png'))
    ml_path=make_dir(os.path.join(os.environ['PLOT'],'eos','mlwise',mlp.lower()), return_path=True)
    plt.savefig(os.path.join(ml_path,f'{system}_{tag}.png'))
    plt.close(fig)
    del fig
    gc.collect()

def scatter_b0(out, mlp): #TODO: group with elements
    system = out['system'].to_list()[0]
    fcc_mask = out['bravais'] == 'fcc'
    hcp_mask = out['bravais'] == 'hcp'
    bcc_mask = out['bravais'] == 'bcc'
    fcc = out[fcc_mask]
    hcp = out[hcp_mask]
    bcc = out[bcc_mask]

    fcc_mae = mean_absolute_error(fcc['b0-dft'],fcc[f'b0-{mlp}'])
    fcc_r2 = r2_score(fcc['b0-dft'],fcc[f'b0-{mlp}'])
    hcp_mae = mean_absolute_error(hcp['b0-dft'],hcp[f'b0-{mlp}'])
    hcp_r2 = r2_score(hcp['b0-dft'],hcp[f'b0-{mlp}'])
    bcc_mae = mean_absolute_error(bcc['b0-dft'],bcc[f'b0-{mlp}'])
    bcc_r2 = r2_score(bcc['b0-dft'],bcc[f'b0-{mlp}'])
    mae = mean_absolute_error(out['b0-dft'],out[f'b0-{mlp}'])
    r2 = r2_score(out['b0-dft'],out[f'b0-{mlp}'])


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

    #axs.set_xlim(-50,500)
    #axs.set_ylim(-50,500)
    fig.set_layout_engine('tight')
    path=make_dir(os.path.join(os.environ['PLOT'],'b0'), return_path=True)
    el_path=make_dir(os.path.join(path,'elwise',system.lower()), return_path=True)
    plt.savefig(os.path.join(el_path,f'{system}_{mlp}.png'))
    ml_path=make_dir(os.path.join(path,'mlwise',mlp.lower()), return_path=True)
    plt.savefig(os.path.join(ml_path,f'{system}_{mlp}.png'))
    plt.close(fig)
    del fig
    gc.collect()


def scatter_del_E(out, mlp): #TODO: group with elements
    system = out['system'].to_list()[0]
    fb_mask = out['bravais'] == 'FCC-BCC'
    fh_mask = out['bravais'] == 'FCC-HCP'
    hb_mask = out['bravais'] == 'HCP-BCC'
    fb = out[fb_mask]
    fh = out[fh_mask]
    hb = out[hb_mask]
    fb_mae = mean_absolute_error(fb['del_E-dft'],fb[f'del_E-{mlp}'])
    fb_r2 = r2_score(fb['del_E-dft'],fb[f'del_E-{mlp}'])
    fh_mae = mean_absolute_error(fh['del_E-dft'],fh[f'del_E-{mlp}'])
    fh_r2 = r2_score(fh['del_E-dft'],fh[f'del_E-{mlp}'])
    hb_mae = mean_absolute_error(hb['del_E-dft'],hb[f'del_E-{mlp}'])
    hb_r2 = r2_score(hb['del_E-dft'],hb[f'del_E-{mlp}'])
    mae = mean_absolute_error(out['del_E-dft'],out[f'del_E-{mlp}'])
    r2 = r2_score(out['del_E-dft'],out[f'del_E-{mlp}'])

    fig, axs = plt.subplots(figsize=(6.5,6.5))
    axs.set_title(mlp, loc='right',fontsize=17, fontweight='bold', pad=10)
    x=np.arange(-1500,1250,0.1)
    y=np.arange(-1500,1250,0.1)
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

    path=make_dir(os.path.join(os.environ['PLOT'],'delE'), return_path=True)
    el_path=make_dir(os.path.join(path,'elwise',system.lower()), return_path=True)
    plt.savefig(os.path.join(el_path,f'{system}_{mlp}.png'))
    ml_path=make_dir(os.path.join(path,'mlwise',mlp.lower()), return_path=True)
    plt.savefig(os.path.join(ml_path,f'{system}_{mlp}.png'))
    plt.close(fig)
    del fig
    gc.collect()

def set_lim(dft_out, mlp_out):
    min_val = min(dft_out.min(), mlp_out.min())
    max_val = max(dft_out.max(), mlp_out.max())
    val_range = abs(max_val-min_val)
    max_lim = max_val + val_range*0.3
    min_lim = min_val - val_range*0.3
    return np.linspace(min_lim, max_lim, 300)

def _scatter(out, prop, system, mlp): #TODO: group with elements
    dft_out, mlp_out = _flatten(out, mlp, prop)
    fig, axs = plt.subplots(figsize=(6.5,6.5))
    if prop == 'force':
        axs.set_ylabel("MLP force (ev/A)", fontsize=19, labelpad=0, fontweight='bold')
        axs.set_xlabel('DFT force (ev/A)', fontsize=17, labelpad=-1, fontweight='bold')

    elif prop == 'stress':
        axs.set_ylabel("MLP stress (kBar)", fontsize=19, labelpad=0, fontweight='bold')
        axs.set_xlabel('DFT stress (kBar)', fontsize=17, labelpad=-1, fontweight='bold')

    else:
        axs.set_ylabel("MLP energy (ev/atom)", fontsize=19, labelpad=0, fontweight='bold')
        axs.set_xlabel('DFT energy (ev/atom)', fontsize=17, labelpad=-1, fontweight='bold')

    mae = mean_absolute_error(dft_out, mlp_out)
    r2 = r2_score(dft_out, mlp_out)

    axs.set_title(f'{system}-{mlp}', loc='right',fontsize=24, fontweight='bold', pad=10)
    lim = set_lim(dft_out, mlp_out)
    axs.plot(lim, lim, color = "#78a5bf", linestyle='--', linewidth=2, zorder=1)
    axs.scatter(dft_out, mlp_out,color='#4952d6', marker='o', edgecolors='#0d114a', s=20, alpha=0.9, zorder=2)
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
    path=make_dir(os.path.join(os.environ['PLOT'],'scatter'), return_path=True)
    el_path=make_dir(os.path.join(path,'elwise',system.lower()), return_path=True)
    plt.savefig(os.path.join(el_path,f'{system}_{mlp}_{prop}.png'))
    ml_path=make_dir(os.path.join(path,'mlwise',mlp.lower()), return_path=True)
    plt.savefig(os.path.join(ml_path,f'{system}_{mlp}_{prop}.png'))
    plt.close(fig)
    del fig
    gc.collect()


def _flatten(out, mlp, prop):
    dft_out, mlp_out = out[f'{prop}-dft'], out[f'{prop}-{mlp}']
    dft_out, mlp_out = np.concatenate([f for f in dft_out]), np.concatenate([f for f in mlp_out])
    return dft_out, mlp_out

