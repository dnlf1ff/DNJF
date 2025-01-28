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
