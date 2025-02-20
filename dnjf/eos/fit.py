import os
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from loguru import logger
from util import load_dict, save_dict, tot_sys_mlps, set_env, load_conf
from plot import eos_plot
import sys
import numpy as np
import gc

def birch_murnaghan_energy(V, E0, V0, B0, B0_prime):
    eta = (V0 / V)**(2/3)  # Strain factor
    term1 = (eta - 1)
    term2 = 6 - 4 * eta
    return E0 + (9 * V0 * B0 / 16) * (term1**3 * B0_prime + term1**2 * term2)


def fit_bm(system, vol, pe, conf, verbose=False, maxfev=50000,ftol=1e-8, xtol=1e-8, gtol=1e-8):
    initial_guess =  conf[system]['initial_guess']
    params, covariance, infodict, mesg, ier = curve_fit(f=birch_murnaghan_energy, xdata=vol, ydata=pe,  p0=initial_guess, full_output=True, maxfev=maxfev,ftol=ftol, xtol=xtol, gtol=gtol)
    if verbose:
        logger.debug(f"Fit Information:")
        logger.debug(f"  Number of function evaluations: {infodict['nfev']}")
        logger.debug(f"  Infodict: {infodict}\n")
        logger.debug(f"  Fit message: {mesg}\n")
        logger.debug(f"  Fit success: {ier}\n")
    E0_fit, V0_fit, B0_fit, B0_prime_fit = params
    B0_fit = B0_fit * 160.2176620
    logger.info(f"Fitted Parameters:")
    logger.info(f"  E0 = {E0_fit:.4f} eV/atom")
    logger.info(f"  V0 = {V0_fit:.4f} A³/atom")
    logger.info(f"  B0 = {B0_fit:.6f} eV/A³, {B0_fit} GPa")
    logger.info(f"  B0' = {B0_prime_fit:.4f}\n")
    vol_fit = np.linspace(min(vol), max(vol), 360) #200
    pe_fit = birch_murnaghan_energy(vol_fit, *params)
    return vol_fit, pe_fit, B0_fit #mp.arrays

def fit_dft(system, conf=None, out=None):
    if conf is None:
        conf = load_conf()
    if out is None:
        out = load_dict(f'{system}')
    out['vol-dft-fit'] = None
    out['pe-dft-fit'] = None
    for i, row in out.iterrows():
        try:
            fit_vol, fit_pe, b0 = fit_bm(system, vol=row['vol-dft'], pe=row['pe-dft'], conf=conf, verbose=True)
        except:
            fit_vol, fit_pe, b0 = fit_bm(system, vol=row['vol-dft'], pe=row['pe-dft'], conf=conf, verbose=True,ftol=1e-4,xtol=1e-4, gtol=1e-4)
        out.at[i,'vol-dft-fit'] = fit_vol
        out.at[i, 'pe-dft-fit'] = fit_pe
    save_dict(out, f'{system}_tot')
    return

def fit_mlps(system, mlps, conf=None, out= None):
    if conf is None:
        conf = load_conf()
    if out is None:
        out = load_dict(f'{system}')
    for mlp in mlps:
        out[f'vol-{mlp}-fit'] = None
        out[f'pe-{mlp}-fit'] =  None
        for i, row in out.iterrows():
            try:
                fit_vol_mlp, fit_pe_mlp, b0_mlp = fit_bm(system, vol=row[f'vol-{mlp}'], pe=row[f'pe-{mlp}'], conf=conf, verbose=True)
            except:
                fit_vol_mlp, fit_pe_mlp, b0_mlp = fit_bm(system, vol=row[f'vol-{mlp}'], pe=row[f'pe-{mlp}'], conf=conf, verbose=True,ftol=1e-4,xtol=1e-4, gtol=1e-4)
            pe_min = min(row[f'pe-{mlp}'])
            out.at[i,f'vol-{mlp}-fit'] = fit_vol_mlp #list of np.array
            out.at[i, f'pe-{mlp}-fit'] = fit_pe_mlp # ''

        eos_plot(system, out, mlp,tot=True) #TODO: may not be functional coding but qnd ...
        save_dict(out, f'{system}_tot')
    del mlp, out
    gc.collect()
    return

def fit_main():
    systems, mlps = tot_sys_mlps('tot')
    for system in systems:
        print('system',system)
        out = load_dict(f'bak/{system}')
        fit_dft(system, out=out)
        fit_mlps(system, mlps, out=out)
        del out
        gc.collect()


def scatter_mlps(system, mlps, conf=None, out=None):
    if conf is None:
        conf = load_conf()
    if out is None:
        out = load_dict(f'{system}')
    for mlp in mlps:
        for i, row in out.iterrows():
            plt.scatter(row[f'vol-{mlp}'], row[f'pe-{mlp}'], label=mlp)
        plt.xlabel('Volume (A³/atom)')
        plt.ylabel('Potential Energy (eV/atom)')
        plt.title(f'{system} {mlp} Scatter')
        plt.legend()
        plt.show()
    return


if __name__ == '__main__':
    set_env('eos',sys.argv[1])
    fit_main()
