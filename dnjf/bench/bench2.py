from ase.io import read
import os
import numpy as np
from ase.units import GPa
import pandas as pd
import pickle
from sevenn.calculator import SevenNetCalculator
import sys
from util import save_dict, set_env, load_dict
from plot import scatter
import gc

kBar = 1602.1766208

def get_calc(mlp):
    print(f'\n\nrunning with {mlp}\n\n')
    calc = SevenNetCalculator(os.path.join(os.environ['MLP'],mlp,'checkpoint_best.pth'))
    return calc 

def run_mlp(atoms, name, mlp):
    calc = get_calc(mlp)
    print(f'running {name}')
    pes, vols, forces, stresses = [], [], [], []
    for i, atom in enumerate(atoms):
        print('\nrunning', i)
        nion = len(atom)
        atom.calc = calc
        pes.append((pe := atom.get_potential_energy(force_consistent=True)/nion))
        vols.append((vol := atom.get_volume()/nion))
        forces.append((force := atom.get_forces()))
        stresses.append((stress := -atom.get_stress()*kBar))
        print(f'nion: {nion}, pe: {pe}, vol: {vol}, force: {force}, stress: {stress}\n')
    del calc
    gc.collect()
    return pes, vols, forces, stresses

def run_vasp(atoms, name):
    print(f'running {name}')
    nions, pes, vols, forces, stresses = [], [], [], [], []
    for i, atom in enumerate(atoms):
        print('\nrunning', i)
        nion = len(atom)
        pes.append((pe := atom.get_potential_energy()/nion))
        vols.append((vol := atom.get_volume()/nion))
        forces.append((force := atom.get_forces()))
        stresses.append((stress := -atom.get_stress()*kBar))
        nions.append(nion)
        print(f'nion: {nion}, pe: {pe}, vol: {vol}, force: {force}, stress: {stress}\n')
    return nions, pes, vols, forces, stresses


def run_system(atoms, name, mlp):
    if os.path.isfile(os.path.join(os.environ['JAR'],f'{name}.pkl')):
        out = load_dict(os.path.join(os.environ['JAR'], f'{name}.pkl'))
    else:
        print(f'running {name} -- vasp')
        nions, pes, vols, forces, stresses = run_vasp(atoms, name)
        out = pd.DataFrame({'nion': nions, 'pe-dft': pes, 'vol-dft': vols, 'force-dft': forces, 'stress-dft': stresses})
        save_dict(out, os.path.join(os.environ['JAR'], f'{name}.pkl'))
    
    print(f'running {name} -- {mlp}')
    pes, vols, forces, stresses = run_mlp(atoms, name, mlp)
    out[f'pe-{mlp}'] = pes
    out[f'vol-{mlp}'] = vols
    out[f'force-{mlp}'] = forces
    out[f'stress-{mlp}'] = stresses
    save_dict(out, os.path.join(os.environ['JAR'], f'{name}.pkl'))
    scatter(out, 'pe', name, mlp)
    scatter(out, 'force', name, mlp)
    scatter(out, 'stress', name, mlp)
    print(f'saved outputs')
    del out
    gc.collect()

def run_mlps():
    set_env(task='bench')
    # mlps = ['chgTot','chgTot_l3i3','chgTot_l3i5','chgTot_l4i3','omat_epoch1','omat_epoch2','omat_epoch3','omat_epoch4','omat_i5_epoch1','omat_i5_epoch2','omat_i5_epoch3','omat_i5_epoch4','omat_i3pp','omat_i5pp_epoch1','omat_i5pp_epoch2','omat_i5pp_epoch3','omat_i5pp_epoch4','ompa_i5pp_epoch1','ompa_i5pp_epoch2','ompa_i5pp_epoch3','ompa_i5pp_epoch4', 'omat_ft_r5', 'f5pp']
    mlps = ['omat_i5_epoch3','omat_i5_epoch4','omat_i3pp','omat_i5pp_epoch1','omat_i5pp_epoch2','omat_i5pp_epoch3','omat_i5pp_epoch4','ompa_i5pp_epoch1','ompa_i5pp_epoch2','ompa_i5pp_epoch3','ompa_i5pp_epoch4', 'omat_ft_r5', 'f5pp']
    # mlps = ['omat_i5_epoch1','omat_i5_epoch2','omat_i5_epoch3','omat_i5_epoch4','omat_i3pp','omat_i5pp_epoch1','omat_i5pp_epoch2','omat_i5pp_epoch3','omat_i5pp_epoch4','ompa_i5pp_epoch1','ompa_i5pp_epoch2','ompa_i5pp_epoch3','ompa_i5pp_epoch4', 'omat_ft_r5', 'f5pp']
    # systems = ['Ag','Au','Cd','Co','Cr','Cu','Fe','Hf','Hg','Ir','Mn','Mo','Nb','Ni','Os','Pd','Pt','Re','Rh','Ru','Ta','Tc','Ti','V','W','Zr','Zn']
    systems = ['Ag','Au','Cd','Co','Cr','Cu','Fe','Hf']
    for mlp in mlps:
        for system in systems:
            atoms = read(os.path.join(os.environ['CONF'],f'{system}.xyz'), index=':')
            run_system(atoms, name = system, mlp = mlp)   
        del atoms
        gc.collect()


if __name__ == '__main__':
    run_mlps()
