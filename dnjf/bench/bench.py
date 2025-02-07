from ase.io import read
import os
import numpy as np
from ase.units import GPa
import pandas as pd
import pickle
from sevenn.calculator import SevenNetCalculator
import sys
from util import save_dict, set_env
from plot import scatter
kBar = 1602.1766208

def get_calc(mlp):
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


def run_n_save(atoms, name, mlp):
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
    print(f'saved outputs')

if __name__ == '__main__':
    set_env(task='bench')
    atoms = read(os.path.join(os.environ['CONF'], f'W_{sys.argv[1]}_nequip_train.xyz'), index=':')
    run_n_save(atoms, name = f'W_{sys.argv[1]}', mlp = sys.argv[2])   
