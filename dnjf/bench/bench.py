from ase.io import read
import os
import numpy as np
from ase.units import GPa
import pandas as pd
import pickle
import sys
from util import save_dict, set_env, load_dict, tot_sys_mlps, group_systems,  get_device
from plot import scatter
import gc
from loguru import logger
kBar = 1602.1766208

def get_mlp(mlp):
    if 'mace' in mlp.lower():
        from mace.calculators import mace_mp as calculator
    elif 'grace' in mlp.lower():
        from tensorpotential.calculator import calculator
    elif 'mat' in mlp.lower()
        from mattersim.forcefield import MatterSimCalculator as calculator
    else:
        from sevenn.calculator import SevenNetCalculator as calculator

def set_grace(mlp):
    if 'oam' in mlp.lower():
        if '2l' in mlp.lower():
            calculator=  calculator('GRACE_2L_OAM_28Jan25')
        else:
            calculator=calculator('GRACE-1L-OAM_2Feb25')
    elif '1l' in mlp.lower():
        if 'r6' in mlp.lower():
            calculator=calculator('MP_GRACE_1L_r6_07Nov2024')
        else:
            calculator=calculator('MP_GRACE_1L_r6_4Nov2024')
    elif '2l' in mlp.lower():
         if 'r6' in mlp.lower():
             calculator=calculator('MP_GRACE_2L_r6_11Nov2024')
         else:
             calculator=calculator('MP_GRACE_2L_r5_07Nov2024')
    return calculator

def set_seven(mlp):
    calculator= SevenNetCalculator(model=os.path.join(os.environ['JAR'], mlp, 'checkpoint_best.pth')
    return calculator

def set_matsim(mlp):
    calculator=calculator(device=device.type)
    return calculator

def set_mace(mlp):
    mace_path=os.path.join(os.path.environ['MLP'],'mace')
    if 'omat' in mlp.lower():
        calculator=calculator(model=os.path.join(mace_path,'mace-omat-0-medium.model'), device=device.type)
    elif 'mpa' in mlp.lower():
        calculator=calculator(model=os.path.join(mace_path,'mace-mpa-0-medium.model'), device=device.type)
    else:
        calculator=calculator(model='medium',dispersion=False,default_dtype='float32',device=device.type)

def set_mlp(atoms, mlp, logger):
    logger.log("DEBUG", "set mlp function")
    get_mlp(mlp)
    if 'mace' in mlp.lower():
        calculator=set_mace(mlp)
    elif 'mat' in mlp.lower():
        calculator=set_matsim(mlp) 
    elif 'grace' in mlp.lower():
        calculator=set_grace(mlp)
    else:
        calculator=set_seven(mlp)
    logger.debug(f"MLP = {mlp}")
    atoms.calc =calculator
    return atoms

def run_mlp(atoms, name, mlp):
    calc = get_calc(mlp)
    print(f'running {name}')
    pes, vols, forces, stresses = [], [], [], []
    for i, atom in enumerate(atoms):
        print('\nrunning', i)
        nion = len(atom)
        atom.calc = calc
        pes.append((pe := atom.get_potential_energy()/nion))
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


def run_system(atoms, name, mlp, parity=True):
    if os.path.isfile(os.path.join(os.environ['JAR'],f'{name}.pkl')):
        out = load_dict(f'{name}')
    elif '_' not in name:
        print(f'running {name} -- vasp')
        nions, pes, vols, forces, stresses = run_vasp(atoms, name)
        out = pd.DataFrame({'nion': nions, 'pe-dft': pes, 'vol-dft': vols, 'force-dft': forces, 'stress-dft': stresses})
        save_dict(out, f'{name}')
    else:
        out = pd.DataFrame()
        save_dict(out, f'{name}')

    print(f'running {name} -- {mlp}')
    pes, vols, forces, stresses = run_mlp(atoms, name, mlp)

    out[f'pe-{mlp}'] = pes
    out[f'vol-{mlp}'] = vols
    out[f'force-{mlp}'] = forces
    out[f'stress-{mlp}'] = stresses
    save_dict(out, f'{name}')
    if parity:
        props = ['pe','force','stress']
        for prop in props: 
            scatter(out, prop, name, mlp)
    print(f'saved outputs')
    del out
    gc.collect()

def run_mlps(systems, mlps):
    set_env(task='bench')
    for mlp in mlps:
        for system in systems:
            atoms = read(os.path.join(os.environ['CONF'],f'{system}.xyz'), index=':')
            run_system(atoms, name = system, mlp = mlp)   
        del atoms
        gc.collect()

def run_bench(systems, mlps):
    set_env(task='bench')
    for mlp in mlps:
        for system in systems:
            atoms = read(os.path.join(os.environ['CONF'],f'{system}.xyz'), index=':')
            run_system(atoms, name = f'{system}_{mlp}', mlp = mlp, parity=False)   
        del atoms
        gc.collect()


def get_neglected_out():
    set_env('bench')
    systems, _ = tot_sys_mlps()
    out = group_systems(systems)
    save_dict(out, 'neglected.1')

def run_neglected(group):
    set_env('bench')
    out = load_dict('neglected')
    systems = out[group]['systems']
    mlps = out[group]['mlps']
    run_mlps(systems, mlps)


if __name__ == '__main__':
    systems, _ = tot_sys_mlps()
    mlp = sys.argv[1]
    if 'grace' in mlp.lower():
        from tensorpotential.calculator import grace_fm
        if '2l' in mlp.lower():
            mlps = ['GRACE-2L','GRACE-2L_r6']
        elif '1l' in mlp.lower():
            mlps = ['GRACE-1L','GRACE-1L_r6']
        else:
            mlps = ['GRACE-2L-OAM', 'GRACE-1L-OAM']
    run_bench(systems, mlps)
