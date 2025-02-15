from ase.io import read
import sys, os, gc, pickle
import numpy as np
import pandas as pd
from util import save_dict, set_env, load_dict, tot_sys_mlps, group_systems,  get_device
from plot import scatter
from loguru import logger
from numba import jit

kBar=1602.1766208

try:
    device = get_device()
except:
    import tensorflow as tf    
    device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"

def get_calculator(mlp):
    if 'mace' in mlp.lower():
        calculator=set_mace(mlp)
    elif 'grace' in mlp.lower():
        calculator=set_grace(mlp)
    elif 'mat' in mlp.lower():
        calculator=set_matsim(mlp)
    else:
        calculator=set_seven(mlp)
    logger.debug(f"MLP={mlp}")
    return calculator

def set_grace(mlp):
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.optimizer.set_jit(True)
    from tensorpotential.calculator import grace_fm 

    if 'oam' in mlp.lower():
        if '2l' in mlp.lower():
            calculator=grace_fm('GRACE_2L_OAM_28Jan25')
        else:
            calculator=grace_fm('GRACE-1L-OAM_2Feb25')
    elif '1l' in mlp.lower():
        if 'r6' in mlp.lower():
            calculator=grace_fm('MP_GRACE_1L_r6_07Nov2024')
        else:
            calculator=grace_fm('MP_GRACE_1L_r6_4Nov2024')
    elif '2l' in mlp.lower():
         if 'r6' in mlp.lower():
             calculator=grace_fm('MP_GRACE_2L_r6_11Nov2024')
         else:
             calculator=grace_fm('MP_GRACE_2L_r5_4Nov2024')
    return calculator

def set_seven(mlp):
    from sevenn.calculator import SevenNetCalculator
    calculator= SevenNetCalculator(model=os.path.join(os.environ['JAR'], mlp, 'checkpoint_best.pth'))
    return calculator

def set_matsim(mlp):
    from mattersim.forcefield import MatterSimCalculator
    calculator=MatterSimCalculator(device=device.type)
    return calculator

def set_mace(mlp):
    from mace.calculators import mace_mp
    mace_path=os.path.join(os.environ['MLP'],'mace')
    if 'omat' in mlp.lower():
        calculator=mace_mp(model=os.path.join(mace_path,'mace-omat-0-medium.model'), device=device.type)
    elif 'mpa' in mlp.lower():
        calculator=mace_mp(model=os.path.join(mace_path,'mace-mpa-0-medium.model'), device=device.type)
    else:
        calculator=mace_mp(model='medium',dispersion=False,default_dtype='float32',device=device.type)
    return calculator

@jit(nopython=True)
def compute_stress(stress):
    return -stress*kBar

def run_mlp(atoms, calculator):
    pes, vols, forces, stresses = [], [], [], []
    for i, atom in enumerate(atoms):
        atom.calc = calculator
        print('\nrunning', i)
        nion = len(atom)
        pes.append((pe := atom.get_potential_energy()/nion))
        vols.append((vol := atom.get_volume()/nion))
        forces.append((force := atom.get_forces()))
        stresses.append((stress := compute_stress(atom.get_stress())))
        print(f'nion: {nion}, pe: {pe}, vol: {vol}, force: {force}, stress: {stress}\n')
    del atoms
    gc.collect()
    return pes, vols, forces, stresses

def run_vasp(atoms, system):
    print(f'running {system}')
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


def run_system(system, atoms, calculator,mlp, parity=True):
    if os.path.isfile(os.path.join(os.environ['JAR'],f'{system}_mlp.pkl')):
        out = load_dict(f'{system}_mlp')
    elif '_' not in system:
        print(f'running {system} -- vasp')
        nions, pes, vols, forces, stresses = run_vasp(atoms, system)
        out = pd.DataFrame({'nion': nions, 'pe-dft': pes, 'vol-dft': vols, 'force-dft': forces, 'stress-dft': stresses})
        save_dict(out, f'{system}_mlp')
    else:
        out = pd.DataFrame()
        save_dict(out, f'{system}_mlp')

    pes, vols, forces, stresses = run_mlp(atoms, calculator)
    print(f'\n{system.upper()} {mlp.upper()} DONE\n')
    out[f'pe-{mlp}'] = pes
    out[f'vol-{mlp}'] = vols
    out[f'force-{mlp}'] = forces
    out[f'stress-{mlp}'] = stresses
    save_dict(out, f'{system}_mlp')
    if parity:
        props = ['pe','force','stress']
        for prop in props: 
            scatter(out, prop, system, mlp)
    print(f'saved outputs')
    del out
    gc.collect()

def run_mlps(systems, mlps):
    set_env(task='bench')
    for mlp in mlps:
        calculator = get_calculator(mlp)
        for system in systems:
            atoms = read(os.path.join(os.environ['CONF'],f'{system}.xyz'), index=':')
            run_system(system, atoms, calculator, mlp)   
            del atoms
        del calculator
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
    systems, mlps = tot_sys_mlps(mlp = sys.argv[1])
    run_mlps(systems, mlps)
