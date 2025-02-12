from ase.io.trajectory import Trajectory
from ase.io import read, write
from ase.optimize import FIRE, BFGS 
from ase.filters import UnitCellFilter
import sys, copy, gc, os
from loguru import logger
import numpy as np
import pandas as pd
from vasp import write_out, get_vasp_result
from util import load_dict, save_dict, get_device, set_env, group_systems, tot_sys_mlps
from numba import jit

kBar = 1602.1766208
device = get_device()

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

def run_mlp(system, atoms, isif=2):
    logger.debug("run_mlp FUNC")
    filtered=UnitCellFilter(atoms, mask=[0]*6, constant_volume=True)
    calculation_type='static calculation'
    optimizer=FIRE(filtered,logfile=os.path.join(os.environ['LOG'],'mlp',f'{system.lower()}.ase'))
    optimizer.run(fmax=0.00001)
    natoms=len(atoms)
    pe=atoms.get_potential_energy()/natoms
    vol=atoms.get_volume()/natoms
    force=atoms.get_forces()
    stress=compute_stress(atoms.get_stress())
    return pe, vol, force, stress

def strain_vol(row, calculator, num_points=15):
    logger.debug("strain_vol FUNC")
    system=row['system']
    pes, vols, forces, stresses = [], [], [], []
    for i in np.arange(0,num_points,1):
        logger.info(f"{row['system'].upper()} {i}th STRAIN applied ...")
        atoms=read(os.path.join(os.environ['DFT'],row['system'].lower(),row['bravais'],'strain',str(i),'POSCAR'), format='vasp')
        atoms.calc = calculator
        pe, vol, force, stress = run_mlp(system, atoms)
        pes.append(pe)
        vols.append(vol)
        forces.append(force)
        stresses.append(stress)
    del atoms
    gc.collect()
    return np.asarray(pes), np.asarray(vols), np.asarray(forces), np.asarray(stresses)

def run_eos(system, mlp, calculator, num_points=15):
    logger.debug("run_eos FUNC")
    out = load_dict(f'{system}_mlp')
    sys_pes, sys_vols, sys_forces, sys_stresses = [], [], [], []
    for i, row in out.iterrows():
        pes, vols, forces, stresses = strain_vol(row, calculator) 
        sys_pes.append(pes)
        sys_vols.append(vols)
        sys_forces.append(forces)
        sys_stresses.append(stresses)
        logger.debug(f"{system.upper()} ROW {i} - pe, vol, force, stress appended for strain: \n{sys_pes} {sys_vols} {sys_forces} {sys_stresses}")
    out[f'pe-{mlp}']=sys_pes
    out[f'vol-{mlp}']=sys_vols
    out[f'force-{mlp}']=sys_forces
    out[f'stress-{mlp}']=sys_stresses
    save_dict(out, f'{system}_mlp')
    
def eos_mlps(systems = None, mlps = None):
    if systems is None or mlps is None:
        systems, mlps = tot_sys_mlps()
    logger.info(f"systems: {systems}, mlps: {mlps}")
    for mlp in mlps:
        calculator=get_calculator(mlp)
        for system in systems:
            run_eos(system, mlp, calculator)
        del calculator
        gc.collect()

def run_neglected(group='B'):
    neg_out = load_dict('neglected0')
    systems, mlps = neg_out[group]['systems'], neg_out[group]['mlps']
    eos_mlps(systems=[systems[0]], mlps=mlps)

if __name__ == '__main__':
    set_env('eos',sys.argv[1])
    systems, mlps=tot_sys_mlps(mlp=sys.argv[2])
    eos_mlps(systems, mlps)
