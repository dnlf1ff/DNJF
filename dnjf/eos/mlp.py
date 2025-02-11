from ase.units import GPa
from ase.io.trajectory import Trajectory
from ase.io import read, write
from ase.optimize import FIRE 
from ase.filters import UnitCellFilter
import sys
import copy
import gc
from loguru import logger
import numpy as np
import os
import pandas as pd
import threading
from vasp import write_out, get_vasp_result
from util import load_dict, save_dict, get_device, set_env, group_systems, tot_sys_mlps
from log import get_logger 
import time
kBar = 1602.1766208
import multiprocessing
from numba import jit
from multiprocessing import Lock

def set_mlp(atoms, mlp, device, logger):
    logger.log("DEBUG", "set mlp function")
    if 'mace' in mlp.lower():
        from mace.calculators import mace_mp
        calculator = mace_mp(model='medium', dispersion = False, default_dtype = 'float32', device='cpu')
        logger.debug(f"MLP = {mlp}")
    elif 'matsim' in mlp.lower() or 'matter' in mlp.lower():
        from mattersim.forcefield import MatterSimCalculator
        calculator = MatterSimCalculator(device=device)
        logger.debug(f"MLP = {mlp}")
    else:
        from sevenn.calculator import SevenNetCalculator
        mlp_path = os.path.join(os.environ['MLP'],mlp,'checkpoint_best.pth')
        logger.debug(f"MLP = {mlp_path}")
        calculator = SevenNetCalculator(model= mlp_path, device=device)
    atoms.calc = calculator
    return atoms

@jit(nopython=True)
def compute_stress(stress):
    return -stress*kBar

def run_mlp(system, atoms, mlp, device, logger, isif=3):
    logger.log("DEBUG", "run_mlp FUNC")
    atoms = set_mlp(atoms, mlp, device, logger)
    calculation_type='relaxation'
    if isif == 2:
        filtered = UnitCellFilter(atoms, mask = [False]*6, constant_volume=True)
        calculation_type='static calculation'
    
    optimizer = FIRE(filtered, logfile=os.path.join(os.environ['LOG'],'mlp',f'{system.lower()}.{mlp}.ase'))
    optimizer.run(fmax=0.00001)
    
    natoms=len(atoms)
    pe = atoms.get_potential_energy(force_consistent=True)/natoms
    vol = atoms.get_volume()/natoms
    force = atoms.get_forces()
    stress = compute_stress(atoms.get_stress())
    logger.info(f"\n\nOUTPUT result for {calculation_type} of {system} with {mlp} - {natoms} {pe} {vol} {stress} {force}\n\n")

    return pe, vol, force, stress

def strain_vol(row, mlp, device, logger, num_points=15):
    logger.log("DEBUG", "strain_vol FUNC")
    system=row['system']
    pes, vols, forces, stresses = [], [], [], []
    for i in np.arange(0,num_points,1):
        logger.debug(f"STRAIN {i}th strain applied ...")
        atoms = read(os.path.join(os.environ['DFT'],row['system'].lower(),row['bravais'],'strain',str(i),'POSCAR'), format='vasp')
        pe, vol, force, stress = run_mlp(system, atoms, mlp, device, logger, isif=2)
        pes.append(pe)
        vols.append(vol)
        forces.append(force)
        stresses.append(stress)
        
    atoms.calc = None
    del atoms
    gc.collect()

    return np.asarray(pes), np.asarray(vols), np.asarray(forces), np.asarray(stresses)

def run_eos(system, mlp, device, lock, num_points=15):
    with lock:
        logger.log("DEBUG", "run_eos FUNC")
        out = load_dict(f'{system}_mlp')
        sys_pes, sys_vols, sys_forces, sys_stresses = [], [], [], []
        for i, row in out.iterrows():
            pes, vols, forces, stresses = strain_vol(row, mlp, device, logger) 
            sys_pes.append(pes)
            sys_vols.append(vols)
            sys_forces.append(forces)
            sys_stresses.append(stresses)
            logger.debug(f"ROW {i} - pe, vol, force, stress appended for strain: \n{sys_pes} {sys_vols} {sys_forces} {sys_stresses}")
        out[f'pe-{mlp}'] = sys_pes
        out[f'vol-{mlp}'] = sys_vols
        out[f'force-{mlp}'] = sys_forces
        out[f'stress-{mlp}'] = sys_stresses
        save_dict(out, f'{system}_mlp')

    
def eos_mlps(pbe, systems = None, mlps = None, re=False):

    device = get_device()
    if systems is None or mlps is None:
        systems, mlps = tot_sys_mlps()
    # multiprocessing.set_start_method('spawn', force=True)
    num_workers = 28
    manager = multiprocessing.Manager()
    lock = manager.Lock()

    processes = []
    for mlp in mlps:
        for system in systems:
            logger.info(f"DEVICE: {device}")
            #if not os.path.isfile(os.path.join(os.environ['JAR'],f'{system}_mlp.pkl')):
            #    logger.info(f"collecting DFT results for {system}")
            #    write_out(system)
            #    get_vasp_result(system)
            processes.append((system, mlp, device, lock))
    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.starmap(run_eos, processes)

def get_neglected_out():
    systems = ['Ag','Al','Au','Ca','Cd','Co','Cs','Cu','Fe','Hf','Ir','K','Li','Mg','Mo','Na','Nb','Os','Pd','Pt','Rb','Re','Rh','Sr','Ta','Ti','V','W','Zn','Zr']
    out = group_systems(systems)
    save_dict(out, 'neglected.1')
    return out

def run_multi(pbe, group):
   neg_out = load_dict('neglected0')
   systems = neg_out[group]['systems']
   mlps = neg_out[group]['mlps']
   eos_mlps(pbe=pbe, systems=systems, mlps=mlps)

if __name__ == '__main__':
    set_env('eos',sys.argv[1])
    get_neglected_out()
