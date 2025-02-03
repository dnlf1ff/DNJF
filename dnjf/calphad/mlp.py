from ase.units import GPa
from ase.io.trajectory import Trajectory
from ase.io import read, write
from ase.optimize import FIRE 
from ase.constraints import UnitCellFilter

import copy

from loguru import logger
import numpy as np
import os
import pandas as pd

import subprocess

from util import *
from log import *

def set_mlp(atoms, mlp, device, return_mlp=False,logger=logger):
    logger.log("DEBUG", "set mlp function")
    if 'mace' in mlp.lower():
        from mace.calculators import mace_mp
        calculator = mace_mp(model='medium', dispersion = False, default_dtype = 'float64', device='cpu')
        logger.debug(f"MLP = {mlp}")
    elif 'matsim' in mlp.lower() or 'matter' in mlp.lower():
        from mattersim.forcefield import MatterSimCalculator
        calculator = MatterSimCalculator(device=device)
        logger.debug(f"MLP = {mlp}")
    else:
        from sevenn.sevennet_calculator import SevenNetCalculator
        mlp_path = os.path.join(os.environ['MLP'],mlp,'checkpoint_best.pth')
        logger.debug(f"MLP = {mlp_path}")
        calculator = SevenNetCalculator(model= mlp_path, device=device)
    atoms.calc = calculator
    if return_mlp:
        return atoms, calculator
    return atoms

def mlp_relax(system, atoms, mlp, device, isif=3, return_results = True,logger=logger):
    logger.log("DEBUG", "run_mlp FUNC")
    
    atoms = set_mlp(atoms, mlp, device)
    log_dir = make_dir(os.path.join(os.environ['LOG'],'mlp',system.lower()), return_path = True)
    filtered = UnitCellFilter(atoms, mask = [True]*6, constant_volume=False)
    if isif == 2:
        filtered = UnitCellFilter(atoms, mask = [False]*6, constant_volume=True)

    optimizer = FIRE(filtered, logfile=os.path.join(log_dir,f'{system.lower()}.{mlp}.ase'), trajectory=os.path.join(os.environ['TRAJ'],f'{system}.{mlp}.traj'))
    
    optimizer.run(fmax=0.0001)
    pe = atoms.get_potential_energy(force_consistent=True)
    force = atoms.get_forces()
    stress = atoms.get_stress()/GPa
    
    logger.debug(f"OUTPUT structural relaxation with ase calculator - {pe} {stress} {force}")

    if return_results:
        return pe, force, stress
    return

def run_mlp(binary_system, mlp, out=None, return_out = False, logger=logger):
    device = get_device()
    if out is None:
        out = load_dict(os.path.join(os.environ['JAR'],f'{binary_system.lower()}.pkl'))
    fes = []
    for i, row in out.iterrows():
        path = os.path.join(os.environ['DFT'],row['system'])
        atoms = read(os.path.join(path, 'POSCAR'), format='vasp') 
        pe, _, _ = mlp_relax(system=row['system'], atoms=atoms, mlp=mlp, device=device) 
        fes.append(pe)
        logger.debug(f"OUTPUT by {mlp} - potential energy for {row['system']} appended \n {fes}")
    
    out[f'fe-{mlp}'] = fes
    save_dict(out, os.path.join(os.environ['JAR'],f'{binary_system.lower()}.pkl'))
    if return_out:
        return out
    return
 

def run_svn(binary_system, logger=logger):
    logger = get_logger(system=binary_system, logfile=f'{binary_system}.mlp.log', job='mlp')
    mlps = ['chgTot','chgTot_l3i3','chgTot_l3i5','chgTot_l4i3','m3g_n','m3g_r6','m3g_r55','omat_epoch1','omat_epoch2','omat_epoch3','omat_epoch4','omat_ft_r5','r5pp','omat_i5pp_epoch1','omat_i5pp_epoch2','omat_i5pp_epoch3','omat_i5pp_epoch4','omat_i5_epoch1','omat_i5_epoch2','omat_i5_epoch3','omat_i5_epoch4','omat_i3pp']
    
    for mlp in mlps:
        out = run_mlp(binary_system,mlp=mlp, return_out=True) 
        save_dict(out, os.path.join(os.environ['JAR'],f'{binary_system}.pkl'))
    return

if __name__ == '__main__':
    run_svn(sys.argv[1])
