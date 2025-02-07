from ase.units import GPa
from ase.io.trajectory import Trajectory
from ase.io import read, write
from ase.optimize import FIRE 
from ase.constraints import UnitCellFilter
import sys
import copy
import gc
from loguru import logger
import numpy as np
import os
import pandas as pd

from vasp import write_out, get_vasp_result
from util import load_dict, save_dict, get_device
from log import get_logger 

def set_mlp(atoms, mlp, device, return_calc=False,logger=logger):
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
    if return_calc:
        return atoms, calculator
    return atoms

def run_mlp(system, atoms, mlp, device, isif=3, return_results = True,logger=logger):
    logger.log("DEBUG", "run_mlp FUNC")
    atoms = set_mlp(atoms, mlp, device)
    calculation_type='relaxation'
    if isif == 2:
        filtered = UnitCellFilter(atoms, mask = [False]*6, constant_volume=True)
        calculation_type='static calculation'
    
    optimizer = FIRE(filtered, logfile=os.path.join(os.environ['LOG'],'mlp',f'{system.lower()}.{mlp}.ase'), trajectory=os.path.join(os.environ['TRAJ'],f'{system}.{mlp}.traj'))
    optimizer.run(fmax=0.00001)
    natoms=len(atoms)
    pe = atoms.get_potential_energy(force_consistent=True)/natoms
    vol = atoms.get_volume()/natoms
    force = atoms.get_forces()
    stress = atoms.get_stress()/GPa
    logger.debug(f"OUTPUT {calculation_type} with ase calculator - {natoms} {pe} {vol} {stress} {force}")

    if return_results:
        return pe, vol, force, stress
    return

def strain_vol(row, mlp, device, x=0.157, num_points=15,logger=logger):
    logger.log("DEBUG", "strain_vol FUNC")
    volume_factors = np.arange(0,15,1)
    system=row['system']
    pes, vols, forces, stress_s = [], [], [], []
    for i in volume_factors:
        logger.debug(f"STRAIN {i}th strain applied ...")
        try:
            atoms = read(os.path.join(os.environ['DFT'],row['system'].lower(),row['bravais'],'strain',str(i),'CONTCAR'), format='vasp')
        except Exception as e:
            logger.debug(f"can't load contcar from strained path? {e}")
            logger.debug("will load poscar file instead")
            atoms = read(os.path.join(os.environ['DFT'],row['system'].lower(),row['bravais'],'strain',str(i),'POSCAR'), format='vasp')
        post_calc = run_mlp(system, atoms, mlp, device, isif=2)
        vols.append(post_calc[1])
        pes.append(post_calc[0])
        forces.append(post_calc[2])
        stress_s.append(post_calc[3])
        logger.debug(f"post_calc: {post_calc}")

        del atoms
        gc.collect()

    return np.asarray(pes), np.asarray(vols), np.asarray(forces), np.asarray(stress_s)

def run_eos(system, mlp, device,x=0.157, num_points=15, logger=logger):
    logger.log("DEBUG", "run_eos FUNC")
    out = load_dict(os.path.join(os.environ['JAR'], f'{system}_mlp.pkl'))
    sys_pes, sys_vols, sys_forces, sys_stress_s = [], [], [], []
    for i, row in out.iterrows():
        pes, vols, forces, stress_s = strain_vol(row, mlp=mlp, device=device) 
        sys_pes.append(pes)
        sys_vols.append(vols)
        sys_forces.append(forces)
        sys_stress_s.append(stress_s)
        logger.debug(f"ROW {i} - pe, vol, force, stress appended for strain: \n{sys_pes} {sys_vols} {sys_forces} {sys_stress_s}")
        save_dict(out, os.path.join(os.environ['JAR'],f'{system}_mlp.pkl'))
        gc.collect()
    
    return out.assign(**{f'pe-{mlp}': sys_pes, f'vol-{mlp}': sys_vols, f'force-{mlp}': sys_forces, f'stress-{mlp}': sys_stress_s})
 
 def pray():
    device = get_device()
    systems = ['Ag','Al','Au','Ca','Cd','Co','Cs','Cu','Fe','Hf','Ir','K','Li','Mg','Mo','Na','Nb','Os','Pd','Pt','Rb','Re','Rh','Sr','Ta','Ti','V','W','Zn','Zr']
    for system in systems:
        logger = get_logger(system=system, logfile=f'{system}.upper().{mlp}.log', job= 'mlp')
        logger.info(f"DEVICE: {device}")
        if not os.path.exists(os.path.join(os.environ['JAR'],f'{system}_mlp.pkl')):
            logger.info(f"collecting DFT results for {system}")
            write_out(system)
            get_vasp_result(system)
        out = run_eos(system=system,mlp=mlp, device=device, logger=logger) 
        save_dict(out, os.path.join(os.environ['JAR'],f'{system}_mlp.pkl'))
        del out, logger
        gc.collect()
    return


def pray_till_you_get_it(re=False):
    device = get_device()
    systems = os.environ['SYSTEMS']
    mlps = os.environ['MLPS'][2:]
    for mlp in mlps:
    for system in systems:
        logger = get_logger(system=system, logfile=f'{system}.upper().{mlp}.log', job= 'mlp')
        logger.info(f"DEVICE: {device}")
        if not os.path.exists(os.path.join(os.environ['JAR'],f'{system}_mlp.pkl')):
            logger.info(f"collecting DFT results for {system}")
            write_out(system)
            get_vasp_result(system)
        out = run_eos(system=system,mlp=mlp, device=device, logger=logger) 
        save_dict(out, os.path.join(os.environ['JAR'],f'{system}_mlp.pkl'))
        del out, logger
        gc.collect()
    return

if __name__ == '__main__':
    pray()
