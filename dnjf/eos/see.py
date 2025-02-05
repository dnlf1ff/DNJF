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

def run_mlp(system, atoms, mlp, device, isif=3, return_results = True,logger=logger):
    logger.log("DEBUG", "run_mlp FUNC")
    
    atoms = set_mlp(atoms, mlp, device)
    filtered = UnitCellFilter(atoms, mask = [True]*6, constant_volume=False)
    calculation_type='relaxation'
    if isif == 2:
        filtered = UnitCellFilter(atoms, mask = [False]*6, constant_volume=True)
        calculation_type='static calculation'

    optimizer = FIRE(filtered, logfile=os.path.join(os.environ['LOG'],'mlp',system.lower(),f'{system.lower()}.{mlp}.ase'), trajectory=os.path.join(os.environ['TRAJ'],f'{system}.{mlp}.traj'))
    
    optimizer.run(fmax=0.0001)
    natoms=len(atoms)
    pe = atoms.get_potential_energy(force_consistent=True)
    vol = atoms.get_volume()
    force = atoms.get_forces()
    stress = atoms.get_stress()/GPa
    logger.debug(f"OUTPUT {calculation_type} with ase calculator - {natoms} {pe} {vol} {stress} {force}")

    if return_results:
        return len(atoms), pe, vol, force, stress
    return

def vasp_out(out, device = device, logger = logger, return_out=True):
    logger.info("getting vol, pe, force, stress for vasp results")
    nions = []
    pes = []
    vols = []
    forces = []
    stresses = []
    for i, row in out.iterrows():
        system=row['system']
        path = os.path.join(os.environ['DFT'], system.lower(), row['bravais'])
        vasprun = Vasprun(os.path.join(path, 'vasprun.xml'))
        final_structure = vasprun.final_structure.as_dict()
        ionic_step = vasprun.ionic_steps[-1]
        pe = ionic_step['e_fr_energy']
        vol = final_structure['lattice']['volume'] 
        stress = np.asarray(ionic_step['stress'])
        force = np.asarray(ionic_step['forces'])

        pes.append(pe)
        vols.append(vol)
        forces.append(force)
        stresses.append(stress)

    out[f'force-{mlp}'] = forces
    out[f'vol-{mlp}'] = vols
    out[f'pe-{mlp}'] = pes 
    out[f'stress-{mlp}'] = stresses
    save_dict(out, os.path.join(os.environ['JAR'], f'{system}.pkl'))
    if return_out:
        return out
    return
 
def mlp_out(out, mlps, device = device, logger = logger, return_out=True):
    for mlp in mlps:
        logger.info(f"getting vol, pe, force ,stress for {mlp}")
        nions = []
        pes = []
        vols = []
        forces = []
        stresses = []
        for i, row in out.iterrows():
            system=row['system']
            path = os.path.join(os.environ['DFT'], system.lower(), row['bravais'])
            atoms = read(os.path.join(path, 'POSCAR'), format='vasp')
            nion, pe, vol, force, stress = run_mlp(system, atoms, mlp, device=device) 
            nions.append(nion)
            pes.append(pe)
            vols.append(vol)
            forces.append(force)
            stresses.append(stress)
        out['nion'] = nions
        out[f'force-{mlp}'] = forces
        out[f'vol-{mlp}'] = vols
        out[f'pe-{mlp}'] = pes 
        out[f'stress-{mlp}'] = stresses
        save_dict(out, os.path.join(os.environ['JAR'], f'{system}.pkl'))
    if return_out:
        return out
    return

def run_mlp(sys.argv):
    if sys.argv[-1] in ['matsim', 'mace']
        systems = sys.argv[1:-1]
        name = mlps
        mlps = [sys.argv[-1]]
    else:
        systems = sys.argv[1:]
        name = 'svn'
        mlps = ['chgTot','chgTot_l3i3','chgTot_l3i5','chgTot_l4i3','m3g_n','m3g_r6','m3g_r55','omat_epoch1','omat_epoch2','omat_epoch3','omat_epoch4','omat_ft_r5','r5pp','omat_i5pp_epoch1','omat_i5pp_epoch2','omat_i5pp_epoch3','omat_i5pp_epoch4','omat_i5_epoch1','omat_i5_epoch2','omat_i5_epoch3','omat_i5_epoch4','omat_i3pp', 'ompa_i5pp_epoch1','ompa_i5pp_epoch2','ompa_i5pp_epoch3','ompa_itpp_epoch4']
    device = get_device()
    logger = get_logger(system=system, logfile=f'{name}.log', job= 'parity')
    logger.info(f'device: {device}')
    for system in systems:
        out = load_dict(os.path.join(os.environ['JAR'],f'{system}0.pkl'))
        out = vasp_out(out=out,mlp=mlp, device=device) 
        save_dict(out, os.path.join(os.environ['JAR'],f'{system.}pkl'))
        out = mlp_out(out, mlps)
        save_dict(out, os.path.join(os.environ['JAR'],f'{system.}pkl'))
    return

if __name__ == '__main__':
    run_mlp(systems=sys.argv[1:])
