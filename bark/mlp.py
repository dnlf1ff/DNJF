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


def strain_vol(row, mlp, device, x=0.157, num_points=15,logger=logger):
    logger.log("DEBUG", "strain_vol FUNC")
    #volume_factors = np.linspace(1-x, 1+x, num_points)
    volume_factors = np.arange(0,15,1)
    #cell = atoms.get_cell()
    #a, b, c, alpha, beta, gamma =  cell.cellpar() #[a, b, c, alpha, beta, gamma]
    vols = []
    pes = []

    for i in volume_factors:
        logger.debug(f"PARAMS {i}th strain applied ...")

        try:
            atoms = read(os.path.join(os.environ['DFT'],row['system'].lower(),row['bravais'],'strain',str(i),'CONTCAR'), format='vasp')
        except Exception as e:
            logger.debug(f"can't load contcar from strained path? {e}")
            logger.debug("will load poscar file instead")
            atoms = read(os.path.join(os.environ['DFT'],row['system'].lower(),row['bravais'],'strain',str(i),'POSCAR'), format='vasp')
        
        post_calc = run_mlp(row['system'],atoms, mlp, device, isif=2)
        vols.append(post_calc[2]/post_calc[0])
        pes.append(post_calc[1]/post_calc[0])

        logger.debug(f"post_calc: {post_calc}")
        logger.debug(f"OUTPUT pes and vols for strained structure ... {vols} {pes}") 
    
    #if np.isclose(alpha, 90) and np.isclose(beta, 90) and np.isclose(gamma, 120):
    #    logger.info('Detected hexagonal (hcp) structure. Applying anisotropic strain')
    #    ca_ratio = c/a     

    #    for i, factor in enumerate(volume_factors):
    #        logger.debug(f"PARAMS {i}th strain applied ...")
    #        new_vol = atoms.get_volume() * factor
    #        new_a = (new_vol/(np.sqrt(3) * ca_ratio /2))**(1/3)
    #        new_c = ca_ratio * new_a
    #        new_cell = np.array([[new_a, 0, 0], [-0.5*new_a, np.sqrt(3)/2 * new_a, 0], [0,0,new_c]])
    #        strained_atoms = copy.deepcopy(atoms)
    #        strained_atoms.set_cell(new_cell, scale_atoms=True)
    #        
    #        post_calc = run_mlp(system, strained_atoms, mlp, device=device,isif=2)
    #        vols.append(post_calc[2]/post_calc[0])
    #        pes.append(post_calc[1]/post_calc[0]) 
    #        logger.debug(f"OUTPUT vols and pes for strained structure ... {vols} {pes}") 
    #else:
    #    logger.info('Detected cubic (fcc, bcc) structure. Applying isotropic strain')
    #    for i, factor in enumerate(volume_factors):
    #        logger.debug(f"PARAMS {i}th strain applied ...")
    #        strained_atoms = copy.deepcopy(atoms)
    #        new_cell = cell*factor**(1/3)
    #        strained_atoms.set_cell(new_cell, scale_atoms=True)
    #        
    #        post_calc = run_mlp(system, strained_atoms, mlp, device, isif=2)
    #        logger.debug(f"post_calc: {post_calc}")
    #        vols.append(post_calc[2]/post_calc[0])
    #        pes.append(post_calc[1]/post_calc[0])
    #        logger.debug(f"OUTPUT pes and vols for strained structure ... {vols} {pes}") 
    
    return np.asarray(pes,dtype=np.float64), np.asarray(vols,dtype=np.float64)

def run_eos(out, mlp, device,x=0.157, num_points=15, logger=logger):
    logger.log("DEBUG", "run_eos FUNC")
    sys_pes = []
    sys_vols = []
    for row in out.iterrows():
        row=row[1]
        #path = os.path.join(os.environ['DFT'], system.lower(), row['bravais'])
        #atoms = read(os.path.join(path, 'POSCAR'))
        #atoms = set_mlp(atoms, mlp, device)
        #run_mlp(system, atoms, mlp, device,isif=3, return_results=False)
            
        pes, vols = strain_vol(row, mlp=mlp, device=device) 
        sys_pes.append(pes)
        sys_vols.append(vols)
        logger.debug(f"OUTPUTS - pes and vols appended: \n{sys_pes} {sys_vols}")
    
    out[f'pe-{mlp}'] = sys_pes
    out[f'vol-{mlp}'] = sys_vols
    return out
 
def run_bench(systems, mlp='matsim'):
    for system in systems:
        logger = get_logger(system=system, logfile=f'{system}.{mlp}.log', job= 'mlp')
        device = get_device()
        logger.info(f'device: {device}')
        out = load_dict(os.path.join(os.environ['JAR'],f'{system}_mlp.pkl'))
        out = run_eos(out=out, mlp=mlp, device = device)
        save_dict(out, (os.path.join(os.environ['JAR'],f'{system}_mlp.pkl')))


def run_svn(systems, logger=logger):
    for system in systems:
        logger = get_logger(system=system, logfile=f'{system}.svn.log', job= 'mlp')
        mlps = ['chgTot','chgTot_l3i3','chgTot_l3i5','chgTot_l4i3','m3g_n','m3g_r6','m3g_r55','omat_epoch1','omat_epoch2','omat_epoch3','omat_epoch4','omat_ft_r5','r5pp','omat_i5pp_epoch1','omat_i5pp_epoch2','omat_i5pp_epoch3','omat_i5pp_epoch4','omat_i5_epoch1','omat_i5_epoch2','omat_i5_epoch3','omat_i5_epoch4','omat_i3pp']
        device=get_device()
        logger.info(f'device: {device}')
        out = load_dict(os.path.join(os.environ['JAR'],f'{system}_mlp.pkl'))
        for mlp in mlps:
            out = run_eos(out=out,mlp=mlp, device=device) 
            save_dict(out, os.path.join(os.environ['JAR'],f'{system}_mlp.pkl'))
    return


if __name__ == '__main__':
    # run_bench(systems=sys.argv[1:-1], mlp=sys.argv[-1])
    run_svn(systems=sys.argv[1:])
