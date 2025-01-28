from ase.units import GPa
from ase.io.trajectory import Trajectory
from ase.io import read, write
from ase.optimize import FIRE 
from ase.constraints import UnitCellFilter

from sevenn.sevennet_calculator import SevenNetCalculator
# from mace.calculators import mace_mp
# from mattersim.forcefield import MatterSimCalculator
import copy

from loguru import logger
import numpy as np
import os
import pandas as pd

from mob_utils import *
from logging_utils import *

def set_mlp(atoms, mlp, device, return_mlp=False,logger=logger):
    logger.log("FROST", "set mlp function")
    if 'mace' in mlp.lower():
        calculator = mace_mp(model='medium', dispersion = False, default_dtype = 'float32', device='cpu')
        logger.debug(f"MLP = {mlp}")
    elif 'matsim' in mlp.lower() or 'matter' in mlp.lower():
        calculator = MatterSimCalculator(device=device)
        logger.debug(f"MLP = {mlp}")
    else:
        mlp_path = os.path.join(os.environ['MLP'],mlp,'checkpoint_best.pth')
        logger.debug(f"MLP = {mlp_path}")
        calculator = SevenNetCalculator(model= mlp_path, device=device)
    atoms.calc = calculator
    if return_mlp:
        return atoms, calculator
    return atoms

def run_mlp(atoms, mlp, device, trajfile, isif=3, return_results = True,logger=logger):
    logger.log("FROST", "run_mlp FUNC")
    os.environ['TRAJ'] = os.path.join(os.environ['OUT'], 'calc','traj')
    atoms = set_mlp(atoms, mlp, device)
    filtered = UnitCellFilter(atoms, mask = [True]*6, constant_volume=False)
    if isif == 2:
        filtered = UnitCellFilter(atoms, mask = [False]*6, constant_volume=True)
    log_cap = LogCapture("[ASE] structural relaxation")

    optimizer = FIRE(filtered, logfile=log_cap, trajectory='traj.traj')
    optimizer.run(fmax=0.0001)
    natoms=len(atoms)
    pe = atoms.get_potential_energy(force_consistent=True)
    vol = atoms.get_volume()
    force = atoms.get_forces()
    stress = atoms.get_stress()/GPa
    
    logger.debug(f"OUTPUT structural relaxation with ase calculator - {natoms} {pe} {vol} {stress} {force}")

    if return_results:
        return len(atoms), pe, vol, force, stress
    return


def strain_vol(system, atoms, mlp, device, x=0.157, num_points=15,logger=logger):
    logger.log("FROST", "strain_vol FUNC")
    volume_factors = np.linspace(1-x, 1+x, num_points)
    cell = atoms.get_cell()
    a, b, c, alpha, beta, gamma =  cell.cellpar() #[a, b, c, alpha, beta, gamma]
    logger.info("\napplying volumetric strain to the atoms object\n") 
    logger.debug('...checking crystal structure of unit cell...')
    vols = []
    pes = []
    if np.isclose(alpha, 90) and np.isclose(beta, 90) and np.isclose(gamma, 120):
        logger.info('Detected hexagonal (hcp) structure. Applying anisotropic strain')
        ca_ratio = c/a     

        for i, factor in enumerate(volume_factors):
            logger.debug(f"PARAMS {i}th strain applied ...")
            new_vol = atoms.get_volume() * factor
            new_a = (new_vol/(np.sqrt(3) * ca_ratio /2))**(1/3)
            new_c = ca_ratio * new_a
            new_cell = np.array([[new_a, 0, 0], [-0.5*new_a, np.sqrt(3)/2 * new_a, 0], [0,0,new_c]])
            strained_atoms = copy.deepcopy(atoms)
            strained_atoms.set_cell(new_cell, scale_atoms=True)
            
            post_calc = run_mlp(strained_atoms, mlp, device=device, trajfile=os.path.join(os.environ['TRAJ'],f'{system}-eos-hex.traj'), isif=2)
            vols.append(post_calc[2]/post_calc[0])
            pes.append(post_calc[1]/post_calc[0]) 
            logger.debug(f"OUTPUT vols and pes for strained structure ... {vols} {pes}") 
    else:
        logger.info('Detected cubic (fcc, bcc) structure. Applying isotropic strain')
        for i, factor in enumerate(volume_factors):
            logger.debug(f"PARAMS {i}th strain applied ...")
            strained_atoms = copy.deepcopy(atoms)
            new_cell = cell*factor**(1/3)
            strained_atoms.set_cell(new_cell, scale_atoms=True)
            
            post_calc = run_mlp(strained_atoms, mlp, device, trajfile=os.path.join(os.environ['TRAJ'],f'{system}-eos-cbc.traj'), isif=2)
            logger.debug(f"post_calc: {post_calc}")
            vols.append(post_calc[2]/post_calc[0])
            pes.append(post_calc[1]/post_calc[0])
            logger.debug(f"OUTPUT pes and vols for strained structure ... {vols} {pes}") 
    return pes, vols

def run_eos(system, df, mlp, device,x=0.157, num_points=15, logger=logger):
    logger.log("FROST", "run_eos FUNC")
    os.environ['TRAJ'] = os.path.join(os.environ['OUT'], f'{mlp}','traj')
    sys_pes = []
    sys_vols = []
    for index, row in df.iterrows():
        task_id = row['task_id']
        mp_id = row['mp_id']
        path = os.path.join(os.environ['DFT'], system.lower(), row['bravais_lattice'], mp_id, task_id)
        atoms = read(os.path.join(path, 'POSCAR'))
        atoms = set_mlp(atoms, mlp, device)
        run_mlp(atoms, mlp, device, trajfile=os.path.join(os.environ['TRAJ'],f'{system}-{row["bravais_lattice"]}.traj'), isif=3, return_results=False)
            
        logger.info(f"RUNNING strain_vol for {system} {mlp}")
        pes, vols = strain_vol(system=system, atoms=atoms, mlp=mlp, device=device) 
        sys_pes.append(pes)
        sys_vols.append(vols)
        logger.debug(f"OUTPUTS - pes and vols appended: \n{sys_pes} {sys_vols}")
    
    df[f'pe-{mlp}'] = sys_pes
    df[f'vol-{mlp}'] = sys_vols
    return df
 
def run_eos_comp(system, df, mlp, device,x=0.157, num_points=15, logger=logger):
    comp_dict = {'Ag':3, 'Au':2, 'Cd':3, 'Co':[2,3], 'Cs':2, 'Mo':[2,3],'Pt':[1,2],'Pd':2,'V':2,'W':1}
    comp_idx = comp_dict[system]
    
    new_df = pd.DataFrame()

    logger.log("FROST", "run_eos COMP FUNC")
    os.environ['TRAJ'] = os.path.join(os.environ['OUT'], f'{mlp}','traj')
    
    if isinstance(comp_idx, list):
        for idx in comp_idx:
            row = df.loc[idx]
            task_id = row['task_id']
            mp_id = row['mp_id']
            path = os.path.join(os.environ['DFT'], system.lower(), row['bravais_lattice'], mp_id, task_id)
            atoms = read(os.path.join(path, 'POSCAR'))
            atoms = set_mlp(atoms, mlp, device)
            run_mlp(atoms, mlp, device, trajfile=os.path.join(os.environ['TRAJ'],f'{system}-{row["bravais_lattice"]}.traj'), isif=3, return_results=False)
            
            logger.info(f"RUNNING strain_vol for {system} {mlp}")
            pes, vols = strain_vol(system=system, atoms=atoms, mlp=mlp, device=device) 
    
            df[f'pe-{mlp}'][idx] = np.asarray(pes)
            df[f'vol-{mlp}'][idx] = np.asarray(vols)
    
        return df

    if isinstance(comp_idx, int):
        row = df.loc[comp_idx]
        task_id = row['task_id']
        mp_id = row['mp_id']
        path = os.path.join(os.environ['DFT'], system.lower(), row['bravais_lattice'], mp_id, task_id)
        atoms = read(os.path.join(path, 'POSCAR'))
        atoms = set_mlp(atoms, mlp, device)
        run_mlp(atoms, mlp, device, trajfile=os.path.join(os.environ['TRAJ'],f'{system}-{row["bravais_lattice"]}.traj'), isif=3, return_results=False)
            
        logger.info(f"RUNNING strain_vol for {system} {mlp}")
        pes, vols = strain_vol(system=system, atoms=atoms, mlp=mlp, device=device) 
    
        df[f'pe-{mlp}'][comp_idx] = np.asarray(pes)
        df[f'vol-{mlp}'][comp_idx] = np.asarray(vols)

        return df

        
def eos_main(system, task='eos',comp=False):
    set_env(task=task)
    device=get_device()
    logger = get_logger(f'{system.lower()}.{task}.mlp')
    df = pd.read_csv(os.path.join(os.environ['OUT'],f'{system}_mrg.csv'), index_col=0)
    mlps = ['chgTot','chgTot_l3i3','chgTot_l3i5','chgTot_l4i3','m3g_n','m3g_r6','m3g_r55','omat_epoch1','omat_epoch2','omat_epoch3','omat_epoch4','omat_ft_r5','omat_i5pp_epoch1','omat_i5pp_epoch2','omat_i5pp_epoch3','omat_i5pp_epoch4','r5p','r5pp']
    for mlp in mlps:
        if comp:
            df =run_eos_comp(system, df, mlp, device=device)
            df.to_csv(os.path.join(os.environ['OUT'],f'{system}_comp.csv'))
        else:
            df = run_eos(system, df, mlp, device = device, x=0.157, num_points=15)
            df.to_csv(os.path.join(os.environ['OUT'],f'{system}_mlp.csv'))

def run_bench(task='eos'):
    # systems = ['Ag','Au','Co','Cs','Cu','Fe','K','Li','Mg','Na','Ni','Pb','Pd','Pt','Rb','V','K','W','Zn','Cd']
    # systems= ['Pt','Co']
    systems = ['Cu','Cs','Na']
    systems = [systems[int(sys.argv[1])]]
    
    set_env(task=task)
    device=get_device()
    logger = get_logger(f'matsim.{task}.mlp')
    
    for system in systems:
        try:
            df = pd.read_csv(os.path.join(os.environ['OUT'],f'{system}_mrg.csv'), index_col=0)
            df = run_eos(system, df, mlp='mace', device = device, x=0.157, num_points=15)
            df.to_csv(os.path.join(os.environ['OUT'],f'{system}_mace.csv'))

        except Exception as e:
            print(e)
            continue

def run_svn():
    # mlps = ['chgTot','chgTot_l3i3','chgTot_l3i5','chgTot_l4i3','m3g_n','m3g_r6','m3g_r55','omat_epoch1','omat_epoch2','omat_epoch3','omat_epoch4','omat_ft_r5','omat_i5pp_epoch1','omat_i5pp_epoch2','omat_i5pp_epoch3','omat_i5pp_epoch4','r5p','r5pp']
    # mlps = ['omat_i5pp_epoch1','omat_i5pp_epoch2','omat_i5pp_epoch3','omat_i5pp_epoch4','r5p','r5pp','matsim']
    systems=['Ag','Au','Cd','Co','Cs','Mo','Pt','Pd','V','W']
    
    systems = systems[int(sys.argv[1]):int(sys.argv[2])]

    for system in systems:
        eos_main(system=system, task='eos', comp=True)

if __name__ == '__main__':
    run_svn()
