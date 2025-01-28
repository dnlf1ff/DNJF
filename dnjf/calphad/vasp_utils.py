from ase.io import read
import copy
from mp_api.client import MPRester
from loguru import logger
import numpy as np
import shutil
import subprocess
from pymatgen.io.vasp import Incar, Kpoints, Poscar, Outcar, Potcar
from pymatgen.core import Structure
import os
import pandas as pd

from loguru import *
from logging_utils import *
from mob_utils import *

def get_system_df(df, system=None, mp_id=None):
    try:
        mask = df['formula_pretty'].apply(lambda x: system in x)
    except:
        if mp_id is None:
            mask = df['system'].apply(lambda x: system in x)
        else:
            mask = df['mp_id'].apply(lambda x: x == mp_id)
    return df[mask]

def write_inputs(system, mp_id, task_id, bravais, check_potpaw=True):
    task = mpr.materials.tasks.search([task_id])[0]
    inputs = task['input']
    orig_inputs = task['orig_inputs']
    
    incar=Incar(inputs['incar'])
    kpoints = Kpoints.from_dict(orig_inputs['kpoints'])
    poscar = Poscar(Structure.from_dict(inputs['structure']))
    
    path = os.path.join(os.environ['DFT'], system.lower(), bravais, mp_id, task_id)
    path = make_dir(path, return_path=True)
    
    incar.write_file(os.path.join(path, 'INCAR'))
    kpoints.write_file(os.path.join(path, 'KPOINTS'))
    poscar.write_file(os.path.join(path, 'POSCAR'))
    
    if check_potpaw:
        potcar_spec = inputs['potcar_spec']
        print(f"{potcar_spec}")
        return potcar_spec
    

def pre_dft(system, df):
    df = get_system_df(df,system)
    bravais = df['bravais_lattice']
    mp_ids = df['mp_id']
    task_ids = df['task_id']
    potpaws = {}
    for bravai, mp_id, task_id in zip(bravais, mp_ids, task_ids):
        potcar_spec = write_inputs(system, mp_id, task_id, bravai, check_potpaw=True)
        potpaws[task_id] = potcar_spec
    return df, potpaws

def get_potpaw(system, df, potpaws):
    for task_id, potcar_spec in potpaws.items():
        pseudo,system, date = potcar_spec[0]['titel'].split(' ')
        print(pseudo, system, date)
        potcar_path = os.environ( ['POTCAR_LEGACY'], system, "POTCAR")
        potcar = Potcar.from_file(potcar_path)
        vasp_potcar_spec = str(potcar).split('\n')[0]
        if vasp_potcar_sepc == potcar_spec:
            print('POTCAR is the same')
            os.chdir(os.path.join(os.environ['POTCAR_LEGACY'], system))
            shutil.copy('POTCAR', os.path.join(os.environ['DFT'], system.lower(), 'POTCAR'))
            os.chdir(os.environ['DFT'])
        
def run_relax(system, df):
    for index, row in df.iterrows():
        task_id = row['task_id']
        mp_id = row['mp_id']
        path = os.path.join(os.environ['DFT'], system.lower(), row['bravais_lattice'], mp_id, task_id)
        job_file = os.path.join(os.environ ['DNJF'], 'jobs','run-vasp.sh')
        shutil.copy(job_file, os.path.join(path,'run.sh'))
        shutil.copy(os.path.join(os.environ['DFT'],system.lower(),'POTCAR'), os.path.join(path,'POTCAR'))
        subprocess.run(['chmod','-w','POTCAR'], cwd=path)
        subprocess.run(['chmod','+x','POTCAR'], cwd=path)
        subprocess.run(['sbatch', 'run.sh'], cwd=path)

def relax_main(system):
    set_env('eos')
    mpr = get_mpr()
    df = pd.read_csv(os.path.join(os.environ['OUT'],f'{system}.csv'), index_col=0)
    df = get_system_df(system=system, df = df)
    df.to_csv(os.path.join(os.environ['OUT'], f'{system}.csv'))
    run_relax(system =system, df=df)


def strain_vol(system, path, x=0.157, num_points=15):
    incar = Incar.from_file('INCAR')
    incar['ISIF'] = 2
    kpoints = Kpoints.from_file('KPOINTS')
    
    atoms = read('CONTCAR')
    volume_factors = np.linspace(1-x, 1+x, num_points)
    cell = atoms.get_cell() 
    a, b, c, alpha, beta, gamma =  cell.cellpar()
    
    if np.isclose(alpha, 90) and np.isclose(beta, 90) and np.isclose(gamma, 120):
        logger.info('Detected hexagonal (hcp) structure. Applying anisotropic strain')
        ca_ratio = c/a 
        
        for i, factor in enumerate(volume_factors):
            strain_path = make_dir(os.path.join(path, 'strain', str(i)),return_path=True)
            new_vol = atoms.get_volume() * factor
            new_a = (new_vol/(np.sqrt(3) * ca_ratio /2))**(1/3)
            new_c = ca_ratio * new_a
            new_cell = np.array([[new_a, 0, 0], [-0.5*new_a, np.sqrt(3)/2 * new_a, 0], [0,0,new_c]])
            strained_atoms = copy.deepcopy(atoms)    
            strained_atoms.set_cell(new_cell, scale_atoms=True)
            
            incar.write_file(os.path.join(strain_path, 'INCAR'))
            kpoints.write_file(os.path.join(strain_path, 'KPOINTS'))
            strained_atoms.write(os.path.join(strain_path, 'POSCAR'))
            shutil.copy(os.path.join(os.environ['DFT'],system.lower(),'POTCAR'), os.path.join(strain_path,'POTCAR'))
            subprocess.run(['chmod','-w','POTCAR'], cwd=strain_path)
            subprocess.run(['chmod','+x','POTCAR'], cwd=strain_path)
    else:
        logger.info('Detected cubic (fcc, bcc) structure. Applying isotropic strain')
        for i, factor in enumerate(volume_factors):
            strain_path = make_dir(os.path.join(os.getcwd(), 'strain', str(i)), return_path=True)
            strained_atoms = copy.deepcopy(atoms)
            new_cell = cell*factor**(1/3)
            strained_atoms.set_cell(new_cell, scale_atoms=True) 
            
            incar.write_file(os.path.join(strain_path, 'INCAR'))
            kpoints.write_file(os.path.join(strain_path, 'KPOINTS'))
            strained_atoms.write(os.path.join(strain_path, 'POSCAR')) 
            shutil.copy(os.path.join(os.environ['DFT'],system.lower(),'POTCAR'), os.path.join(strain_path,'POTCAR'))
            subprocess.run(['chmod','-w','POTCAR'], cwd=strain_path)
            subprocess.run(['chmod','+x','POTCAR'], cwd=strain_path)

def run_eos(system, df, x=0.157, num_points=15):
        for index, row in df.iterrows():
            task_id = row['task_id']
            mp_id = row['mp_id']
            path = make_dir(os.path.join(os.environ['DFT'], system.lower(), row['bravais_lattice'], mp_id, task_id,'strain'),return_path=True)
            
            job_file = os.path.join(os.environ ['DNJF'], 'jobs','run-eos.sh')
            shutil.copy(job_file, os.path.join(path,'run.sh'))
            os.chdir(os.path.join(path, '..'))
            
            try:
                strain_vol(system=system, path=os.getcwd(), x=x, num_points=num_points) 
            except:
                continue
            subprocess.run(['sbatch', 'run.sh'], cwd=path)

def eos_main(system, task='eos'):
    set_env(task)
    df = pd.read_csv(os.path.join(os.environ['OUT'],f'{system}.csv'), index_col=0)
    run_eos(system, df)


def get_vasp_results(system, task='eos', logger=logger):
    set_env(task)
    df = pd.read_csv(os.path.join(os.environ['OUT'],f'{system}.csv'), index_col=0)
    volume_factors = np.linspace(0,15,15)
    vols_fp = []
    pes_fp = []
    
    for index, row in df.iterrows():
            task_id = row['task_id']
            mp_id = row['mp_id']
            path = os.path.join(os.environ['DFT'], system.lower(), row['bravais_lattice'], mp_id, task_id,'strain')
            vols = []
            pes = []
            for i, factor in enumerate(volume_factors):
                strain_path = os.path.join(path, str(i))
                logger.log("TLQKF",f"strain_path : {strain_path}")
                outcar = Outcar(os.path.join(strain_path, "OUTCAR"))
                a = read(os.path.join(strain_path,'POSCAR'))
                 
                vols.append(a.get_volume()/len(a))
                pes.append(outcar.final_fr_energy/len(a))
                logger.log("TLQKF",f"{i}th volume: {a.get_volume()/len(a)}") 
                logger.log("TLQKF",f"{i}th pe: {outcar.final_fr_energy/len(a)}")
                logger.log("TLQKF", f"appended volumes: {vols}")
                logger.log("TLQKF",f"appended pes: {pes}")
            vols_fp.append(vols)
            pes_fp.append(pes)
    
    df['pe-fp'] = pes_fp
    df['vol-fp'] = vols_fp
    df.to_csv(os.path.join(os.environ['OUT'],f'{system}_fp.csv'))


if __name__ == '__main__':
    # relax_main('Au')
    # relax_main('Ag')
    # eos_main('Au')
    # eos_main('Co')
    # eos_main('Cu')

    # eos_main('Ni')
    # eos_main('Na')
    # eos_main('Mo')
    # eos_main('Li')

    # Cs, Fe error
    # eos_main('Cs')
    # eos_main('Rb')
    # eos_main('V')
    # eos_main('W')
    # eos_main('Pd')
    # eos_main('Cd')
    # relax_main('Zn')
    set_env('eos')
    logger = get_logger('vasp.out')
    systems = ['Co','Cu','Cs','Ag','Au','Na','Mo','Ni','Pd','Pt','Li','Fe','Rb','Cd','Cr',]
    # systems = systems[:1]
    # for system in systems:
        # try:
         #   get_vasp_results(system=system)
        #except Exception as e:
        #    print(e)


    eos_main('V')
    eos_main('W')

