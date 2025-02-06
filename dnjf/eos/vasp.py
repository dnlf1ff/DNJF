from ase.io import read
import copy
from mp_api.client import MPRester
import numpy as np
import shutil
import subprocess
from pymatgen.io.vasp import Incar, Kpoints, Poscar, Vasprun
from pymatgen.core import Structure
import os
import pandas as pd
import sys
from util import load_conf, make_dir, load_dict, save_dict 
from shell import vasp_job 
from log import get_logger 

def write_out(system, inp=None, return_out=False):
    if inp is None:
        inp = load_conf()
    path = os.path.join(os.environ['JAR'], f'{system}0.pkl')
    out = pd.DataFrame()
    out['mp_id'] = inp[system]['mp_id']
    out['bravais'] = inp[system]['bravais']
    out['system'] = [system] * len(out)
    save_dict(data=out, path=path)
    if return_out:
        return out
    return

def write_inputs(system, logger, out=None, check_potpaw=True):
    for system in systems:
        if out is None:
            out = load_dict(os.path.join(os.environ['JAR'],f'{system}0.pkl'))
            mpr=MPRester(api_key=os.environ['API_KEY'],use_document_model=False)
            _path = make_dir(os.path.join(os.environ['DFT'], system.lower()), return_path=True)
            shutil.copy(os.path.join(os.environ['PBE'],system.lower(),'POTCAR'), os.path.join(_path,'POTCAR'))
        for row in out.iterrows():
            row=row[1]
            mp_id=row['mp_id']
            task = mpr.materials.tasks.search([mp_id])[0]
            inputs = task['input']
            orig_inputs = task['orig_inputs']
        
            incar=Incar(inputs['incar'])
            kpoints = Kpoints.from_dict(orig_inputs['kpoints'])
            poscar = Poscar(Structure.from_dict(inputs['structure']))
    
            path = make_dir(os.path.join(_path, row['bravais']), return_path=True)

            incar.write_file(os.path.join(path, 'INCAR'))
            kpoints.write_file(os.path.join(path, 'KPOINTS'))
            poscar.write_file(os.path.join(path, 'POSCAR'))
            shutil.copy(os.path.join(os.environ['PBE'],system.lower(),'POTCAR'), os.path.join(path,'POTCAR'))
            if check_potpaw:
                potcar_spec = inputs['potcar_spec']
                logger.log("INFO",f"{potcar_spec}")


def run_relax(system, partition, out=None):
    if out is None:
        out = load_dict(os.path.join(os.environ['JAR'],f'{system}0.pkl'))
    for row in out.iterrows():
        row = row[1]
        path = os.path.join(os.environ['DFT'], system.lower(), row['bravais'])
        shutil.copy(os.path.join(os.environ['PBE'],system.lower(),'POTCAR'), os.path.join(path,'POTCAR'))
        subprocess.run(['chmod','-w','POTCAR'], cwd=path)
        subprocess.run(['chmod','+x','POTCAR'], cwd=path)
        vasp_job(system=system, path=path, partition=partition, return_path=False, run=True)

def strain_vol(system, system_path, x=0.157, num_points=15):
    incar = Incar.from_file(os.path.join(system_path,'INCAR'))
    incar['ISIF'] = 2
    if system in ['Cs','Rb']:
        incar['NBANDS'] = 320
        incar['PREC'] = 'Normal'

    kpoints = Kpoints.from_file(os.path.join(system_path,'KPOINTS'))
   
    atoms = read(os.path.join(system_path, 'CONTCAR'))
    volume_factors = np.linspace(1-x, 1+x, num_points)
    cell = atoms.get_cell() 
    a, b, c, alpha, beta, gamma =  cell.cellpar()
    if 'hcp' in system_path:
        logger.info('Detected hexagonal (hcp) structure. Applying anisotropic strain')
        ca_ratio = c/a 
        
        for i, factor in enumerate(volume_factors):
            strain_path = make_dir(os.path.join(system_path, 'strain', str(i)),return_path=True)
            new_vol = atoms.get_volume() * factor
            new_a = (new_vol/(np.sqrt(3) * ca_ratio /2))**(1/3)
            new_c = ca_ratio * new_a
            new_cell = np.array([[new_a, 0, 0], [-0.5*new_a, np.sqrt(3)/2 * new_a, 0], [0,0,new_c]])
            strained_atoms = copy.deepcopy(atoms)    
            strained_atoms.set_cell(new_cell, scale_atoms=True)
            
            incar.write_file(os.path.join(strain_path, 'INCAR'))
            kpoints.write_file(os.path.join(strain_path, 'KPOINTS'))
            strained_atoms.write(os.path.join(strain_path, 'POSCAR'))
            subprocess.run(['cp', f'{os.path.join(system_path,"POTCAR")}', f'{os.path.join(strain_path,"POTCAR")}'])
            subprocess.run(['chmod','-w','POTCAR'], cwd=strain_path)
            subprocess.run(['chmod','+x','POTCAR'], cwd=strain_path)
    elif 'fcc' in system_path or 'bcc' in system_path:
        logger.info('Detected cubic (fcc, bcc) structure. Applying isotropic strain')
        for i, factor in enumerate(volume_factors):
            strain_path = make_dir(os.path.join(system_path, 'strain', str(i)), return_path=True)
            strained_atoms = copy.deepcopy(atoms)
            new_cell = cell*factor**(1/3)
            strained_atoms.set_cell(new_cell, scale_atoms=True) 
            
            incar.write_file(os.path.join(strain_path, 'INCAR'))
            kpoints.write_file(os.path.join(strain_path, 'KPOINTS'))
            strained_atoms.write(os.path.join(strain_path, 'POSCAR')) 
            subprocess.run(['cp', f'{os.path.join(system_path,"POTCAR")}', f'{os.path.join(strain_path,"POTCAR")}'])
            subprocess.run(['chmod','-w','POTCAR'], cwd=strain_path)
            subprocess.run(['chmod','+x','POTCAR'], cwd=strain_path)

def run_eos(system,logger, out=None):
    if out is None:
        out = load_dict(os.path.join(os.environ['JAR'],f'{system}0.pkl')) 
    for i, row in out.iterrows():
        system_strain_path = make_dir(os.path.join(os.environ['DFT'], system.lower(), row['bravais'],'strain'), return_path=True)
        subprocess.run(['rm', 'run-eos.sh'], cwd=system_strain_path)
        job = os.path.join(os.environ ['DNJF'], 'jobs','run-eos.sh')
        subprocess.run(['cp', job, os.path.join(system_strain_path,'run-eos.sh')])
        system_path = os.path.join(os.environ['DFT'], system.lower(), row['bravais'])
        strain_vol(system=system, system_path=system_path) 
        subprocess.run(['sbatch', 'run-eos.sh'], cwd=system_strain_path)


def get_vasp_result(system, return_out=False):
    logger = get_logger(system, logfile=f'{system}.vasp.log',job='mlp')
    out = load_dict(os.path.join(os.environ['JAR'],f'{system}0.pkl'))
    volume_factors = np.linspace(0,15,15)
    nions_dft, pes_dft, vols_dft, forces_dft, stress_dft= [], [], [], [], []
    for _, row in out.iterrows():
        path = os.path.join(os.environ['DFT'], system.lower(), row['bravais'],'strain')
        pes, vols, forces, stress_s = [], [], [], []
        for i, factor in enumerate(volume_factors):
            strain_path = os.path.join(path, str(i))
            logger.log("INFO",f"strain_path : {strain_path}")
            vasprun = Vasprun(os.path.join(strain_path, 'vasprun.xml'))
            fin_structure=vasprun.final_structure
            ionic_step=vasprun.ionic_steps[-1]
            pe = ionic_step['e_fr_energy']
            vol = fin_structure.lattice.volume
            stress = np.asarray(ionic_step['stress'])
            force = np.asarray(ionic_step['stress'])
            nion = fin_structure.num_sites
            forces.append(force)
            pes.append(pe/nion)
            vols.append(vol/nion)
            stress_s.append(stress)
            
            logger.log("INFO",f"{i}th volume: {vol}") 
            logger.log("INFO",f"{i}th pe: {pe}")
            logger.log("INFO",f"{i}th stress: {stress}")
            logger.log("INFO",f"{i}th force: {force}")
        nions_dft.append(nion)
        vols_dft.append(np.asarray(vols))
        pes_dft.append(np.asarray(pes))
        forces_dft.append(np.asarray(forces))
        stress_dft.append(np.asarray(stress_s))
    out[f'pe-dft'] = pes_dft 
    out[f'vol-dft'] = vols_dft 
    out[f'force-dft'] = forces_dft 
    out[f'stress-dft'] = stress_dft 
    save_dict(out, os.path.join(os.environ['JAR'], f'{system}_mlp.pkl'))
    if return_out:
        return out
    return

if __name__ == '__main__':
    'winston'
