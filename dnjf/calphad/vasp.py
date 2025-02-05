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
import sys
from util import *
from shell import *


def concat_potcar(path):
    subprocess.run(['cp', os.path.join(os.environ['POTPAW'],'cu','POTCAR'), os.path.join(path,'POTCAR.cu')])
    subprocess.run(['cp', os.path.join(os.environ['POTPAW'],'au','POTCAR'), os.path.join(path,'POTCAR.au')])
    with open(os.path.join(path, 'POTCAR'), 'wb') as potcar:
        for fname in [os.path.join(path, 'POTCAR.cu'), os.path.join(path, 'POTCAR.au')]:
            with open(fname, 'rb') as infile:
                potcar.write(infile.read())

def write_inputs(system, logger=logger):
    inp = load_conf()
    mpr=MPRester(api_key=os.environ['API_KEY'],use_document_model=False)
    result = mpr.materials.tasks.search(inp[system]['mp_id'])[0]
    inputs = result['input']
    orig_inputs = result['orig_inputs']
    path = make_dir(os.path.join(os.environ['DFT'], system.lower()), return_path=True)
    if len(system) < 3: 
        subprocess.run(['cp', os.path.join(os.environ['POTPAW'],system.lower(),'POTCAR'), os.path.join(path,'POTCAR')])
    else:
        concat_potcar(path)
    
    subprocess.run(['chmod','-w','POTCAR'], cwd=path)
    subprocess.run(['chmod','+x','POTCAR'], cwd=path)
    
    incar=Incar(inputs['incar'])
    kpoints = Kpoints.from_dict(orig_inputs['kpoints'])
    poscar = Poscar(Structure.from_dict(inputs['structure']))
    incar = mod_incar(incar, system,path=path)    
    kpoints.write_file(os.path.join(path, 'KPOINTS'))
    poscar.write_file(os.path.join(path, 'POSCAR'))

def vasp_relax(system, partition):
    path = os.path.join(os.environ['DFT'], system.lower())
    vasp_job(system=system, path=path, partition=partition, return_path=False, run=False)


def write_out(binary_system, systems, return_out = False):
    out = pd.DataFrame()
    out['system'] = [e.lower() for e in systems]
    out['n_cu'] = [1, 0, 2, 3, 1]
    out['n_au'] = [0, 1, 2, 1, 3]
    out['n_ion'] = [1, 1, 4, 4, 4]
    out['s_id'] = [0] * 5
    save_dict(out, os.path.join(os.environ['JAR'],f'{binary_system.lower()}.pkl'))
    if return_out:
        return out
    return

def get_vasp_results(binary_system, systems, out=None, return_out=False):
    if out is None:
        out = load_dict(os.path.join(os.environ['JAR'],f'{binary_system.lower()}.pkl'))
    fes = []
    for system in systems:
        path = os.path.join(os.environ['DFT'],system.lower())
        logger.log("INFO",f"path : {path}")
        o = Outcar(os.path.join(strain_path, "OUTCAR"))
        a = read(os.path.join(strain_path,'POSCAR'))
        nions.append(len(a))
        fes.append(o.final_fr_energy)
        logger.log("INFO",f"{i}th pe: {o.final_fr_energy}")
        logger.log("INFO",f"appended pes: {pes}")
    
    out['fe-dft'] = fes
    save_dict(out, path=os.path.join(os.environ['JAR'],f'{binary_system.lower()}.pkl'))
    if return_out:
        return out
    return

def mod_incar(incar_pre, system, path,isif=3, return_incar=False):
    incar = incar_pre
    if isif == 2:
        incar = Incar.from_file(incar_pre)
        incar['ADDGRID'] = '.TRUE.'
    incar['ISIF'] = isif 
    incar['LREAL'] = '.FALSE.'
    incar['LWAVE'] = '.FALSE.'
    incar['system'] = f'{system} opt' if isif == 3 else f'{system}.dis'
    incar['IBRION'] = 2 if isif == 3 else -1 
    incar['ISPIN'] = 1
    incar['KPAR'] = 4
    if isif==2 and "EDIFFG" in incar:
        del incar["EDIFFG"]
    if isif == 3:
        del incar["MAGMOM"]
    incar.write_file(os.path.join(path, 'INCAR'))
    if return_incar:
        return incar
    return

def propagate_vasp(path_in, path_out, system, poscar=None, incar=None, partition='jinvk', run=True):
    kpoints = os.path.join(path_in, 'KPOINTS')
    if poscar is None:
        poscar = os.path.join(path_in, 'CONTCAR')
    potcar = os.path.join(path_in, 'POTCAR')
    subprocess.run(['cp', kpoints, os.path.join(path_out, 'KPOINTS')])
    subprocess.run(['cp', poscar, os.path.join(path_out, 'POSCAR')])
    subprocess.run(['cp', potcar, os.path.join(path_out, 'POTCAR')])
    if incar is not None:
        incar = os.path.join(path, incar)
        subprocess.run(['cp', incar, os.path.join(path_out, 'INCAR')])
    if run:
        vasp_job(system, path=path_out, partition=partition, run=run)


def rabbit_phonons(system, phonon_path, partition, run=True):
    os.chdir(phonon_path)
    poscar_list = []
    for f in os.listdir():
        if 'POSCAR-' in f and f is not os.path.isdir(d):
            poscar_list.append(d)
    for i in range(1, len(poscar_list)+1):
        disp_dir = f"disp-{i}"
        propagate_vasp(path_in =phonon_path, path_out = os.path.join(phonon_path, f'disp_{i}'), system=system, poscar = poscar_list[i-1], incar='INCAR',partition=partition, run=False)
        os.chdir(phonon_path)
        poscars = make_dir(os.path.join(phonon_path, 'poscars'), return_path=True)
        subprocess.run(['mv', poscar_list[-1], poscars])


def baby_phonons(row, return_path=True):
    path = os.path.join(os.environ['DFT'],row['system'])
    phonon_path = make_dir(os.path.join(os.environ['DFT'], row['system'], 'phonon'), return_path=True)
    mod_incar(os.path.join(path, 'INCAR'), system=row['system'], path=phonon_path, isif=2)
    propagate_vasp(path_in = path, path_out = phonon_path, system=row['system'], run=False)
    
    os.chdir(phonon_path)
    subprocess.run(['phono3py', '-d', '--dim', '2 2 2', '-c', 'POSCAR', '-pa','auto'])
    if return_path:
        return phonon_path
    return

def run_phonon(binary_system, out=None):
    if out is None:
        out = load_dict(os.path.join(os.environ['JAR'], f'{binary_system.lower()}.pkl'))
    for i, row in out.iterrows():
        phonon_path = baby_phonons(row)
        rabit_phonons(row['system'], phonon_path,partition='jinvk',run=False) 


if __name__ == '__main__':
    'asdf'

