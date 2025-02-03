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


def concat_potcar(system, path):
    if system in ['CuAu','CuAu3','Cu3Au']:
        subprocess.run(['cp', os.path.join(os.environ['PBE'],'cu','POTCAR'), os.path.join(path,'POTCAR.cu')])
        subprocess.run(['cp', os.path.join(os.environ['PBE'],'au','POTCAR'), os.path.join(path,'POTCAR.au')])
        subprocess.run(['cat',os.path.join(path, 'POTCAR.cu'), os.path.join(path, 'POTCAR.au'),'>','POTCAR' ],cwd=path)


def write_inputs(system, logger=logger):
    inp = load_conf()
    mpr=MPRester(api_key=os.environ['API_KEY'],use_document_model=False)
    result = mpr.materials.tasks.search(inp[system]['mp_id'])[0]
    inputs = result['input']
    orig_inputs = result['orig_inputs']
    path = make_dir(os.path.join(os.environ['DFT'], system.lower()), return_path=True)
    try:
        subprocess.run(['cp', os.path.join(os.environ['PBE'],system.lower(),'POTCAR'), os.path.join(path,'POTCAR')])
    except:
        concat_potcar(system, path)
    
    subprocess.run(['chmod','-w','POTCAR'], cwd=path)
    subprocess.run(['chmod','+x','POTCAR'], cwd=path)
    
    incar=Incar(inputs['incar'])
    kpoints = Kpoints.from_dict(orig_inputs['kpoints'])
    poscar = Poscar(Structure.from_dict(inputs['structure']))
    
    incar.write_file(os.path.join(path, 'INCAR'))
    kpoints.write_file(os.path.join(path, 'KPOINTS'))
    poscar.write_file(os.path.join(path, 'POSCAR'))

def run_relax(system, partition):
    path = os.path.join(os.environ['DFT'], system.lower())
    vasp_job(system=system, path=path, partition=partition, return_path=False, run=True)


def write_out(binary_system, systems, return_out = False):
    df = pd.DataFrame()
    df['system'] = [e.lower() for e in systems]
    df['n_cu'] = [1, 0, 2, 3, 1]
    df['n_au'] = [0, 1, 2, 1, 3]
    df['n_ion'] = [1, 1, 4, 4, 4]
    df['s_id'] = [0] * 5
    save_dict(df, os.path.join(os.environ['JAR'],f'{binary_system.lower()}.pkl'))
    if return_out:
        return df
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

if __name__ == '__main__':
    'asdf'

