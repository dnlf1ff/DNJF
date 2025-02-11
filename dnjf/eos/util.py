from ase.io import read
import copy
from mp_api.client import MPRester
import numpy as np
import shutil
import subprocess
import os
import pandas as pd
import re
import yaml
import pickle
import torch


def get_device(return_device = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if return_device:
        return device
    return

def get_mpr():
    mpr = MPRester(api_key = os.environ['API_KEY'], use_document_model=False)
    return mpr

def set_env(task,pbe, prefix='.dlfjs'):
    os.environ['API_KEY'] = 'tUCZf2SGk3XSAc8Jpqb12c3Q8Ve8230O'
    os.environ['HOME'] = os.path.join('/home/jinvk', prefix)
    os.environ['DNJF'] = os.path.join(os.environ['HOME'],'DNJF')
    os.environ['BARK'] = os.path.join(os.environ['HOME'], 'BARK')
    os.environ['TASK'] =make_dir(os.path.join(os.environ['HOME'], task, str(pbe)),return_path=True)
    os.environ['DFT'] = make_dir(os.path.join(os.environ['TASK'], 'dft'), return_path=True)
    os.environ['OUT'] = make_dir(os.path.join(os.environ['TASK'], 'out'), return_path=True)
    os.environ['PLOT'] = make_dir(os.path.join(os.environ['OUT'], 'plot'), return_path=True)
    os.environ['LOG']=make_dir(os.path.join(os.environ['OUT'],'log'), return_path=True)
    os.environ['JAR']=make_dir(os.path.join(os.environ['OUT'],'jar'), return_path=True)
    os.environ['TRAJ']=make_dir(os.path.join(os.environ['OUT'],'traj'), return_path=True)

    os.environ['RUN'] = make_dir(os.path.join(os.environ['TASK'],'run'), return_path=True)
    os.environ['PBE'] = str(pbe)
    os.environ['POTPAW'] = os.path.join(os.environ['DNJF'],'potpaw',str(pbe))
    os.environ['MLP'] = os.path.join(os.environ['BARK'], 'mlp')
    os.environ['CONF'] = os.path.join(os.environ['DNJF'],'config',task)
    os.environ['JOB'] = os.path.join(os.environ['BARK'], 'jobs')

def make_dir(path, return_path = True):
    os.makedirs(path, exist_ok=True)
    if return_path:
        return path
    return
  
def load_conf():
    conf = os.path.join(os.environ['CONF'],'inp.yaml')
    with open(conf, 'r') as inp:
        inp_yaml = yaml.safe_load(inp)
        return inp_yaml

def check_py():
    print(f"Using Python executable: {sys.executable}")


def save_dict(data, path):
    path = os.path.join(os.environ['JAR'],f'{path}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_dict(path):
    path = os.path.join(os.environ['JAR'],f'{path}.pkl')
    with open(path, 'rb') as f:
        loaded_data = pickle.load(f)
    return loaded_data

def get_mlps(pickle_jar):
    mlps = []
    for column in pickle_jar.columns:
        if 'force' in column and column.split('-')[1] not in mlps:
            mlps.append(column.split('-')[1])
    return mlps

def get_tot_mlps():
    ag_out = load_dict('Ag_mlp')
    mlps_tot = get_mlps(ag_out)
    # print(f'total mlps .. {mlps_tot}')
    return mlps_tot

def get_after_school_mlps(system):
    mlps_tot = get_tot_mlps()
    sys_out = load_dict(f'{system}_mlp')
    sys_out = get_mlps(sys_out)
    sys_left = list(set(mlps_tot)^set(sys_out))
    # print(f'{system} needs afterschool calculation for {sys_left}')
    return sys_left

def get_systems_mlps(systems):
    todos = {}
    for system in systems:
        sys_left = get_after_school_mlps(system)
        todos[system] = sys_left
    return todos

def group_systems(systems):
    todos = get_systems_mlps(systems)
    neglected_mlps = []
    todo = {} 
    for system, mlps in todos.items():
        if mlps not in neglected_mlps:
            neglected_mlps.append(mlps)
    for i, mlps in enumerate(neglected_mlps):
        todo[chr(i+65)] = {}
        todo[chr(i+65)]['systems'] = list([])
        todo[chr(i+65)]['mlps'] = mlps
    for system, mlps in todos.items():
        for c in todo:
            if mlps == todo[c]['mlps']:
                todo[c]['systems'].append(system) 
    return todo

def tot_sys_mlps():
    mlps = ['chgTot','chgTot_l3i3','chgTot_l3i5','chgTot_l4i3','omat_epoch1','omat_epoch2','omat_epoch4','omat_ft_r5','r5pp','omat_i5pp_epoch1','omat_i5pp_epoch2','omat_i5pp_epoch3','omat_i5pp_epoch4','omat_i5_epoch1','omat_i5_epoch2','omat_i5_epoch3','omat_i5_epoch4','omat_i3pp'] 
    systems = ['Ag','Al','Au','Ca','Cd','Co','Cs','Cu','Fe','Hf','K','Li','Mg','Mo','Nb','Na','Os','Pd','Pt','Rb','Re','Rh','Sr','Ta','Ti','V','W','Zn','Zr']
    return mlps, systems
