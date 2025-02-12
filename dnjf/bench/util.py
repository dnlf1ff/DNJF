from ase.io import read
import copy
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

def set_env(task, prefix='.dlfjs'):
    os.environ['HOME'] = os.path.join('/home/jinvk',prefix)
    os.environ['DNJF'] = os.path.join(os.environ['HOME'],'DNJF')
    os.environ['BARK'] = os.path.join(os.environ['HOME'], 'BARK')
    os.environ['TASK'] =make_dir(os.path.join(os.environ['HOME'], task),return_path=True)
    os.environ['OUT'] = make_dir(os.path.join(os.environ['TASK'], 'out'), return_path=True)
    os.environ['PLOT'] = make_dir(os.path.join(os.environ['OUT'], 'plot'), return_path=True)
    os.environ['LOG']=make_dir(os.path.join(os.environ['OUT'],'log'), return_path=True)
    os.environ['JAR']=make_dir(os.path.join(os.environ['OUT'],'jar'), return_path=True)
    os.environ['TRAJ']=make_dir(os.path.join(os.environ['OUT'],'traj'), return_path=True)

    os.environ['RUN'] = make_dir(os.path.join(os.environ['TASK'],'run'), return_path=True)
    os.environ['MLP'] = os.path.join(os.environ['BARK'], 'mlp')
    os.environ['CONF'] = os.path.join(os.environ['BARK'],'TM23')
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
    path = os.path.join(os.environ['JAR'], f'{path}.pkl')
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

def get_after_school_mlps(system):
    ag_out = load_dict('Ag')
    mlps_tot = get_mlps(ag_out)
    sys_out = load_dict(system)
    sys_mlps = get_mlps(sys_out)

    sys_left = list(set(mlps_tot)^set(sys_mlps))
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
            for c  in todo:
                if mlps == todo[c]['mlps']:
                    todo[c]['systems'].append(system)
    return todo

def tot_sys_mlps():
    mlps =  ['chgTot','chgTot_l3i3','chgTot_l3i5','chgTot_l4i3','m3g_n','m3g_r6','m3g_r55','omat_epoch1','omat_epoch2','omat_epoch3','omat_epoch4','omat_ft_r5','r5pp','omat_i5pp_epoch1','omat_i5pp_epoch2','omat_i5pp_epoch3','omat_i5pp_epoch4','omat_i5_epoch1','omat_i5_epoch2','omat_i5_epoch3','omat_i5_epoch4','omat_i3pp']
    systems = ['Ag','Au','Cd','Co','Cr','Cu','Fe','Hf','Hg','Ir','Mn','Mo','Nb','Ni','Os','Pd','Pt','Re','Rh','Ru','Ta','Tc','Ti','V','W','Zn','Zr']
    return systems, mlps


















