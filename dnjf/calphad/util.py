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
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_dict(path):
    with open(path, 'rb') as f:
        loaded_data = pickle.load(f)
    return loaded_data

