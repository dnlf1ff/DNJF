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
from util import *
from log import *
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

def set_env(task,pbe):
    os.environ['API_KEY'] = 'tUCZf2SGk3XSAc8Jpqb12c3Q8Ve8230O'
    os.environ['HOME'] = '/home/jinvk'
    os.environ['DNJF'] = os.path.join(os.environ['HOME'],'DNJF')
    os.environp['BARK'] = os.path.join(os.environ['HOME'], 'BARK')
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

def make_dir(path, return_path = False):
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


def save_dict(data, path, return_dict=False):
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    if return_dict:
        return data
    return

def load_dict(path):
    with open(path, 'rb') as f:
        loaded_data = pickle.load(f)
    return loaded_data

def debug_input(input_data):
    print("Type:", type(input_data))
    if isinstance(input_data, (list, np.ndarray)):
        print("Contents:", input_data)
    elif isinstance(input_data, str):
        print("String Contents:", input_data)
    else:
        print("Unexpected data type:", input_data)
        
def sanitize_input(input_data):
    if isinstance(input_data, np.ndarray):  # NumPy array
        return input_data.tolist()
    elif isinstance(input_data, str):       # String representation
        cleaned = re.sub(r"[^\d\.\-\s]", "", input_data)
        cleaned = re.sub(r"\s+", ",", cleaned.strip())
        return [float(num) for num in cleaned.split(",")]
    else:
        raise ValueError("Unsupported input format!")

