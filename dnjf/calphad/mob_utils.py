import yaml
import sys
import pickle
import os
from mp_api.client import MPRester
import torch

def get_device(return_device = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if return_device:
        return device
    return

def get_mpr():
    mpr = MPRester(api_key = os.environ['API_KEY'], use_document_model=False)
    return mpr

def set_env(task):
    os.environ['DFT'] = os.path.join(os.environ['HOME'], f'Research../{task}/dft')
    make_dir(os.environ['DFT'])
    os.environ['PLOT'] = os.path.join(os.environ['HOME'], f'Research../{task}/plot')
    make_dir(os.environ['PLOT'])
    os.environ['OUT'] = os.path.join(os.environ['HOME'], f'Research../{task}/output')
    make_dir(os.environ['OUT'])
    os.environ['CALC'] = os.path.join(os.environ['HOME'], f'Research../{task}/calc')
    make_dir(os.environ['CALC'])
    os.environ['API_KEY'] = 'tUCZf2SGk3XSAc8Jpqb12c3Q8Ve8230O'
    os.environ['DNJF'] = os.path.join(os.environ['HOME'], 'DNJF')
    os.environ['POTCAR_LEGACY'] = '/TGM/Apps/VASP/POTCAR/POTCAR_LEGACY'
    os.environ['MLP'] = os.path.join(os.environ['DNJF'], 'mlp')
    os.environ['TRAJ'] = os.path.join(os.environ['CALC'], 'traj')
    os.environ['LOG'] = os.path.join(os.environ['CALC'], 'log')
    os.environ['PRESET'] = os.path.join(os.environ['DNJF'],'presets')
    make_dir(os.environ['LOG'])
    make_dir(os.environ['TRAJ'])
    os.environ['JOBS'] = os.path.join(os.environ['DNJF'], 'jobs')
def make_dir(path, return_path = False):
    os.makedirs(path, exist_ok=True)
    if return_path:
        return path
    return
  
def load_conf(conf_yaml):
    with open(conf_yaml, 'r') as inp:
        inp_yaml = yaml.safe_load(inp)
        return inp_yaml

def check_py():
    print(f"Using Python executable: {sys.executable}")

def is_df(task, path):
    df_path = os.path.join('..',task, path)
    return os.path.isfile(df_path)

def save_dict(data, task, path, file=None, return_dict=False):
    if file is None:
        file = 'data.pkl'
    pkl_path = os.path.join('..', task, path, file)
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)
    if return_dict:
        return data
    return

def load_dict(task, path, file):
    if file in None:
        file = 'data,pkl'
    pkl_path = os.path.join('..', task, path, file)
    with open(pkl_path, 'rb') as f:
        loaded_data = pickle.load(f)
    return loaded_data

