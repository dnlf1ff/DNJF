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

def get_system_df(df, system=None, mp_id=None):
    try:
        mask = df['formula_pretty'].apply(lambda x: system in x)
    except:
        if mp_id is None:
            mask = df['system'].apply(lambda x: system in x)
        else:
            mask = df['mp_id'].apply(lambda x: x == mp_id)
    return df[mask]


def write_output(system, inp):
    mp_ids=inp[system]['mp_id']
    bravais=inp[system]['bravais']
    df=pd.DataFrame()
    df['system']=[system]*len(mp_ids)
    df['mp_id']=mp_ids
    df['bravais']=bravais
    save_dict(df, os.path.join(os.environ['JAR'],f'{system}0.pkl'))
    return df



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

