import os
import shutil
import subprocess

from util import load_conf, load_dict, save_dict, set_env 
from log import *
from vasp import vasp_relax 
from shell import python_job, gpu3_job, python_jobs, job_with_node

def run_vasp(binary_system, systems, pbe): # native python or create job
    set_env(task='calphad',pbe=pbe)
    write_out(binary_system, systems)
    for system in systems:
        logger = get_logger(system, logfile=f'{system}.vasp.x', job='vasp')
        write_inputs(system)
        run_relax(system,partition='loki2')

def post_vasp(binary_system, pbe):
    set_env(task='calphad',pbe=pbe)
    python_job(system=binary_system, task='calphad', partition = 'loki2', script='mlp.py',argv_=binary_system, run=True)

if __name__ == '__main__':
    binary_system = 'Cu-Au'
    pbe = sys.argv[1]
    systems = ['Cu','Au','CuAu','Cu3Au','CuAu3']
    # run_vasp(binary_system, systems, pbe)
    post_vasp(binary_system, pbe)
