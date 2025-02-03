import os
import shutil
import subprocess

from util import load_conf, load_dict, save_dict 
from log import *
from vasp import *
from shell import python_job, gpu3_job, python_jobs, job_with_node

def run_vasp(binary_system, systems, pbe): # native python or create job
    set_env(task='calphad',pbe=pbe)
    write_out(binary_system, systems)
    for system in systems:
        logger = get_logger(system, logfile=f'{system}.vasp.x', job='vasp')
        write_inputs(system)
        run_relax(system,partition='loki2')

if __name__ == '__main__':
    binary_system = 'Cu-Au'
    pbe = 52
    systems = ['Cu','Au','CuAu','Cu3Au','CuAu3']
    run_vasp(binary_system, systems, pbe)
