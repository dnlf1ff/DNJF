import os
import shutil
import subprocess

from util import set_env, get_systems 
from vasp import * 
from shell import gpu3_job, job_with_node

def run_vasp(binary_system, pbe): # native python or create job
    set_env(task='calphad',pbe=pbe)
    write_out(binary_system, systems)
    for system in systems:
        logger = get_logger(system, logfile=f'{system}.vasp.x', job='vasp')
        write_inputs(system)
        vasp_relax(system,partition='loki2')
        
def run_vasp_phonon(binary_system, pbe):
    set_env(task='calphad', pbe=pbe)
    vasp_phonon(binary_system)

if __name__ == '__main__':
    binary_system = 'Cu-Au'
    pbe = sys.argv[1]
    run_vasp_vaso(binary_system)
