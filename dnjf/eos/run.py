import os
import shutil
import subprocess

from util import load_conf, load_dict, save_dict 
from log import *
from vasp import *
from shell import python_job, gpu3_job, python_jobs, job_with_node

def run_vasp(systems): # native python or create job
    set_env('eos',pbe=52)
    for system in systems:
        logger = get_logger(system, logfile=f'{system}.vasp.x', job='vasp')
        inp = load_conf(os.path.join(os.environ['PRESET'],'eos','inp.yaml'))
        out = write_output(system, inp,return_out=True) 
        write_inputs(system, out=out)
        run_relax(system,partition='loki2')

def run_vasp_eos(systems): # automized
    set_env('eos',pbe=52)
    logger = get_logger('eos1','log','vasp')
    for system in systems:
        run_eos(system) 

def post_vasp(systems): # natvie python or create job
    set_env('eos',pbe=52)
    for system in systems:
        logger = get_logger(system, logfile=f'{system}.post.log',job='mlp')
        # job_with_node(system=system, task='eos', script='vasp.py',argv_=system, partition='gpul',nodelist='n137',run = True)
        python_job(system=system,task='eos',script='mlp.py',argv_=system, partition='loki4', run=True)

def run_fit(system):
    job = python_job(system=system,task='eos',script='fit.py',argv_=system, partition='gpu2', run=True)



if __name__ == '__main__':
#    systems=['Ca','Ni','Ti','Zn','Zr']
    systems = ['Ca']
    # systems = ['Zr','Ti','Zn','Ni']
    post_vasp(systems)
