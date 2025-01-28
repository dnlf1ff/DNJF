import os
import shutil
import subprocess

from util import load_conf, load_dict, save_dict 
from log import *
from vasp import *
from shell import python_job, gpu3_job

def run_vasp(systems): # native python or create job
    set_env('eos',pbe=52)
    for system in systems:
        logger = get_logger(system, logfile=f'{system}.vasp.x', job='vasp')
        inp = load_conf(os.path.join(os.environ['PRESET'],'eos','inp.yaml'))
        out = write_output(system, inp) 
        write_inputs(system, out)
        save_dict(out, os.path.join(os.environ['JAR'], f'{system}0.pkl'))
        run_relax(system,partition='loki3')

def run_vasp_eos(systems, out=None): # automized
    set_env('eos',pbe=52)
    logger = get_logger('eos1','log','vasp')
    for system in systems:
        if out is None:
            out = load_dict(os.path.join(os.environ['JAR'],f'{system}0.pkl'))
        run_eos(system) 

def post_vasp(systems): # natvie python or create job
    set_env('eos',pbe=52)
    for system in systems:
        logger = get_logger(system, logfile=f'{system}.post.log',job='mlp')
        python_jobs(system=system,task='eos',script_1='vasp.py',script_2 = 'mlp.py',argv_=system, partition='gpu2', run=True)

def run_fit(system):
    job = python_job(system=system,task='eos',script='fit.py',argv_=system, partition='gpul', run=True)



if __name__ == '__main__':
#    systems = ['Hf','Ir', 'Mg','Nb','Os','Re','Ta','Y']
    # systems = ['Ag','Au','Al','Co','Cu','Cs','Cd']
    # post_vasp(systems)
 #  run_vasp(['Cd'])
    #write_inputs('Ag')
    systems = ['Cd']
    run_vasp_eos(systems)
# systems=['Cd','Ag']
# post_vasp(systems)
