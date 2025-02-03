import os
import shutil
import subprocess

from util import load_conf, load_dict, save_dict, set_env 
from log import *
from vasp import *
from shell import python_job, gpu3_job, python_jobs, job_with_node

def run_vasp(systems, pbe): # native python or create job
    set_env('eos',pbe=pbe)
    for system in systems:
        inp = load_conf(os.path.join(os.environ['PRESET'],'eos','inp.yaml'))
        out = write_output(system, inp,return_out=True) 
        write_inputs(system, out=out)
        run_relax(system,partition='loki2')

def run_vasp_eos(systems, pbe): # automized
    set_env(task='eos',pbe=pbe)
    for system in systems:
        run_eos(system) 

def post_vasp(systems, pbe): # natvie python or create job
    set_env(task='eos',pbe=pbe)
    for system in systems:
        # job_with_node(system=system, task='eos', script='vasp.py',argv_=system, partition='gpul',nodelist='n137',run = True)
        python_job(system=system,task='eos',script='mlp.py',argv_=system, partition='loki1',run=True)

def run_fit(systems, pbe):
    set_env(task='eos', pbe=pbe)
    for system in systems:
        python_job(system=system,task='eos',script='fit.py',argv_=system,  partition='loki1', run=True)



if __name__ == '__main__':
    systems = ['Au','Ag','Al','Ca','Co','Cu','Fe','Hf','Ir','K','Li','Mg','Mo','Na','Nb','Ni','Os','Pd','Pt','Re','Rb','Rh','Sr','Ti','Ta','V','W','Zn','Zr','Cs'] #Y, Cd
    pbe = sys.argv[1]
    post_vasp(systems, pbe)
    # run_fit(systems, pbe)
