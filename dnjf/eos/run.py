import os
import shutil
import subprocess

from util import load_conf, load_dict, save_dict, set_env 
from log import *
from vasp import *
from shell import gpu3_job, job_with_node, gpu3_jobs, jobs_with_node

def run_vasp(systems, pbe): # native python or create job
    set_env('eos',pbe=pbe)
    for system in systems:
        inp = load_conf(os.path.join(os.environ['PRESET'],'eos','inp.yaml'))
        out = write_outs(system, inp,return_out=True) 
        write_inputs(system, out=out)
        run_relax(system,partition='loki2')

def run_vasp_eos(systems, pbe): # automized
    set_env(task='eos',pbe=pbe)
    for system in systems:
        run_eos(system) 

def post_vasp(systems, pbe): # natvie python or create job
    set_env(task='eos',pbe=pbe)
    jobs_with_node(argv_=systems,task='eos',script_1='vasp.py', script_2='mlp.py', job_name='eos.54',partition='gpu',nodelist='n008', run=True)
    # jobs_with_node(argv_=systems,task='eos',script_1='vasp.py', script_2='mlp.py', job_name='eos.ns',partition='gpul', nodelist='n137',run=True)

def run_fit(systems, pbe):
    set_env(task='eos', pbe=pbe)
    for system in systems:
        job_with_node(system=system,task='eos',script='fit.py',argv_=system,  partition='gpul', nodelist='n137',run=True)



if __name__ == '__main__':
    # systems = 'Ca Co Cu Cd Ir Mo Mg Li K'
    # systems = 'Cs Na Nb Ni Os Pd Pt Re Rb Rh Sr'
    # systems = 'Ti Ta V W Zn Zr Y'
    systems = 'Au Ag Al Ca Co Cu Fe Hf Ir K Li Mg Mo Na Nb Ni Os Pd Pt Re Rb Rh Sr Ti Ta V W Zn Zr Cs Cd Y' 
    pbe = sys.argv[1]
    post_vasp(systems, pbe)
