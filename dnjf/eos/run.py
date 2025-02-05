import os
import shutil
import subprocess

from util import load_conf, load_dict, save_dict, set_env 
from log import *
from vasp import *
from shell import gpu3_job, job_with_node, gpu3_jobs, jobs_with_node, inter_env_jobs

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
    #inter_env_jobs(argv_=systems,argv_s=['mace', 'matsim'],task='eos', script='mlp.py', job_name='bench', partition='gpu',nodelist='n007',run=True)
    job_with_node(argv_=systems,task='eos', script='mlp.py', job_name='tlsdltldu', partition='gpu',nodelist='n008',run=True)
    # jobs_with_node(argv_=systems,task='eos',script_1='vasp.py', script_2='mlp.py', job_name='eos.ns',partition='gpul', nodelist='n137',run=True)

def run_fit(systems, pbe):
    set_env(task='eos', pbe=pbe)
    jobs_with_node(argv_=systems,task='eos',script_1='fit.py', script_2 = 'beat.py',  partition='gpu', nodelist='n008',job_name='post', run=True)



if __name__ == '__main__':
    # systems='Al Ag Au Ca Co Cd Cs Cu Fe Hf Ir K Li Mg Mo Na Nb Os Pd Pt Re Rb Rh Sr Ti Ta V W Zn Zr'
    systems='Cd Cs Cu Fe Hf Ir K Li Mg Mo Na Nb Re Rb Rh Sr V W Zn Zr'
    pbe = sys.argv[1]
    post_vasp(systems, pbe)
    # run_fit(systems, pbe)
