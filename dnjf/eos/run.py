import os
import shutil
import subprocess
import gc
from util import load_conf, load_dict, save_dict, set_env 
from vasp import write_inputs, run_relax, run_eos 
from shell import gpu3_job, job_with_node, gpu3_jobs, jobs_with_node, inter_env_jobs
from log import get_logger
import sys

def run_vasp(systems, pbe): # native python or create job
    set_env('eos',pbe=pbe)
    for system in systems:
        logger = get_logger(system, f'{system}.relax.log', job='dft')
        inp = load_conf(os.path.join(os.environ['PRESET'],'eos','inp.yaml'))
        out = write_out(system, inp,return_out=True) 
        write_inputs(system, logger=logger, out=out)
        run_relax(system,partition='loki2')
        del logger, out
        gc.collect()
        

def run_vasp_eos(systems, pbe): # automized
    set_env(task='eos',pbe=pbe)
    for system in systems:
        logger = get_logger(system, f'{system}.eos.log', job='dft')
        run_eos(system, logger=logger)

def post_vasp(mlp, pbe): # natvie python or create job
    set_env(task='eos',pbe=pbe)
    job_with_node(argv_=mlp,task='eos', script='mlp.py', job_name='post2', partition='gpu',nodelist='n008',run=True)

def run_fit(systems, pbe):
    set_env(task='eos', pbe=pbe)
    jobs_with_node(argv_=systems,task='eos',script_1='fit.py', script_2 = 'beat.py',  partition='gpu', nodelist='n008',job_name='post', run=True)

if __name__ == '__main__':
    mlps= ['chgTot','chgTot_l3i3','chgTot_l3i5','chgTot_l4i3','m3g_n','m3g_r6','m3g_r55','omat_epoch1','omat_epoch2','omat_epoch3','omat_epoch4','omat_ft_r5','r5pp','omat_i5pp_epoch1','omat_i5pp_epoch2','omat_i5pp_epoch3','omat_i5pp_epoch4','omat_i5_epoch1','omat_i5_epoch2','omat_i5_epoch3','omat_i5_epoch4','omat_i3pp','ompa_i5pp_epoch1','ompa_i5pp_epoch2','ompa_i5pp_epoch3','ompa_i5pp_epoch4']
    pbe = sys.argv[1]
    post_vasp(mlp=mlps[2], pbe=pbe)
