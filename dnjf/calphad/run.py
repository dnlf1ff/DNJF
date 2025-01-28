import os
import shutil
import subprocess

from shell_utils import *
# from mlp_utils import *

def run_(argv_1, argv_2):
    
    job_file = create_sbatch2('fit1', script='../eos_utils.py', argv_1=argv_1, argv_2=argv_2, partition='loki4')
    os.chdir('run')
    subprocess.run(['sbatch',f'{job_file}'])
    os.chdir('..')

if __name__ == '__main__':
    # systems = ['Ag','Au','Co','Cu','Cs','Cr']
    # systems = ['Ni','Na','Mo','K','Li','Fe']
    # systems = ['V','W','Zn','Rb','Pd','Pt','Cd']
    run_(argv_1=0, argv_2=2)
