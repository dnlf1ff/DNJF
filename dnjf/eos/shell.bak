import os
import subprocess
import shutil
import sys

def get_ntasks(partition):
    if partition=='loki4':
        return 28
    elif partition=='loki3':
        return 24
    elif partition=='loki2':
        return 20
    elif partition=='loki1':
        return 16
    elif partition=='csc2':
        return 32
    else:
        return 1

def get_gpu(partition):
    if 'gpu' in partition:
        return 'export CUDA_DEVICE_ORDER=PCI_BUS_ID\nexport CUDA_VISIBLE_DEVICES=1'
    else:
        return 'echo "GPU DEVICE not allotcated"' 


def gpu3_jobs(argv_,task, script_1, script_2,  partition='gpu3', job_name=None, return_path=False, run=False):
    if job_name is None:
        job_name = task
    ntasks=get_ntasks(partition)
    script_1 = os.path.join(os.environ['DNJF'],'dnjf',task, script_1) 
    script_2 = os.path.join(os.environ['DNJF'],'dnjf',task, script_2) 
    
    sbatch_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=calc.%j.x
#SBATCH --error=calc.%j.x
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks={ntasks}
#SBATCH --partition={partition}
#SBATCH --gres=gpu:1

echo "SLRUM_NTASKS: $SLURM_NTASKS"
if [ -z "$SLURM_NTASKS" ] || [ "$SLURM_NTASKS" -le 0 ]; then
    echo "Error: SLRUM_NTASKS is not set or less than or equal to 0"
    exit 1
fi

echo "running {script_1} for {argv_} ..."

python {script_1} {argv_}

echo "{script_1} for {argv_} completed"

echo "now running {script_2} for {argv_} ..."

python {script_2} {argv_}

echo "{script_2} for {argv_} done"

"""
    path = os.environ['RUN']
    job_name = f"{job_name}.sh"
    with open(os.path.join(path, job_name), "w") as sbatch_file:
        sbatch_file.write(sbatch_content)
    print(f"SBATCH file {job_name} created successfully!")
    
    if run:
        os.chdir(path)
        subprocess.run(['sbatch',job_name]) 
    if return_path:
        return path
    return 


def gpu3_job(argv_,task, script, job_name=None, partition='gpu3', return_path=False, run=False):
    if job_name is None:
        job_name=f'{task}.{os.environ["PBE"]}'
    else:
        job_name=f'{job_name}.{os.environ["PBE"]}'

    ntasks=get_ntasks(partition)
    script = os.path.join(os.environ['DNJF'],'dnjf',task, script) 
    
    sbatch_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=calc.%j.x
#SBATCH --error=calc.%j.x
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks={ntasks}
#SBATCH --partition={partition}
#SBATCH --gres=gpu:1

echo "SLRUM_NTASKS: $SLURM_NTASKS"
if [ -z "$SLURM_NTASKS" ] || [ "$SLURM_NTASKS" -le 0 ]; then
    echo "Error: SLRUM_NTASKS is not set or less than or equal to 0"
    exit 1
fi

python {script} {argv_}
"""
    path = os.environ['RUN']
    job_name = f"{job_name}.sh"
    with open(os.path.join(path, job_name), "w") as sbatch_file:
        sbatch_file.write(sbatch_content)
    print(f"SBATCH file {job_name} created successfully!")
    
    if run:
        os.chdir(path)
        subprocess.run(['sbatch',job_name]) 
    if return_path:
        return path
    return 

#TODO: listify argv_s

def listify(argv_, script, argv_s):
    nenvs = len(argv_s)
    pythons = [os.path.join('/home/jinvk/.venv',argv,'bin','python') for argv in argv_s]
    job_sh = f'\n
echo "running {script} for {argv_} ..."\n\n

python {script} {argv_}\n\n
echo "{script} for svn potentials done"\n\n

echo "setting python path to -{pythons[0]}"\n\n

{pythons[0]} {script} {argv_} {argv_s[0]}\n\n

echo "{script} for {argv_} completed"\n\n

echo "now running {script} for {argv_} ..."\n\n

echo "...and setting python path to - {pythons[1]}"\n\n

{pythons[1]} {script} {argv_} {argv_s[1]}\n\n

echo "{script} for {argv_} done"\n\n'
    return job_sh

def inter_env_jobs(argv_, argv_s, task, script, partition, nodelist, job_name=None ,output_file=None, return_path=False, run=False):
    if job_name is None:
        job_name = f'{task}.{os.environ["PBE"]}'
    else:
        job_name = f'{job_name}.{os.environ["PBE"]}'
    script = os.path.join(os.environ['DNJF'],'dnjf',task,script) 
    ntasks=get_ntasks(partition)
    job_sh = listify(argv_, script, argv_s) 
    sbatch_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=calc.%j.x
#SBATCH --error=calc.%j.x
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks={ntasks}
#SBATCH --partition={partition}
#SBATCH --nodelist={nodelist}

echo "SLRUM_NTASKS: $SLURM_NTASKS"
if [ -z "$SLURM_NTASKS" ] || [ "$SLURM_NTASKS" -le 0 ]; then
    echo "Error: SLRUM_NTASKS is not set or less than or equal to 0"
    exit 1
fi

{job_sh}

"""
    path = os.environ['RUN']
    job_name = f'{job_name}.sh'
    with open(os.path.join(path, job_name), "w") as sbatch_file:
        sbatch_file.write(sbatch_content)
    print(f"SBATCH file {job_name} created successfully!")
    
    if run:
        os.chdir(path)
        subprocess.run(['sbatch',job_name])

    if return_path:
        return path
    return


def jobs_with_node(argv_, task, script_1, script_2,  partition, nodelist, job_name=None ,output_file=None, return_path=False, run=False):
    if job_name is None:
        job_name = f'{task}.{os.environ["PBE"]}'
    else:
        job_name = f'{job_name}.{os.environ["PBE"]}'

    script_1 = os.path.join(os.environ['DNJF'],'dnjf',task,script_1) 
    script_2 = os.path.join(os.environ['DNJF'],'dnjf',task,script_2) 
    ntasks=get_ntasks(partition)
    sbatch_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=calc.%j.x
#SBATCH --error=calc.%j.x
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks={ntasks}
#SBATCH --partition={partition}
#SBATCH --nodelist={nodelist}

echo "SLRUM_NTASKS: $SLURM_NTASKS"
if [ -z "$SLURM_NTASKS" ] || [ "$SLURM_NTASKS" -le 0 ]; then
    echo "Error: SLRUM_NTASKS is not set or less than or equal to 0"
    exit 1
fi


echo "running {script_1} for {argv_} ..."

python {script_1} {argv_}

echo "{script_1} for {argv_} completed"

echo "now running {script_2} for {argv_} ..."

python {script_2} {argv_}

echo "{script_2} for {argv_} done"

"""
    path = os.environ['RUN']
    job_name = f'{job_name}.sh'
    with open(os.path.join(path, job_name), "w") as sbatch_file:
        sbatch_file.write(sbatch_content)
    print(f"SBATCH file {job_name} created successfully!")
    
    if run:
        os.chdir(path)
        subprocess.run(['sbatch',job_name])

    if return_path:
        return path
    return



def job_with_node(argv_, task, script,  partition, nodelist, job_name=None, output_file=None, return_path=False, run=False):
    if output_file is None:
        script = os.path.join(os.environ['DNJF'],'dnjf',task,script) 
    ntasks=get_ntasks(partition)
    if job_name is None:
        job_name = f'{task}.{os.environ["PBE"]}'
    else:
        job_name = f'{job_name}.{os.environ["PBE"]}'
    sbatch_content = f"""#!/bin/bash

#SBATCH --job-name={job_name}
#SBATCH --output=calc.%j.x
#SBATCH --error=calc.%j.x
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks={ntasks}
#SBATCH --partition={partition}
#SBATCH --nodelist={nodelist}

echo "SLRUM_NTASKS: $SLURM_NTASKS"
if [ -z "$SLURM_NTASKS" ] || [ "$SLURM_NTASKS" -le 0 ]; then
    echo "Error: SLRUM_NTASKS is not set or less than or equal to 0"
    exit 1
fi

echo "running {script} for {argv_} ..."

python {script} {argv_}
"""
    path = os.environ['RUN']
    job_name = f"{job_name}.sh"

    with open(os.path.join(path, job_name), "w") as sbatch_file:
        sbatch_file.write(sbatch_content)
    print(f"SBATCH file {job_name} created successfully!")
    
    if run:
        os.chdir(path)
        subprocess.run(['sbatch',job_name])

    if return_path:
        return path
    return


def vasp_job(system, path, partition, output_file='stdout.x', return_path=True, run=False):
    if output_file is None:
        output_file = f'{system}.vasp.x'
    ntasks=get_ntasks(partition)
    sbatch_content = f"""#!/bin/bash
#SBATCH --job-name={system}.vasp      # Job name
#SBATCH --output={system}.%j.log      # Output file (%j will be replaced with job ID)
#SBATCH --error={system}.%j.log        # Error file (%j will be replaced with job ID)
#SBATCH --nodes=1                                # Number of nodes
#SBATCH --ntasks={ntasks}                             # Total number of tasks (MPI processes)
#SBATCH --time=04:00:00                          # Time limit hrs:min:sec
#SBATCH --partition={partition}		  	 # Partition name (update based on your cluster)

echo "SLURM_NTASKS: $SLURM_NTASKS"

source ~/.bash_profile

if [ -z "$SLURM_NTASKS" ] || [ "$SLURM_NTASKS" -le 0 ]; then
	echo "Error: SLURM_NTASKS is not set or is less than or equal to 0"
	exit 1
fi

mpiexec.hydra -np "$SLURM_NTASKS" /TGM/Apps/VASP/VASP_BIN/6.3.2/vasp.6.3.2.std.x >& stdout.x
"""
    with open(os.path.join(path, 'run.sh'), "w") as sbatch_file:
        sbatch_file.write(sbatch_content)
    print(f"SBATCH file run.sh at {path} created successfully!")
    
    if run:
        os.chdir(path)
        subprocess.run(['sbatch','run.sh'])
    if return_path:
        return path
    return

if __name__ == '__main__':
    print('against the animal farm')
