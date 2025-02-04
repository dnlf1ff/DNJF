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

def gpu3_jobs(system,task, script_1, script_2, argv_,  partition='gpu3', output_file=None, return_path=False, run=False):
    ntasks=get_ntasks(partition)
    script_1 = os.path.join(os.environ['DNJF'],'dnjf',task, script_1) 
    script_2 = os.path.join(os.environ['DNJF'],'dnjf',task, script_2) 
    
    sbatch_content = f"""#!/bin/bash
#SBATCH --job-name={system}
#SBATCH --output=system.%j.x
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

python {script_1} {argv_}
python {script_2} {argv_}
"""
    path = os.environ['RUN']
    job = f"{system.lower()}.sh"
    with open(os.path.join(path, job), "w") as sbatch_file:
        sbatch_file.write(sbatch_content)
    print(f"SBATCH file {job} created successfully!")
    
    if run:
        os.chdir(path)
        subprocess.run(['sbatch',job]) 
    if return_path:
        return path
    return 


def gpu3_job(system,task, script, argv_,  partition='gpu3', output_file=None, return_path=False, run=False):
    if output_file is None:
        output_file = f'{system}.out.x'
    ntasks=get_ntasks(partition)
    script = os.path.join(os.environ['DNJF'],'dnjf',task, script) 
    
    sbatch_content = f"""#!/bin/bash
#SBATCH --job-name={system}
#SBATCH --output=system.%j.x
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

python {script} {argv_} > {output_file}
"""
    path = os.environ['RUN']
    job = f"{system.lower()}.sh"
    with open(os.path.join(path, job), "w") as sbatch_file:
        sbatch_file.write(sbatch_content)
    print(f"SBATCH file {job} created successfully!")
    
    if run:
        os.chdir(path)
        subprocess.run(['sbatch',job]) 
    if return_path:
        return path
    return 

def python_jobs(system, task, script_1, script_2, argv_,  partition, output_file=None, return_path=False, run=False):
    if output_file is None:
        output_file = f'{system}.out.x'
    script_1 = os.path.join(os.environ['DNJF'],'dnjf',task,script_1) 
    script_2 = os.path.join(os.environ['DNJF'],'dnjf',task,script_2) 
    ntasks=get_ntasks(partition)
    
    sbatch_content = f"""#!/bin/bash
#SBATCH --job-name={system}
#SBATCH --output=calc.%j.x
#SBATCH --error=calc.%j.x
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks={ntasks}
#SBATCH --partition={partition}

echo "SLRUM_NTASKS: $SLURM_NTASKS"
if [ -z "$SLURM_NTASKS" ] || [ "$SLURM_NTASKS" -le 0 ]; then
    echo "Error: SLRUM_NTASKS is not set or less than or equal to 0"
    exit 1
fi

python {script_1} {argv_}

python {script_2} {argv_}
"""
    path = os.environ['RUN']
    job = f"{system.lower()}.sh"

    with open(os.path.join(path, job), "w") as sbatch_file:
        sbatch_file.write(sbatch_content)
    print(f"SBATCH file {job} created successfully!")
    
    if run:
        os.chdir(path)
        subprocess.run(['sbatch',job])

    if return_path:
        return path
    return

def jobs_with_node(system, task, script_1, script_2, argv_,  partition, nodelist, output_file=None, return_path=False, run=False):
    if output_file is None:
        output_file = f'{system}.out.x'
    script_1 = os.path.join(os.environ['DNJF'],'dnjf',task,script_1) 
    script_2 = os.path.join(os.environ['DNJF'],'dnjf',task,script_2) 
    ntasks=get_ntasks(partition)
    
    sbatch_content = f"""#!/bin/bash
#SBATCH --job-name={system}
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

python {script_1} {argv_}
python {script_2} {argv_}
"""
    path = os.environ['RUN']
    job = f"{system.lower()}.sh"

    with open(os.path.join(path, job), "w") as sbatch_file:
        sbatch_file.write(sbatch_content)
    print(f"SBATCH file {job} created successfully!")
    
    if run:
        os.chdir(path)
        subprocess.run(['sbatch',job])

    if return_path:
        return path
    return



def job_with_node(system, task, script, argv_,  partition, nodelist, output_file=None, return_path=False, run=False):
    if output_file is None:
        output_file = f'{system}.out.x'
    script = os.path.join(os.environ['DNJF'],'dnjf',task,script) 
    ntasks=get_ntasks(partition)
    
    sbatch_content = f"""#!/bin/bash
#SBATCH --job-name={system}
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

python {script} {argv_}
"""
    path = os.environ['RUN']
    job = f"{system.lower()}.sh"

    with open(os.path.join(path, job), "w") as sbatch_file:
        sbatch_file.write(sbatch_content)
    print(f"SBATCH file {job} created successfully!")
    
    if run:
        os.chdir(path)
        subprocess.run(['sbatch',job])

    if return_path:
        return path
    return



def python_job(system, task, script, argv_,  partition, output_file=None, return_path=False, run=False):
    if output_file is None:
        output_file = f'{system}.out.x'
    script = os.path.join(os.environ['DNJF'],'dnjf',task,script) 
    ntasks=get_ntasks(partition)
    
    sbatch_content = f"""#!/bin/bash
#SBATCH --job-name={system}
#SBATCH --output=calc.%j.x
#SBATCH --error=calc.%j.x
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks={ntasks}
#SBATCH --partition={partition}

echo "SLRUM_NTASKS: $SLURM_NTASKS"
if [ -z "$SLURM_NTASKS" ] || [ "$SLURM_NTASKS" -le 0 ]; then
    echo "Error: SLRUM_NTASKS is not set or less than or equal to 0"
    exit 1
fi

python {script} {argv_}
"""
    path = os.environ['RUN']
    job = f"{system.lower()}.sh"

    with open(os.path.join(path, job), "w") as sbatch_file:
        sbatch_file.write(sbatch_content)
    print(f"SBATCH file {job} created successfully!")
    
    if run:
        os.chdir(path)
        subprocess.run(['sbatch',job])

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
