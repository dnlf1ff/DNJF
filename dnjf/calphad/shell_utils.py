import os
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

def create_sbatch(job_name, script, argv_1, time_limit='4:00:00', partition='loki4', output_file=None):
    if output_file is None:
        output_file = f'{job_name}.out.x'
    ntasks=get_ntasks(partition)
    sbatch_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=calc.%j.x
#SBATCH --error=calc.%j.x
#SBATCH --time={time_limit}
#SBATCH --nodes=1
#SBATCH --ntasks={ntasks}
#SBATCH --partition={partition}

echo "SLRUM_NTASKS: $SLURM_NTASKS"
if [ -z "$SLURM_NTASKS" ] || [ "$SLURM_NTASKS" -le 0 ]; then
    echo "Error: SLRUM_NTASKS is not set or less than or equal to 0"
    exit 1
fi

python {script} {argv_1} > {argv_1}.out.x
"""
    job = f"run-{job_name}.sh"
    with open(os.path.join("run", f"run-{job_name}.sh"), "w") as sbatch_file:
        sbatch_file.write(sbatch_content)
    print(f"SBATCH file 'run-{job_name}.sh' created successfully!")
    return job 


def create_sbatch2(job_name, script, argv_1, argv_2, time_limit='4:00:00', partition='loki4', output_file=None):
    if output_file is None:
        output_file = f'{job_name}.out.x'
    ntasks=get_ntasks(partition)
    sbatch_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=calc.%j.x
#SBATCH --error=calc.%j.x
#SBATCH --time={time_limit}
#SBATCH --nodes=1
#SBATCH --ntasks={ntasks}
#SBATCH --partition={partition}

echo "SLRUM_NTASKS: $SLURM_NTASKS"
if [ -z "$SLURM_NTASKS" ] || [ "$SLURM_NTASKS" -le 0 ]; then
    echo "Error: SLRUM_NTASKS is not set or less than or equal to 0"
    exit 1
fi

python {script} {argv_1} {argv_2} 
"""
    job = f"run-{job_name}.sh"
    with open(os.path.join("run", f"run-{job_name}.sh"), "w") as sbatch_file:
        sbatch_file.write(sbatch_content)
    print(f"SBATCH file 'run-{job_name}.sh' created successfully!")
    return job 


