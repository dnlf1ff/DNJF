#!/bin/bash
#SBATCH --job-name=strangling    # Job name
#SBATCH --output=calc_%j.log      # Output file (%j will be replaced with job ID)
#SBATCH --error=calc_%j.log        # Error file (%j will be replaced with job ID)
#SBATCH --nodes=1                                # Number of nodes
#SBATCH --ntasks=28                      # Total number of tasks (MPI processes)
#SBATCH --time=04:00:00                          # Time limit hrs:min:sec
#SBATCH --partition=loki4          # Partition name (update based on your cluster)

source ~/.bash_profile

echo "SLURM_NTASKS: $SLURM_NTASKS"

if [ -z "$SLURM_NTASKS" ] || [ "$SLURM_NTASKS" -le 0 ]; then
    echo "Error: SLURM_NTASKS is not set or is less than or equal to 0"
    exit 1
fi

volumes=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14")
    
for vol in "${volumes[@]}"; do
    if [ -d $vol ]; then
        cd $vol
        echo "calculating $vol % volume"
        mpiexec.hydra -np "$SLURM_NTASKS" /TGM/Apps/VASP/VASP_BIN/6.3.2/vasp.6.3.2.std.x >& stdout.x        
    fi
    cd ..
done
