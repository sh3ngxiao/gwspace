#!/bin/bash
#SBATCH -J test_xuchang
#SBATCH -p cpu_part
#SBATCH -w comput11
#SBATCH -N 1
#SBATCH -n 1

echo "Start time: $(date)"  # show start time
echo "SLURM_JOB_ID: $SLURM_JOB_ID"  # show job id
echo "SLURM_NNODES: $SLURM_NNODES"  # show node count
echo "SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE"  # show tasks per node
echo "SLURM_NTASKS: $SLURM_NTASKS"  # show total tasks
echo "SLURM_JOB_PARTITION: $SLURM_JOB_PARTITION"  # show partition

# export C_INCLUDE_PATH=/usr/include/gsl:$C_INCLUDE_PATH
# export LIBRARY_PATH=/usr/lib64/:$LIBRARY_PATH
python Getforeground.py
echo "End time: $(date)"
