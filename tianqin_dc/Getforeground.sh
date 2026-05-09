#!/bin/bash
#SBATCH -J test_zhuangzhenye
#SBATCH -p cpu_part
#SBATCH -w comput11
#SBATCH -N 1
#SBATCH -n 1
echo "Start time: `date`"   #显示开始时间
echo "SLURM_JOB_ID: $SLURM_JOB_ID"   #显示作业号
echo "SLURM_NNODES: $SLURM_NNODES"   #显示节点数
echo "SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE"  #显示每节点任务数
echo "SLURM_NTASKS: $SLURM_NTASKS"   #显示总任务数
echo "SLURM_JOB_PARTITION: $SLURM_JOB_PARTITION"   #显示作业分区

# export C_INCLUDE_PATH=/usr/include/gsl:$C_INCLUDE_PATH
# export LIBRARY_PATH=/usr/lib64/:$LIBRARY_PATH
python Getforeground.py
echo "End time: `date`"  
