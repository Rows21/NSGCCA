#!/bin/bash
#
#SBATCH --job-name=dgcca
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10GB
#SBATCH --time=1:00:00
#SBATCH --output=output_%A_%a.out
#SBATCH --error=error_%A_%a.err

module purge;
module load r/gcc/4.1.2;
cd /scratch/rw2867/projects/SNGCCA/OtherMethods/hpcres
mode=1
N=100
num=5
tol=100

if [[ $(lscpu | grep "Intel(R) Xeon(R) Platinum 8268 CPU @ 2.90GHz" | wc -l) -ne 1 ]]; then exit; fi

Rscript /scratch/rw2867/projects/SNGCCA/OtherMethods/rgcca.R ${SLURM_ARRAY_TASK_ID} $mode $N $num $tol


