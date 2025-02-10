#!/bin/bash
#
#SBATCH --job-name=dgcca
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40GB
#SBATCH --time=10:00:00
#SBATCH --output=output_%A_%a.out
#SBATCH --error=error_%A_%a.err

module purge;
source activate /scratch/rw2867/envs/penv2;
cd /scratch/rw2867/projects/SNGCCA/SNGCCA;

if [[ $(lscpu | grep "Intel(R) Xeon(R) Platinum 8268 CPU @ 2.90GHz" | wc -l) -ne 1 ]]; then exit; fi

python /scratch/rw2867/projects/SNGCCA/SNGCCA/survival_hpc.py ${SLURM_ARRAY_TASK_ID} $K