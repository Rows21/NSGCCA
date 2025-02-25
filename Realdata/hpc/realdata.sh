#!/bin/bash
#
#SBATCH --job-name=sakgcca
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=60GB
#SBATCH --time=80:00:00
#SBATCH --output=output_%A_%a.out
#SBATCH --error=error_%A_%a.err

module purge;
source activate /scratch/rw2867/envs/penv;
cd /scratch/rw2867/projects/SNGCCA/baselines;

if [[ $(lscpu | grep "Intel(R) Xeon(R) Platinum 8268 CPU @ 2.90GHz" | wc -l) -ne 1 ]]; then exit; fi

python /scratch/rw2867/projects/SNGCCA/baselines/sakcca_realdata.py ${SLURM_ARRAY_TASK_ID} $K