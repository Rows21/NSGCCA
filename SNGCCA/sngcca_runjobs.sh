#!/bin/bash
#
#SBATCH --job-name=siml11
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10GB
#SBATCH --time=1:00:00
#SBATCH --output=output_%A_%a.out
#SBATCH --error=error_%A_%a.err

module purge;
source /scratch/rw2867/envs/penv/bin/activate;
cd /scratch/rw2867/projects/SNGCCA/SNGCCA
data_files=$(sed -n "${SLURM_ARRAY_TASK_ID}p" /scratch/rw2867/projects/SNGCCA/SNGCCA/Data/file_list.txt)

# 使用空格分隔获取文件路径
data_file1=$(echo $data_files | cut -d' ' -f1)
data_file2=$(echo $data_files | cut -d' ' -f2)
data_file3=$(echo $data_files | cut -d' ' -f3)

if [[ $(lscpu | grep "Intel(R) Xeon(R) Platinum 8268 CPU @ 2.90GHz" | wc -l) -ne 1 ]]; then exit; fi

python /scratch/rw2867/projects/SNGCCA/SNGCCA/test.py ${SLURM_ARRAY_TASK_ID} $data_file1 $data_file2 $data_file3


