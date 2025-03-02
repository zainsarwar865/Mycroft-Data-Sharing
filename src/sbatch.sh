#!/bin/bash

#SBATCH --partition=gpuA40x4
#SBATCH --account=bdgs-delta-gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=45g
#SBATCH --time=01:00:00
#SBATCH --output=/u/npatil/dvw/Cleanup/Mycroft-Data-Sharing/newlogs/p%j.%N.MT_stdout
#SBATCH --error=/u/npatil/dvw/Cleanup/Mycroft-Data-Sharing/newlogs/%j.%N.MT_stderr
#SBATCH --job-name=torch
#SBATCH --gpus-per-node=1

source /u/npatil/miniconda3/etc/profile.d/conda.sh
conda activate test


module load cuda/12.4.0

./enola_e2e_dogsvswolves.sh 
