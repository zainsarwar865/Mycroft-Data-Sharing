#!/bin/bash
#SBATCH --partition=gpuA40x4
#SBATCH --account=bdgs-delta-gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=50g
#SBATCH --time=02:00:00
#SBATCH --output=/u/npatil/dvw/Cleanup/Mycroft-Data-Sharing/newlogs/p%j.%N.AUG_stdout
#SBATCH --error=/u/npatil/dvw/Cleanup/Mycroft-Data-Sharing/newlogs/%j.%N.AUG_stderr
#SBATCH --job-name=torch
#SBATCH --gpus-per-node=1

conda activate /u/npatil/miniconda3/envs/test

bash /u/npatil/dvw/Cleanup/Mycroft-Data-Sharing/src/enola_e2e_dogsvswolves.sh

conda deactivate