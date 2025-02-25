#!/bin/bash
#SBATCH --partition=gpuA40x4
#SBATCH --account=bdgs-delta-gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=00:30:00
#SBATCH --output=/projects/bdgs/zsarwar/logs/p%j.%N.stdout
#SBATCH --error=/projects/bdgs/zsarwar/logs/%j.%N.stderr
#SBATCH --job-name=torch
#SBATCH --gpus-per-node=1


conda activate /u/zsarwar/c_envs/c_enola

bash /projects/bdgs/zsarwar/Mycroft-Data-Sharing/src/Mycroft/bash_scripts/v1/enola_e2e_dogsvswolves.sh

conda deactivate