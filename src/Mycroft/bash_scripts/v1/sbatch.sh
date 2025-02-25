#!/bin/bash

#SBATCH --partition=gpuA40x4-preempt
#SBATCH --account=bdgs-delta-gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=00:30:00
#SBATCH --output=/u/zsarwar/logs/p%j.%N.stdout
#SBATCH --error=/u/zsarwar/logs/%j.%N.stderr
#SBATCH --job-name=torch
#SBATCH --gpus-per-node=1

conda init bash
conda activate /u/zsarwar/c_envs/c_enola


module load cuda/12.4.0

./enola_e2e_food101_upmc.sh 

