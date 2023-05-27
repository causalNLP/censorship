#!/bin/bash
#SBATCH -n12
#SBATCH --job-name=finetuning
#SBATCH -N1
#SBATCH -p DGX
#SBATCH --gpus=8
#SBATCH --mem=200gb
#SBATCH --time=48:00:00
#SBATCH --output=my_job_%j.out
#SBATCH --error=my_job_%j.err

conda activate torch
echo $SLURM_JOB_ID
python train.py -v --batch-size 24 --epochs 25 --model-path models/${SLURM_JOB_ID}_model  --GPUs 8
```