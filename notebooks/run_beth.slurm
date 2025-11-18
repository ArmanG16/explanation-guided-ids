#!/bin/bash
#SBATCH --job-name=beth_ids_training
#SBATCH --output=beth_ids.out
#SBATCH --error=beth_ids.err
#SBATCH --time=08:00:00
#SBATCH --mem=512G
#SBATCH --cpus-per-task=8
#SBATCH --partition=short

set -x

echo "HOST: $(hostname)"
echo "PWD: $(pwd)"
ls

cd /home/nsewell/explanation-guided-ids
echo "PWD after cd: $(pwd)"
ls

# Confirm python version
/home/nsewell/py312/bin/python --version

# Run your training script
/home/nsewell/py312/bin/python src/Beth_Dataset_Example/train_beth.py
