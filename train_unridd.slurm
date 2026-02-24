#!/bin/bash
#SBATCH --job-name=unridd_train
#SBATCH --mem=16G
#SBATCH -c 4
#SBATCH -t 01:00:00
#SBATCH -o logs/train_%j.out
#SBATCH -e logs/train_%j.err

cd ~/explanation-guided-ids
source venv/bin/activate
export PYTHONPATH=$PWD

python src/UNR-IDD/Train_UNR-IDD.py
