#!/bin/bash
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -J "Train on USNW-NB15"
#SBATCH -p short
#SBATCH -t 24:00:00
#SBATCH --mem=256g


echo "Training on USNW-NB15..."
source /home/csgilbert/explanation-guided-ids/venv/bin/activate
python -u src/USNW-NB15/run_USNW_pyIDS.py

